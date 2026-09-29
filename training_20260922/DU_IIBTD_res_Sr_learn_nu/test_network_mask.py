"""Coverage context must not change the observation or spatial writeback masks."""
import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from .DU_IIBTD import UnfoldingSrLayer


class CaptureMask(nn.Module):
    def forward(self, x):
        self.input = x.detach().clone()
        return x[:, 2:3]


class NetworkMaskTests(unittest.TestCase):
    def fixture(self, mode='global', dtype=torch.float32, device='cpu'):
        layer = UnfoldingSrLayer((2, 3), sr_update_mode=mode,
                                 global_update_weight=0.2).to(device=device, dtype=dtype)
        layer.net = CaptureMask()
        obs = dict(affected_mask=torch.tensor([[0, 1, 0], [0, 0, 0]], dtype=dtype, device=device),
                   affected_idx=torch.tensor([1], device=device),
                   Weights=torch.tensor([[1, 0], [0, 1], [0, 0], [0.4, 0], [0, 0], [0, 0]],
                                        dtype=dtype, device=device),
                   I_flat=torch.tensor([1, 1, 1, 1, 1, 0], device=device).bool())
        return layer, torch.full((1, 2, 3), 2., device=device, dtype=dtype), torch.zeros((6, 1, 1), device=device, dtype=dtype), obs

    def test_global_context_retains_old_coverage_but_not_unobserved_cells(self):
        for dtype in (torch.float32, torch.float64):
            layer, old, theta, obs = self.fixture(dtype=dtype)
            before = copy.deepcopy(obs)
            output = layer(old, theta, obs)
            expected = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=dtype)
            torch.testing.assert_close(layer.net.input[0, 2], expected, rtol=0, atol=0)
            for key in obs:
                torch.testing.assert_close(obs[key], before[key], rtol=0, atol=0)
            # An uncovered valid cell is still updated by a global model.
            self.assertNotEqual(output[0, 0, 2].item(), old[0, 0, 2].item())
            # Invalid cells retain the previous map, even in global mode.
            self.assertEqual(output[0, 1, 2].item(), old[0, 1, 2].item())

    def test_local_retains_input_and_restricts_writeback(self):
        layer, old, theta, obs = self.fixture('local')
        del obs['Weights']  # Other modes do not acquire a new input dependency.
        output = layer(old, theta, obs)
        torch.testing.assert_close(layer.net.input[0, 2], obs['affected_mask'], rtol=0, atol=0)
        self.assertNotEqual(output[0, 0, 1].item(), old[0, 0, 1].item())
        keep = ~obs['affected_mask'].bool()
        torch.testing.assert_close(output[0][keep], old[0][keep], rtol=0, atol=0)

    def test_soft_global_retains_original_blending(self):
        layer, old, theta, obs = self.fixture('soft_global')
        del obs['Weights']
        output = layer(old, theta, obs)
        torch.testing.assert_close(layer.net.input[0, 2], obs['affected_mask'], rtol=0, atol=0)
        self.assertAlmostEqual(output[0, 0, 0].item(), 0.2 * F.softplus(torch.tensor(0.)).item() + 0.8 * 2.)
        self.assertEqual(output[0, 1, 2].item(), 2.)

    def test_empty_coverage_is_not_replaced_with_all_ones(self):
        layer, old, theta, obs = self.fixture()
        obs['Weights'] = torch.zeros(6, 0)
        layer(old, theta, obs)
        self.assertEqual(torch.count_nonzero(layer.net.input[0, 2]).item(), 0)

    def test_all_sample_batch_matches_original_input(self):
        layer, old, theta, obs = self.fixture()
        obs['affected_mask'] = torch.any(obs['Weights'] > 0, dim=1).float().reshape(2, 3)
        layer(old, theta, obs)
        torch.testing.assert_close(layer.net.input[0, 2], obs['affected_mask'], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_context_and_output_stay_on_model_device(self):
        layer, old, theta, obs = self.fixture(device='cuda:0')
        output = layer(old, theta, obs)
        self.assertEqual(output.device, old.device)
        self.assertEqual(layer.net.input.device, old.device)
        self.assertEqual(layer.net.input[0, 2].sum().item(), 3)


if __name__ == '__main__':
    unittest.main()
