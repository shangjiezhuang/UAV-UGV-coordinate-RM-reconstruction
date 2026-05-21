from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


POLY_DIM = 6


def poly_features_from_diff(diff: torch.Tensor) -> torch.Tensor:
    """Build [1, dx, dy, dx^2, dx*dy, dy^2] features."""
    dx = diff[..., 0]
    dy = diff[..., 1]
    ones = torch.ones_like(dx)
    return torch.stack((ones, dx, dy, dx * dx, dx * dy, dy * dy), dim=-1)


def epanechnikov_kernel(dists: torch.Tensor, h: float) -> torch.Tensor:
    if h <= 0:
        raise ValueError(f"kernel_bandwidth must be positive, got {h}.")
    u = dists / h
    weights = 0.75 * (1.0 - u.pow(2))
    return torch.where(torch.abs(u) <= 1.0, weights, torch.zeros_like(weights))


def make_grid_norm(n1: int, n2: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return normalized grid coordinates with shape (n1*n2, 2) in [-1, 1]."""
    xs = torch.linspace(-1.0, 1.0, n1, device=device, dtype=dtype)
    ys = torch.linspace(-1.0, 1.0, n2, device=device, dtype=dtype)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)


def append_observation(
    obs: Optional[dict],
    z_new_norm: torch.Tensor,
    gamma_new: torch.Tensor,
    omega_new: torch.Tensor,
    grid_norm: torch.Tensor,
    grid_size: tuple[int, int],
    kernel_bandwidth: float,
    I_flat: Optional[torch.Tensor] = None,
    sample_grid_idx: Optional[torch.Tensor] = None,
) -> dict:
    """
    Append one or more UAV observations and rebuild GPU-side kernel caches.

    z_new_norm:      (B, 2)
    gamma_new:       (B, K)
    omega_new:       (B, K)
    sample_grid_idx: (B,), optional linear grid indices for observation loss
    """
    n1, n2 = grid_size
    device = grid_norm.device
    dtype = grid_norm.dtype
    n_grid = n1 * n2

    z_new_norm = z_new_norm.to(device=device, dtype=dtype).reshape(-1, 2)
    gamma_new = gamma_new.to(device=device, dtype=dtype).reshape(z_new_norm.shape[0], -1)
    omega_new = omega_new.to(device=device, dtype=dtype).reshape_as(gamma_new)

    if obs is None:
        locs_norm = z_new_norm
        gamma = gamma_new
        omega = omega_new
        if sample_grid_idx is None:
            all_sample_idx = None
        else:
            all_sample_idx = sample_grid_idx.to(device=device, dtype=torch.long).reshape(-1)
    else:
        locs_norm = torch.cat((obs["locs_norm"], z_new_norm), dim=0)
        gamma = torch.cat((obs["Gamma"], gamma_new), dim=0)
        omega = torch.cat((obs["Omega"], omega_new), dim=0)
        if sample_grid_idx is None:
            all_sample_idx = obs.get("sample_grid_idx")
        else:
            new_idx = sample_grid_idx.to(device=device, dtype=torch.long).reshape(-1)
            old_idx = obs.get("sample_grid_idx")
            all_sample_idx = new_idx if old_idx is None else torch.cat((old_idx, new_idx), dim=0)

    new_dists = torch.cdist(grid_norm, z_new_norm)
    new_weights = epanechnikov_kernel(new_dists, kernel_bandwidth)
    if obs is None:
        weights = new_weights
    else:
        weights = torch.cat((obs["Weights"], new_weights), dim=1)

    affected_flat = torch.any(new_dists / kernel_bandwidth < 1.0, dim=1).to(dtype=dtype)
    affected_idx = torch.where(affected_flat > 0.5)[0]
    affected_mask = affected_flat.reshape(n1, n2)

    if I_flat is None and obs is not None and "I_flat" in obs:
        i_flat_t = obs["I_flat"].to(device=device, dtype=torch.bool).reshape(-1)
    elif I_flat is None:
        i_flat_t = torch.ones(n_grid, device=device, dtype=torch.bool)
    else:
        i_flat_t = I_flat.to(device=device, dtype=torch.bool).reshape(-1)
    if i_flat_t.numel() != n_grid:
        raise ValueError(f"I_flat must contain {n_grid} values, got {i_flat_t.numel()}.")

    out = {
        "locs_norm": locs_norm,
        "Gamma": gamma,
        "Omega": omega,
        "Weights": weights,
        "affected_idx": affected_idx,
        "affected_mask": affected_mask,
        "I_flat": i_flat_t,
    }
    if all_sample_idx is not None:
        out["sample_grid_idx"] = all_sample_idx
    return out


class UnfoldingThetaLayer(nn.Module):
    """
    R=1 GPU batched WLS update matching II_BTD_Opt_GPU._update_theta.

    Theta layout is (N_grid, 1, POLY_DIM).
    """

    def __init__(self, nu: float, ridge: float = 1e-7):
        super().__init__()
        self.nu = float(nu)
        self.ridge = float(ridge)

    def forward(
        self,
        theta_old: torch.Tensor,
        phi: torch.Tensor,
        sr: torch.Tensor,
        obs: dict,
        grid: dict,
    ) -> torch.Tensor:
        if theta_old.ndim != 3 or theta_old.shape[1:] != (1, POLY_DIM):
            raise ValueError(f"R=1 Theta must have shape (N_grid, 1, {POLY_DIM}), got {tuple(theta_old.shape)}.")
        if phi.ndim != 2 or phi.shape[0] != 1:
            raise ValueError(f"R=1 Phi must have shape (1, K), got {tuple(phi.shape)}.")

        grid_norm = grid["grid_norm"]
        locs_norm = obs["locs_norm"]
        gamma = obs["Gamma"]
        omega = obs["Omega"]
        weights_raw = obs["Weights"]
        i_flat = obs["I_flat"].to(device=theta_old.device, dtype=torch.bool).reshape(-1)
        grid_idx = obs.get("affected_idx")
        if grid_idx is None:
            grid_idx = torch.arange(theta_old.shape[0], device=theta_old.device, dtype=torch.long)
        else:
            grid_idx = grid_idx.to(device=theta_old.device, dtype=torch.long).reshape(-1)

        if locs_norm.numel() == 0 or grid_idx.numel() == 0:
            return theta_old

        valid_mask = i_flat.index_select(0, grid_idx)
        candidate_idx = grid_idx[valid_mask]
        if candidate_idx.numel() == 0:
            return theta_old

        weights_sel = weights_raw.index_select(0, candidate_idx)
        active_mask = torch.any(weights_sel > 1e-6, dim=1)
        valid_idx = candidate_idx[active_mask]
        if valid_idx.numel() == 0:
            return theta_old

        weights_sel = weights_sel[active_mask]
        n2 = int(sr.shape[-1])
        phi_vec = phi[0]
        phi_om_phi = torch.einsum("k,mk,k->m", phi_vec, omega, phi_vec)
        phi_gam = torch.einsum("k,mk->m", phi_vec, gamma * omega)
        theta_new = theta_old.clone()

        grid_sel = grid_norm.index_select(0, valid_idx)
        diff = locs_norm.unsqueeze(0) - grid_sel.unsqueeze(1)
        x_feat = poly_features_from_diff(diff)

        coeff = weights_sel * phi_om_phi.unsqueeze(0)
        ata = torch.einsum("gm,gmi,gmj->gij", coeff, x_feat, x_feat)
        ata[:, 0, 0] = ata[:, 0, 0] + self.nu

        coeff_b = weights_sel * phi_gam.unsqueeze(0)
        atb = torch.einsum("gm,gmi->gi", coeff_b, x_feat)
        i_g = torch.div(valid_idx, n2, rounding_mode="floor")
        j_g = torch.remainder(valid_idx, n2)
        atb[:, 0] = atb[:, 0] + self.nu * sr[0, i_g, j_g]

        eye = torch.eye(POLY_DIM, dtype=theta_old.dtype, device=theta_old.device) * self.ridge
        system = ata + eye.unsqueeze(0)
        try:
            theta = torch.linalg.solve(system, atb.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            theta = torch.linalg.lstsq(system, atb.unsqueeze(-1)).solution.squeeze(-1)
        theta_new[valid_idx, 0, :] = theta

        return theta_new


class UnfoldingPhiLayer(nn.Module):
    """R=1 nonnegative closed-form Phi update with a learnable blend."""

    def __init__(self, K: int, normalize_phi: bool = True):
        super().__init__()
        self.K = int(K)
        self.normalize_phi = bool(normalize_phi)
        self.raw_blend = nn.Parameter(torch.tensor(2.0))

    def forward(self, phi_old: torch.Tensor, theta: torch.Tensor, obs: dict, grid: dict) -> torch.Tensor:
        if phi_old.shape != (1, self.K):
            raise ValueError(f"R=1 Phi must have shape (1, {self.K}), got {tuple(phi_old.shape)}.")

        grid_norm = grid["grid_norm"]
        locs_norm = obs["locs_norm"]
        gamma = obs["Gamma"]
        omega = obs["Omega"]
        weights_raw = obs["Weights"]
        i_flat = obs["I_flat"].to(device=phi_old.device, dtype=torch.bool).reshape(-1)

        grid_idx = torch.where(i_flat)[0]
        if locs_norm.numel() == 0 or grid_idx.numel() == 0:
            return phi_old

        weights_sel = weights_raw.index_select(0, grid_idx)
        active = torch.any(weights_sel > 1e-6, dim=1)
        if not bool(torch.any(active)):
            return phi_old

        grid_idx = grid_idx[active]
        weights_sel = weights_sel[active]
        grid_sel = grid_norm.index_select(0, grid_idx)
        diff = locs_norm.unsqueeze(0) - grid_sel.unsqueeze(1)
        x_feat = poly_features_from_diff(diff)
        theta_sel = theta.index_select(0, grid_idx)[:, 0, :]
        pred = torch.einsum("gmd,gd->gm", x_feat, theta_sel)

        weighted_pred_sum = torch.sum(weights_sel * pred, dim=0)
        weighted_pred_sq_sum = torch.sum(weights_sel * pred.square(), dim=0)
        gamma_obs = gamma * omega
        numerator = torch.sum(weighted_pred_sum.unsqueeze(1) * gamma_obs, dim=0)
        denominator = torch.sum(weighted_pred_sq_sum.unsqueeze(1) * omega, dim=0)
        phi_closed = torch.clamp(numerator / denominator.clamp_min(1e-12), min=0.0)

        if self.normalize_phi:
            phi_closed = self.K * phi_closed / phi_closed.sum().clamp_min(1e-8)

        blend = torch.sigmoid(self.raw_blend)
        phi_new = (1.0 - blend) * phi_old[0] + blend * phi_closed
        phi_new = torch.clamp(phi_new, min=0.0)
        if self.normalize_phi:
            phi_new = self.K * phi_new / phi_new.sum().clamp_min(1e-8)
        return phi_new.unsqueeze(0)


class UnfoldingSrLayer(nn.Module):
    """Learned proximal-gradient Sr update for R=1."""

    def __init__(self, grid_size: tuple[int, int], hidden: int = 32, local_update: bool = False):
        super().__init__()
        self.N1, self.N2 = int(grid_size[0]), int(grid_size[1])
        hidden = int(hidden)
        self.local_update = bool(local_update)
        self.net = nn.Sequential(
            nn.Conv2d(4, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )
        self.raw_eta_s = nn.Parameter(torch.tensor(0.0))

    def forward(self, sr_old: torch.Tensor, theta: torch.Tensor, obs: dict) -> torch.Tensor:
        psi = theta[:, 0, 0].reshape(self.N1, self.N2)
        s_old = sr_old[0]
        affected_mask = obs["affected_mask"].to(device=sr_old.device, dtype=sr_old.dtype)
        i_mask = obs["I_flat"].to(device=sr_old.device, dtype=sr_old.dtype).reshape(self.N1, self.N2)

        data_mask = affected_mask * i_mask
        update_mask = data_mask if self.local_update else i_mask
        eta_s = torch.sigmoid(self.raw_eta_s)
        y = s_old - eta_s * data_mask * (s_old - psi)

        x = torch.stack((y, psi, affected_mask, i_mask), dim=0).unsqueeze(0)
        candidate = F.softplus(self.net(x).squeeze(0).squeeze(0))
        s_new = update_mask * candidate + (1.0 - update_mask) * s_old
        return s_new.unsqueeze(0)


class UnfoldingLayer(nn.Module):
    def __init__(
        self,
        M: int,
        N: int,
        K: int,
        nu: float,
        hidden: int,
        local_sr_update: bool,
    ):
        super().__init__()
        self.theta_layer = UnfoldingThetaLayer(nu=nu)
        self.phi_layer = UnfoldingPhiLayer(K=K)
        self.sr_layer = UnfoldingSrLayer((M, N), hidden=hidden, local_update=local_sr_update)

    def forward(self, state: dict, obs: dict, grid: dict) -> dict:
        theta_new = self.theta_layer(state["Theta"], state["Phi"], state["Sr"], obs, grid)
        phi_new = self.phi_layer(state["Phi"], theta_new, obs, grid)
        sr_new = self.sr_layer(state["Sr"], theta_new, obs)
        h_hat_new = torch.einsum("rxy,rk->xyk", sr_new, phi_new)
        return {"Theta": theta_new, "Phi": phi_new, "Sr": sr_new, "H_hat": h_hat_new}


class DU_IIBTD(nn.Module):
    """
    Deep-unfolded II-BTD for the current RadioSeerDPM R=1 setting.

    State:
        Theta: (N_grid, 1, POLY_DIM)
        Phi:   (1, K)
        Sr:    (1, M, N)
        H_hat: (M, N, K)
    """

    def __init__(
        self,
        M: int,
        N: int,
        K: int,
        T: int = 2,
        nu: float = 1.0,
        hidden: int = 32,
        local_sr_update: bool = False,
    ):
        super().__init__()
        self.M, self.N, self.K, self.R, self.T = int(M), int(N), int(K), 1, int(T)
        hidden = int(hidden)
        self.dim_poly = POLY_DIM
        self.nu = float(nu)
        self.unfolding_layers = nn.ModuleList(
            [
                UnfoldingLayer(
                    M=self.M,
                    N=self.N,
                    K=self.K,
                    nu=self.nu,
                    hidden=hidden,
                    local_sr_update=local_sr_update,
                )
                for _ in range(self.T)
            ]
        )

    def init_state(self, *, device=None, dtype=torch.float32) -> dict:
        n_grid = self.M * self.N
        theta = torch.zeros((n_grid, 1, self.dim_poly), device=device, dtype=dtype)
        phi = torch.ones((1, self.K), device=device, dtype=dtype)
        phi = self.K * phi / phi.sum(dim=1, keepdim=True).clamp_min(1e-8)
        sr = torch.zeros((1, self.M, self.N), device=device, dtype=dtype)
        h_hat = torch.einsum("rxy,rk->xyk", sr, phi)
        return {"Theta": theta, "Phi": phi, "Sr": sr, "H_hat": h_hat}

    def forward(self, state: dict, obs: dict, grid: dict) -> dict:
        for layer in self.unfolding_layers:
            state = layer(state, obs, grid)
        return state
