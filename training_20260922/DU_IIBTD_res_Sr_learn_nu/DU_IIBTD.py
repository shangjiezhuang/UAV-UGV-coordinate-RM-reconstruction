from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


POLY_DIM = 6


def _stable_batched_ridge_solve(
    ata: torch.Tensor,
    atb: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Solve batched local WLS systems with a narrow float64 fallback.

    The ordinary path is identical to ``torch.linalg.solve`` in the model's
    native dtype. Only batches reported singular/non-finite are reconstructed
    from the unregularized ``ata`` in float64 so a small configured ridge is not
    rounded away next to large float32 entries. A least-squares fallback is used
    only if that double-precision ridge solve still reports singularity.
    """
    if ata.ndim != 3 or atb.ndim != 2:
        raise ValueError(
            f"Expected ata=(B,D,D), atb=(B,D), got {tuple(ata.shape)} and {tuple(atb.shape)}."
        )
    if ata.shape[0] != atb.shape[0] or ata.shape[1] != ata.shape[2] or ata.shape[1] != atb.shape[1]:
        raise ValueError(
            f"Incompatible batched solve shapes: ata={tuple(ata.shape)}, atb={tuple(atb.shape)}."
        )
    ridge = float(ridge)
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError(f"ridge must be finite and positive, got {ridge}.")

    dim = int(ata.shape[-1])
    eye = torch.eye(dim, dtype=ata.dtype, device=ata.device)
    rhs = atb.unsqueeze(-1)
    solution, info = torch.linalg.solve_ex(
        ata + eye.unsqueeze(0) * ridge,
        rhs,
        check_errors=False,
    )
    finite_solution = torch.isfinite(solution).all(dim=(-2, -1))
    bad = info.ne(0) | ~finite_solution
    if not bool(torch.any(bad)):
        return solution.squeeze(-1)

    bad_ata = ata[bad]
    bad_rhs = rhs[bad]
    if not bool(torch.isfinite(bad_ata).all() and torch.isfinite(bad_rhs).all()):
        raise RuntimeError("Theta WLS system contains non-finite coefficients.")

    eye64 = torch.eye(dim, dtype=torch.float64, device=ata.device)
    system64 = bad_ata.to(torch.float64) + eye64.unsqueeze(0) * ridge
    rhs64 = bad_rhs.to(torch.float64)
    solution64, info64 = torch.linalg.solve_ex(
        system64,
        rhs64,
        check_errors=False,
    )
    bad64 = info64.ne(0) | ~torch.isfinite(solution64).all(dim=(-2, -1))
    if bool(torch.any(bad64)):
        solution64 = solution64.clone()
        solution64[bad64] = torch.linalg.lstsq(
            system64[bad64],
            rhs64[bad64],
            rcond=1e-12,
        ).solution
    if not bool(torch.isfinite(solution64).all()):
        raise RuntimeError("Theta WLS stable fallback produced non-finite values.")

    solution = solution.clone()
    solution[bad] = solution64.to(dtype=solution.dtype)
    return solution.squeeze(-1)


def inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-12)
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


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

    def __init__(
        self,
        nu: float,
        theta_chunk_size: int = 4096,
        theta_ridge: float = 1e-5,
        learn_nu: bool = True,
        min_nu: float = 1e-8,
    ):
        super().__init__()
        self.theta_chunk_size = int(theta_chunk_size)
        self.theta_ridge = float(theta_ridge)
        self.learn_nu = bool(learn_nu)
        self.min_nu = float(min_nu)
        raw_nu_init = inverse_softplus(max(float(nu) - self.min_nu, 1e-12))
        raw_nu = torch.tensor(raw_nu_init, dtype=torch.float32)
        if self.learn_nu:
            self.raw_nu = nn.Parameter(raw_nu)
        else:
            self.register_buffer("raw_nu", raw_nu)

    def nu_value(self) -> torch.Tensor:
        return F.softplus(self.raw_nu) + self.min_nu

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
        nu = self.nu_value().to(device=theta_old.device, dtype=theta_old.dtype)

        chunk_size = max(1, self.theta_chunk_size)
        for start in range(0, int(valid_idx.numel()), chunk_size):
            end = min(start + chunk_size, int(valid_idx.numel()))
            idx = valid_idx[start:end]
            w = weights_sel[start:end]
            grid_sel = grid_norm.index_select(0, idx)
            diff = locs_norm.unsqueeze(0) - grid_sel.unsqueeze(1)
            x_feat = poly_features_from_diff(diff)

            coeff = w * phi_om_phi.unsqueeze(0)
            ata = torch.einsum("gm,gmi,gmj->gij", coeff, x_feat, x_feat)
            ata[:, 0, 0] = ata[:, 0, 0] + nu

            coeff_b = w * phi_gam.unsqueeze(0)
            atb = torch.einsum("gm,gmi->gi", coeff_b, x_feat)
            i_g = torch.div(idx, n2, rounding_mode="floor")
            j_g = torch.remainder(idx, n2)
            atb[:, 0] = atb[:, 0] + nu * sr[0, i_g, j_g]

            theta = _stable_batched_ridge_solve(ata, atb, self.theta_ridge)
            theta_new[idx, 0, :] = theta

        return theta_new


class UnfoldingPhiLayer(nn.Module):
    """R=1 nonnegative closed-form Phi update with a learnable blend."""

    def __init__(
        self,
        K: int,
        normalize_phi: bool = True,
    ):
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


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.silu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return F.silu(x + y)


class UnfoldingSrLayer(nn.Module):
    """Residual-network proximal Sr update for R=1."""

    def __init__(
        self,
        grid_size: tuple[int, int],
        hidden: int = 32,
        sr_update_mode: str = "local",
        global_update_weight: float = 0.1,
    ):
        super().__init__()
        self.N1, self.N2 = int(grid_size[0]), int(grid_size[1])
        hidden = int(hidden)
        if sr_update_mode not in {"local", "global", "soft_global"}:
            raise ValueError(
                "sr_update_mode must be one of {'local', 'global', 'soft_global'}, "
                f"got {sr_update_mode!r}."
            )
        self.sr_update_mode = str(sr_update_mode)
        self.global_update_weight = float(global_update_weight)
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )
        self.raw_scale = nn.Parameter(torch.tensor(-3.0))

    def forward(self, sr_old: torch.Tensor, theta: torch.Tensor, obs: dict) -> torch.Tensor:
        psi = theta[:, 0, 0].reshape(self.N1, self.N2)
        s_old = sr_old[0]
        affected_mask = obs["affected_mask"].to(device=sr_old.device, dtype=sr_old.dtype)
        i_mask = obs["I_flat"].to(device=sr_old.device, dtype=sr_old.dtype).reshape(self.N1, self.N2)

        # Global checkpoints are trained with the coverage of the complete
        # observation set. Incremental appends retain those observations in
        # Weights, while affected_mask/affected_idx describe only the new batch.
        # Change the CNN context without changing local work or writeback masks.
        network_mask = affected_mask
        if self.sr_update_mode == "global":
            network_mask = torch.any(obs["Weights"] > 0, dim=1).to(
                device=sr_old.device, dtype=sr_old.dtype
            ).reshape(self.N1, self.N2)

        scale = 0.2 * torch.sigmoid(self.raw_scale)
        x = torch.stack((psi, s_old, network_mask), dim=0).unsqueeze(0)
        delta = self.net(x).squeeze(0).squeeze(0)
        candidate = F.softplus(psi + scale * delta)
        data_mask = affected_mask * i_mask
        if self.sr_update_mode == "global":
            update_weight = i_mask
        elif self.sr_update_mode == "soft_global":
            beta = float(max(0.0, min(1.0, self.global_update_weight)))
            update_weight = i_mask * (affected_mask + beta * (1.0 - affected_mask))
        else:
            update_weight = data_mask
        s_new = update_weight * candidate + (1.0 - update_weight) * s_old
        return s_new.unsqueeze(0)


class UnfoldingLayer(nn.Module):
    def __init__(
        self,
        M: int,
        N: int,
        K: int,
        nu: float,
        hidden: int,
        theta_chunk_size: int,
        theta_ridge: float,
        sr_update_mode: str,
        global_sr_update_weight: float,
        learn_nu: bool,
        min_nu: float,
    ):
        super().__init__()
        self.K = int(K)
        self.theta_layer = UnfoldingThetaLayer(
            nu=nu,
            theta_chunk_size=theta_chunk_size,
            theta_ridge=theta_ridge,
            learn_nu=learn_nu,
            min_nu=min_nu,
        )
        self.phi_layer = UnfoldingPhiLayer(
            K=K,
        )
        self.sr_layer = UnfoldingSrLayer(
            (M, N),
            hidden=hidden,
            sr_update_mode=sr_update_mode,
            global_update_weight=global_sr_update_weight,
        )

    def forward(self, state: dict, obs: dict, grid: dict) -> dict:
        theta_new = self.theta_layer(state["Theta"], state["Phi"], state["Sr"], obs, grid)
        phi_new = self.phi_layer(state["Phi"], theta_new, obs, grid)
        sr_new = self.sr_layer(state["Sr"], theta_new, obs)
        scale = phi_new.sum(dim=1, keepdim=True) / float(self.K)
        safe_scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
        phi_new = phi_new / safe_scale
        sr_new = sr_new * safe_scale.reshape(-1, 1, 1)
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
        theta_chunk_size: int = 4096,
        theta_ridge: float = 1e-5,
        sr_update_mode: str = "local",
        global_sr_update_weight: float = 0.1,
        learn_nu: bool = True,
        min_nu: float = 1e-8,
    ):
        super().__init__()
        self.M, self.N, self.K, self.R, self.T = int(M), int(N), int(K), 1, int(T)
        hidden = int(hidden)
        theta_chunk_size = int(theta_chunk_size)
        self.dim_poly = POLY_DIM
        self.nu = float(nu)
        self.learn_nu = bool(learn_nu)
        self.min_nu = float(min_nu)
        self.unfolding_layers = nn.ModuleList(
            [
                UnfoldingLayer(
                    M=self.M,
                    N=self.N,
                    K=self.K,
                    nu=self.nu,
                    hidden=hidden,
                    theta_chunk_size=theta_chunk_size,
                    theta_ridge=theta_ridge,
                    sr_update_mode=sr_update_mode,
                    global_sr_update_weight=global_sr_update_weight,
                    learn_nu=self.learn_nu,
                    min_nu=self.min_nu,
                )
                for _ in range(self.T)
            ]
        )

    def nu_values(self) -> torch.Tensor:
        return torch.stack([layer.theta_layer.nu_value() for layer in self.unfolding_layers])

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
