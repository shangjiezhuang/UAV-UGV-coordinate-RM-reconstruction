"""Shared helpers for the DU-IIBTD residual-Sr learned-nu backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch exists in training env
    torch = None

try:
    from DU_IIBTD_res_Sr_learn_nu.solver_adapter import DU_IIBTDSolverAdapter
except ModuleNotFoundError:  # pragma: no cover - surfaced when reconstruction starts
    DU_IIBTDSolverAdapter = None


LEARN_NU_BACKEND = "du_iibtd_res_sr_learn_nu"
DU_IIBTD_BACKEND = "du_iibtd"
DU_IIBTD_RES_SR_BACKEND = "du_iibtd_res_sr"
DU_IIBTD_BACKENDS = {
    DU_IIBTD_BACKEND,
    DU_IIBTD_RES_SR_BACKEND,
    LEARN_NU_BACKEND,
}
DEFAULT_DU_IIBTD_CHECKPOINTS = [
    "DU_IIBTD/runs_t2_h04_1/checkpoints/best_nmse.pth",
    "DU_IIBTD/runs_t2_h05_1/checkpoints/best_nmse.pth",
    "DU_IIBTD/runs_t2_h06_1/checkpoints/best_nmse.pth",
]
DEFAULT_DU_IIBTD_RES_SR_CHECKPOINTS = [
    "DU_IIBTD_res_Sr/runs_t3_h04_res_balance_bw/checkpoints/best_nmse.pth",
    "DU_IIBTD_res_Sr/runs_t3_h05_res_balance_bw/checkpoints/best_nmse.pth",
    "DU_IIBTD_res_Sr/runs_t3_h06_res_balance_bw/checkpoints/best_nmse.pth",
]
DEFAULT_DU_IIBTD_RES_SR_LEARN_NU_CHECKPOINTS = [
    "DU_IIBTD_res_Sr_learn_nu/runs_t3_h04_res_srGlobal_learnNu/checkpoints/best_nmse.pth",
    "DU_IIBTD_res_Sr_learn_nu/runs_t3_h05_res_srGlobal_learnNu/checkpoints/best_nmse.pth",
    "DU_IIBTD_res_Sr_learn_nu/runs_t3_h06_res_srGlobal_learnNu/checkpoints/best_nmse.pth",
]


def is_du_iibtd_backend(backend: str) -> bool:
    return str(backend or DU_IIBTD_BACKEND).strip().lower() in DU_IIBTD_BACKENDS


def is_res_sr_backend(backend: str) -> bool:
    return str(backend or "").strip().lower() == DU_IIBTD_RES_SR_BACKEND


def is_learn_nu_backend(backend: str) -> bool:
    return str(backend or "").strip().lower() == LEARN_NU_BACKEND


def default_du_iibtd_checkpoints_for_backend(backend: str) -> List[str]:
    backend = str(backend or DU_IIBTD_BACKEND).strip().lower()
    if backend == LEARN_NU_BACKEND:
        return list(DEFAULT_DU_IIBTD_RES_SR_LEARN_NU_CHECKPOINTS)
    if backend == DU_IIBTD_RES_SR_BACKEND:
        return list(DEFAULT_DU_IIBTD_RES_SR_CHECKPOINTS)
    if backend == DU_IIBTD_BACKEND:
        return list(DEFAULT_DU_IIBTD_CHECKPOINTS)
    raise ValueError(f"Unsupported DU-IIBTD backend: {backend!r}")


def _project_root_from_shared_dir(shared_dir: str | Path) -> Path:
    return Path(shared_dir).resolve().parent


def normalize_checkpoint_paths(
    shared_dir: str | Path,
    checkpoint_paths: Optional[List[str]] = None,
    *,
    backend: str = LEARN_NU_BACKEND,
) -> List[str]:
    paths = list(checkpoint_paths or default_du_iibtd_checkpoints_for_backend(backend))
    project_root = _project_root_from_shared_dir(shared_dir)
    resolved: List[str] = []
    for path in paths:
        checkpoint = Path(str(path)).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = project_root / checkpoint
        resolved.append(str(checkpoint))
    return resolved


def resolve_member_checkpoint(
    shared_dir: str | Path,
    member_idx: int,
    checkpoint_paths: Optional[List[str]] = None,
    *,
    backend: str = LEARN_NU_BACKEND,
) -> str:
    paths = normalize_checkpoint_paths(shared_dir, checkpoint_paths, backend=backend)
    if not paths:
        raise ValueError(f"At least one {backend} checkpoint is required.")
    return paths[int(member_idx) % len(paths)]


def resolve_checkpoint_for_bandwidth(
    shared_dir: str | Path,
    kernel_bandwidth: float,
    checkpoint_paths: Optional[List[str]] = None,
) -> str:
    paths = normalize_checkpoint_paths(shared_dir, checkpoint_paths)
    target = float(kernel_bandwidth)
    return min(
        paths,
        key=lambda path: abs(
            checkpoint_config_float(path, "kernel_bandwidth", target) - target
        ),
    )


def read_checkpoint_train_config(checkpoint_path: str) -> Dict[str, object]:
    run_dir = Path(checkpoint_path).resolve().parents[1]
    config_path = run_dir / "train_config.json"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            return dict(json.load(handle))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid checkpoint train_config.json: {config_path}") from exc


def checkpoint_config_float(checkpoint_path: str, key: str, fallback: float) -> float:
    value = read_checkpoint_train_config(checkpoint_path).get(str(key), fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def checkpoint_config_int_or_none(
    checkpoint_path: str,
    key: str,
    fallback: Optional[int],
) -> Optional[int]:
    value = read_checkpoint_train_config(checkpoint_path).get(str(key), fallback)
    if value is None:
        return None
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        if fallback is None:
            return None
        value_int = int(fallback)
    return None if value_int <= 0 else value_int


def resolve_solver_hyperparams(
    checkpoint_path: str,
    *,
    fallback_nu: float,
    fallback_kernel_bandwidth: float,
    fallback_min_sensors_for_update: Optional[int] = None,
    fallback_update_batch_size: Optional[int] = None,
) -> Dict[str, object]:
    return {
        "nu": checkpoint_config_float(
            checkpoint_path,
            "nu",
            float(fallback_nu),
        ),
        "kernel_bandwidth": checkpoint_config_float(
            checkpoint_path,
            "kernel_bandwidth",
            float(fallback_kernel_bandwidth),
        ),
        "min_sensors_for_update": checkpoint_config_int_or_none(
            checkpoint_path,
            "min_sensors_for_update",
            fallback_min_sensors_for_update,
        ),
        "update_batch_size": checkpoint_config_int_or_none(
            checkpoint_path,
            "update_batch_size",
            fallback_update_batch_size,
        ),
    }


def make_solver(
    *,
    shared_dir: str | Path,
    n_sources: int,
    grid_size: Tuple[int, int],
    max_iter: int,
    mu: float,
    nu: float,
    kernel_bandwidth: float,
    warmstart: bool,
    solver_device: str,
    resolve_device: Callable[[Optional[str]], str],
    checkpoint_path: Optional[str] = None,
):
    if DU_IIBTDSolverAdapter is None:
        raise RuntimeError("DU-IIBTD_res_Sr_learn_nu solver adapter is unavailable.")
    if torch is None:
        raise RuntimeError("DU-IIBTD_res_Sr_learn_nu backend requires torch.")

    checkpoint = str(
        checkpoint_path
        or resolve_checkpoint_for_bandwidth(shared_dir, kernel_bandwidth)
    )
    solver_hyperparams = resolve_solver_hyperparams(
        checkpoint,
        fallback_nu=float(nu),
        fallback_kernel_bandwidth=float(kernel_bandwidth),
        fallback_min_sensors_for_update=None,
        fallback_update_batch_size=None,
    )
    return DU_IIBTDSolverAdapter(
        n_sources=n_sources,
        grid_size=grid_size,
        mu=float(mu),
        nu=float(solver_hyperparams["nu"]),
        max_iter=max(1, int(max_iter)),
        kernel_bandwidth=float(solver_hyperparams["kernel_bandwidth"]),
        warmstart=bool(warmstart),
        checkpoint_path=checkpoint,
        device=resolve_device(solver_device),
        dtype=torch.float32,
        min_sensors_for_update=solver_hyperparams["min_sensors_for_update"],
        update_batch_size=solver_hyperparams["update_batch_size"],
    )
