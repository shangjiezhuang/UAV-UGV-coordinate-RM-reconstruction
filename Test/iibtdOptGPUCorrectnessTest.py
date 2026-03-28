import argparse
import json
import os
import sys
import time

import numpy as np
import torch

try:
    from spectrumMapTensorGen import SimConfig, generate_data
except ModuleNotFoundError:
    from Test.spectrumMapTensorGen import SimConfig, generate_data

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from IIBTD.IIBTD_Optimized import II_BTD_Optimized
from IIBTD.IIBTD_Opt_GPU import II_BTD_Opt_GPU


def relative_error(ref, val):
    ref = np.asarray(ref, dtype=np.float64)
    val = np.asarray(val, dtype=np.float64)
    return float(np.linalg.norm(ref - val) / (np.linalg.norm(ref) + 1e-12))


def max_abs_error(ref, val):
    ref = np.asarray(ref, dtype=np.float64)
    val = np.asarray(val, dtype=np.float64)
    return float(np.max(np.abs(ref - val)))


def build_solver_kwargs(cfg, args):
    return dict(
        n_sources=cfg.R,
        grid_size=(cfg.N1, cfg.N2),
        mu=args.mu,
        nu=args.nu,
        max_iter=args.max_iter,
        kernel_bandwidth=args.kernel_bandwidth,
        warmstart=False,
    )


def copy_model_state(dst_model, src_model):
    if hasattr(dst_model, "load_state_from"):
        dst_model.load_state_from(src_model)
        return
    for name in ("Theta", "Phi", "Sr", "H_hat"):
        if hasattr(src_model, name):
            setattr(dst_model, name, np.array(getattr(src_model, name), copy=True))


def make_solver(cfg, args, use_gpu, warmstart=False):
    solver_kwargs = build_solver_kwargs(cfg, args)
    solver_kwargs["warmstart"] = bool(warmstart)
    if use_gpu:
        return II_BTD_Opt_GPU(
            **solver_kwargs,
            device=args.device,
            phi_solver=args.phi_solver,
            dtype=torch.float64,
        )
    return II_BTD_Optimized(**solver_kwargs)


def run_batch_case(data, cfg, bounds, args):
    cpu = make_solver(cfg, args, use_gpu=False, warmstart=False)
    gpu = make_solver(cfg, args, use_gpu=True, warmstart=False)

    t0 = time.perf_counter()
    cpu.fit_2(
        data["sensor_locs"],
        data["Gamma_obs"],
        data["Omega"],
        data["grid_coords"],
        bounds,
        I_mask=data["I_mask"],
        debugFlag=args.debug,
    )
    cpu_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    gpu.fit_2(
        data["sensor_locs"],
        data["Gamma_obs"],
        data["Omega"],
        data["grid_coords"],
        bounds,
        I_mask=data["I_mask"],
        debugFlag=args.debug,
    )
    gpu_time = time.perf_counter() - t0

    cpu_nmse = cpu.evaluate_reconstruction2(cpu.Sr, cpu.Phi, data["S"], data["Phi"], drawFlag=False)
    gpu_nmse = gpu.evaluate_reconstruction2(gpu.Sr, gpu.Phi, data["S"], data["Phi"], drawFlag=False)

    return dict(
        case="batch_fit_2",
        cpu_time_s=float(cpu_time),
        gpu_time_s=float(gpu_time),
        map_rel_err=relative_error(cpu.get_current_map(), gpu.get_current_map()),
        map_max_abs_err=max_abs_error(cpu.get_current_map(), gpu.get_current_map()),
        sr_rel_err=relative_error(cpu.get_source_maps(), gpu.get_source_maps()),
        phi_rel_err=relative_error(cpu.get_spectra(), gpu.get_spectra()),
        nmse_cpu=float(cpu_nmse),
        nmse_gpu=float(gpu_nmse),
        nmse_abs_diff=float(abs(cpu_nmse - gpu_nmse)),
    )


def run_sequential_case(data, cfg, bounds, args):
    cpu = make_solver(cfg, args, use_gpu=False, warmstart=False)
    gpu = make_solver(cfg, args, use_gpu=True, warmstart=False)

    cpu.init_sequential(data["grid_coords"], bounds, K=cfg.K, I_mask=data["I_mask"])
    gpu.init_sequential(data["grid_coords"], bounds, K=cfg.K, I_mask=data["I_mask"])

    t0 = time.perf_counter()
    for start in range(0, cfg.M, args.chunk_size):
        end = min(start + args.chunk_size, cfg.M)
        cpu.add_measurements(
            data["sensor_locs"][start:end],
            data["Gamma_obs"][start:end],
            data["Omega"][start:end],
            n_outer_iter=args.seq_outer_iter,
            max_svt_iter=args.seq_svt_iter,
            debugFlag=args.debug,
        )
    cpu_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for start in range(0, cfg.M, args.chunk_size):
        end = min(start + args.chunk_size, cfg.M)
        gpu.add_measurements(
            data["sensor_locs"][start:end],
            data["Gamma_obs"][start:end],
            data["Omega"][start:end],
            n_outer_iter=args.seq_outer_iter,
            max_svt_iter=args.seq_svt_iter,
            debugFlag=args.debug,
        )
    gpu_time = time.perf_counter() - t0

    cpu_nmse = cpu.evaluate_reconstruction2(cpu.Sr, cpu.Phi, data["S"], data["Phi"], drawFlag=False)
    gpu_nmse = gpu.evaluate_reconstruction2(gpu.Sr, gpu.Phi, data["S"], data["Phi"], drawFlag=False)

    return dict(
        case="sequential_add_measurements",
        cpu_time_s=float(cpu_time),
        gpu_time_s=float(gpu_time),
        map_rel_err=relative_error(cpu.get_current_map(), gpu.get_current_map()),
        map_max_abs_err=max_abs_error(cpu.get_current_map(), gpu.get_current_map()),
        sr_rel_err=relative_error(cpu.get_source_maps(), gpu.get_source_maps()),
        phi_rel_err=relative_error(cpu.get_spectra(), gpu.get_spectra()),
        nmse_cpu=float(cpu_nmse),
        nmse_gpu=float(gpu_nmse),
        nmse_abs_diff=float(abs(cpu_nmse - gpu_nmse)),
    )


def run_history_refit_case(data, cfg, bounds, args):
    prev_cpu = None
    prev_gpu = None

    t0 = time.perf_counter()
    for start in range(0, cfg.M, args.chunk_size):
        end = min(start + args.chunk_size, cfg.M)
        cpu = make_solver(cfg, args, use_gpu=False, warmstart=(prev_cpu is not None))
        cpu.init_sequential(data["grid_coords"], bounds, K=cfg.K, I_mask=data["I_mask"])
        if prev_cpu is not None:
            copy_model_state(cpu, prev_cpu)
        cpu.fit_2(
            data["sensor_locs"][:end],
            data["Gamma_obs"][:end],
            data["Omega"][:end],
            data["grid_coords"],
            bounds,
            I_mask=data["I_mask"],
            debugFlag=args.debug,
        )
        prev_cpu = cpu
    cpu_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for start in range(0, cfg.M, args.chunk_size):
        end = min(start + args.chunk_size, cfg.M)
        gpu = make_solver(cfg, args, use_gpu=True, warmstart=(prev_gpu is not None))
        gpu.init_sequential(data["grid_coords"], bounds, K=cfg.K, I_mask=data["I_mask"])
        if prev_gpu is not None:
            copy_model_state(gpu, prev_gpu)
        gpu.fit_2(
            data["sensor_locs"][:end],
            data["Gamma_obs"][:end],
            data["Omega"][:end],
            data["grid_coords"],
            bounds,
            I_mask=data["I_mask"],
            debugFlag=args.debug,
        )
        prev_gpu = gpu
    gpu_time = time.perf_counter() - t0

    cpu_nmse = prev_cpu.evaluate_reconstruction2(
        prev_cpu.Sr,
        prev_cpu.Phi,
        data["S"],
        data["Phi"],
        drawFlag=False,
    )
    gpu_nmse = prev_gpu.evaluate_reconstruction2(
        prev_gpu.Sr,
        prev_gpu.Phi,
        data["S"],
        data["Phi"],
        drawFlag=False,
    )

    return dict(
        case="history_refit_fit_2",
        cpu_time_s=float(cpu_time),
        gpu_time_s=float(gpu_time),
        map_rel_err=relative_error(prev_cpu.get_current_map(), prev_gpu.get_current_map()),
        map_max_abs_err=max_abs_error(prev_cpu.get_current_map(), prev_gpu.get_current_map()),
        sr_rel_err=relative_error(prev_cpu.get_source_maps(), prev_gpu.get_source_maps()),
        phi_rel_err=relative_error(prev_cpu.get_spectra(), prev_gpu.get_spectra()),
        nmse_cpu=float(cpu_nmse),
        nmse_gpu=float(gpu_nmse),
        nmse_abs_diff=float(abs(cpu_nmse - gpu_nmse)),
    )


def build_speed_summary(results):
    by_case = {item["case"]: item for item in results}
    summary = {}
    seq_case = by_case.get("sequential_add_measurements")
    hist_case = by_case.get("history_refit_fit_2")
    batch_case = by_case.get("batch_fit_2")
    if seq_case is not None and hist_case is not None:
        for prefix, key in (("cpu", "cpu_time_s"), ("gpu", "gpu_time_s")):
            seq_t = float(seq_case[key])
            hist_t = float(hist_case[key])
            summary[f"{prefix}_history_refit_over_sequential"] = (
                float(hist_t / seq_t) if seq_t > 0 else float("inf")
            )
            summary[f"{prefix}_sequential_speedup_vs_history_refit"] = (
                float(hist_t / seq_t) if seq_t > 0 else float("inf")
            )
    if batch_case is not None and hist_case is not None:
        for prefix, key in (("cpu", "cpu_time_s"), ("gpu", "gpu_time_s")):
            batch_t = float(batch_case[key])
            hist_t = float(hist_case[key])
            summary[f"{prefix}_history_refit_over_single_batch"] = (
                float(hist_t / batch_t) if batch_t > 0 else float("inf")
            )
    return summary


def validate_results(results, args):
    failures = []
    for result in results:
        if result["map_rel_err"] > args.map_tol:
            failures.append(f"{result['case']}: map_rel_err={result['map_rel_err']:.3e} > {args.map_tol:.3e}")
        if result["nmse_abs_diff"] > args.nmse_tol:
            failures.append(f"{result['case']}: nmse_abs_diff={result['nmse_abs_diff']:.3e} > {args.nmse_tol:.3e}")
        if result["sr_rel_err"] > args.sr_tol:
            failures.append(f"{result['case']}: sr_rel_err={result['sr_rel_err']:.3e} > {args.sr_tol:.3e}")
        if result["phi_rel_err"] > args.phi_tol:
            failures.append(f"{result['case']}: phi_rel_err={result['phi_rel_err']:.3e} > {args.phi_tol:.3e}")
    return failures


def parse_args():
    parser = argparse.ArgumentParser(description="Compare CPU and GPU II-BTD correctness.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--M", type=int, default=60)
    parser.add_argument("--R", type=int, default=1)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=2)
    parser.add_argument("--seq-outer-iter", type=int, default=2)
    parser.add_argument("--seq-svt-iter", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--mu", type=float, default=1.2)
    parser.add_argument("--nu", type=float, default=1.5)
    parser.add_argument("--kernel-bandwidth", type=float, default=0.46)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--phi-solver", type=str, default="scipy", choices=["scipy", "pgd"])
    parser.add_argument("--map-tol", type=float, default=5e-5)
    parser.add_argument("--sr-tol", type=float, default=5e-5)
    parser.add_argument("--phi-tol", type=float, default=5e-5)
    parser.add_argument("--nmse-tol", type=float, default=5e-5)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    cfg = SimConfig(full_obs=False, rho=args.rho, R=args.R, M=args.M)
    data = generate_data(cfg, seed=args.seed)
    bounds = ((0, cfg.L), (0, cfg.L))

    results = [
        run_batch_case(data, cfg, bounds, args),
        run_sequential_case(data, cfg, bounds, args),
        run_history_refit_case(data, cfg, bounds, args),
    ]

    failures = validate_results(results, args)
    payload = {
        "device": args.device,
        "phi_solver": args.phi_solver,
        "config": {
            "N1": cfg.N1,
            "N2": cfg.N2,
            "K": cfg.K,
            "M": cfg.M,
            "R": cfg.R,
            "rho": cfg.rho,
            "max_iter": args.max_iter,
            "chunk_size": args.chunk_size,
        },
        "results": results,
        "speed_summary": build_speed_summary(results),
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))

    if failures:
        raise AssertionError("\n".join(failures))


if __name__ == "__main__":
    main()
