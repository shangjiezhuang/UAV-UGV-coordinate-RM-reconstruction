from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from DU_IIBTD import DU_IIBTD, append_observation, make_grid_norm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_nmse_loss(H_hat: torch.Tensor, H_true: torch.Tensor, valid_mask: torch.Tensor | None = None, eps: float = 1e-8):
    if valid_mask is None:
        diff = H_hat - H_true
        return diff.square().sum() / H_true.square().sum().clamp_min(eps)
    mask = valid_mask.to(device=H_true.device, dtype=H_true.dtype).unsqueeze(-1)
    diff = (H_hat - H_true) * mask
    return diff.square().sum() / ((H_true * mask).square().sum().clamp_min(eps))


def observation_consistency_loss(H_hat: torch.Tensor, obs: dict, N2: int, eps: float = 1e-8):
    if obs is None or "sample_grid_idx" not in obs:
        return H_hat.new_tensor(0.0)
    sample_idx = obs["sample_grid_idx"].to(device=H_hat.device, dtype=torch.long)
    if sample_idx.numel() == 0:
        return H_hat.new_tensor(0.0)
    i = torch.div(sample_idx, N2, rounding_mode="floor")
    j = torch.remainder(sample_idx, N2)
    pred = H_hat[i, j, :]
    gamma = obs["Gamma"]
    omega = obs["Omega"]
    return (((pred - gamma) ** 2) * omega).sum() / omega.sum().clamp_min(eps)


def read_manifest(dataset_root: Path) -> list[dict]:
    manifest_path = dataset_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    valid_rows = []
    for row in rows:
        rel = row.get("arrays_npz", "")
        if rel and (dataset_root / rel).exists():
            valid_rows.append(row)
    if not valid_rows:
        raise FileNotFoundError(f"No usable arrays_npz entries found in {manifest_path}")
    return valid_rows


def split_by_base_crop(
    rows: list[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    max_samples: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    by_base: dict[str, list[dict]] = {}
    for row in rows:
        by_base.setdefault(row["base_name"], []).append(row)

    base_names = sorted(by_base)
    rng = np.random.default_rng(seed)
    rng.shuffle(base_names)

    n_base = len(base_names)
    n_test = int(round(n_base * test_ratio))
    n_val = int(round(n_base * val_ratio))
    test_bases = base_names[:n_test]
    val_bases = base_names[n_test : n_test + n_val]
    train_bases = base_names[n_test + n_val :]

    train = [row for base in train_bases for row in by_base[base]]
    val = [row for base in val_bases for row in by_base[base]]
    test = [row for base in test_bases for row in by_base[base]]
    for split_rows in (train, val, test):
        rng.shuffle(split_rows)

    if max_samples > 0:
        train = train[:max_samples]
        val_cap = max(1, int(round(max_samples * max(val_ratio, 0.05))))
        test_cap = max(1, int(round(max_samples * max(test_ratio, 0.05))))
        val = val[: min(len(val), val_cap)]
        test = test[: min(len(test), test_cap)]
    return train, val, test


def write_split_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_npz_sample(dataset_root: Path, row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(dataset_root / row["arrays_npz"]) as data:
        S = np.asarray(data["S"], dtype=np.float32)
        Phi = np.asarray(data["Phi"], dtype=np.float32)
        valid_mask = np.asarray(data["non_building_mask"], dtype=bool)
    return S, Phi, valid_mask


def choose_frequency_bands(
    K: int,
    mode: str,
    band_min: int,
    band_max: int,
    n_points: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if mode == "full":
        band = np.arange(K, dtype=np.int64)
        return [band for _ in range(n_points)]

    width_min = max(1, min(K, int(band_min)))
    width_max = max(width_min, min(K, int(band_max)))

    if mode == "contiguous":
        width = int(rng.integers(width_min, width_max + 1))
        start = int(rng.integers(0, K - width + 1))
        band = np.arange(start, start + width, dtype=np.int64)
        return [band for _ in range(n_points)]

    if mode != "per_point_contiguous":
        raise ValueError(f"Unsupported freq_mode: {mode}")

    bands = []
    for _ in range(n_points):
        width = int(rng.integers(width_min, width_max + 1))
        start = int(rng.integers(0, K - width + 1))
        bands.append(np.arange(start, start + width, dtype=np.int64))
    return bands


def epoch_subset_indices(n_items: int, samples_per_epoch: int, epoch: int, seed: int) -> list[int]:
    n_items = int(n_items)
    samples_per_epoch = int(samples_per_epoch)
    if n_items <= 0:
        return []
    if samples_per_epoch <= 0 or samples_per_epoch >= n_items:
        return list(range(n_items))

    cursor = (max(1, int(epoch)) - 1) * samples_per_epoch
    indices: list[int] = []
    seen: set[int] = set()
    while len(indices) < samples_per_epoch:
        cycle = cursor // n_items
        offset = cursor % n_items
        rng = np.random.default_rng(int(seed) + cycle * 10000019)
        order = np.arange(n_items, dtype=np.int64)
        rng.shuffle(order)
        for value in order[offset:]:
            cursor += 1
            value = int(value)
            if value in seen:
                continue
            indices.append(value)
            seen.add(value)
            if len(indices) == samples_per_epoch:
                break
    return indices


class RadioSeerDpmIibtdDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        rows: list[dict],
        *,
        spatial_points_min: int,
        spatial_points_max: int,
        freq_mode: str,
        freq_band_min: int,
        freq_band_max: int,
        seed: int,
    ):
        self.dataset_root = dataset_root
        self.rows = rows
        self.spatial_points_min = int(spatial_points_min)
        self.spatial_points_max = int(spatial_points_max)
        self.freq_mode = freq_mode
        self.freq_band_min = int(freq_band_min)
        self.freq_band_max = int(freq_band_max)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, int(epoch))

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        S, Phi, valid_mask = load_npz_sample(self.dataset_root, row)
        n1, n2 = S.shape
        K = Phi.shape[0]
        H = (S[:, :, None] * Phi[None, None, :]).astype(np.float32)

        rng = np.random.default_rng(self.seed + self.epoch * 1000000007 + idx * 1000003)
        candidates = np.flatnonzero(valid_mask.reshape(-1))
        if candidates.size == 0:
            candidates = np.arange(n1 * n2, dtype=np.int64)

        n_obs = min(int(rng.integers(self.spatial_points_min, self.spatial_points_max + 1)), int(candidates.size))
        sample_idx = rng.choice(candidates, size=n_obs, replace=False).astype(np.int64)

        rows_i = sample_idx // n2
        cols_j = sample_idx % n2
        locs_norm = np.column_stack(
            (
                2.0 * rows_i.astype(np.float32) / max(1, n1 - 1) - 1.0,
                2.0 * cols_j.astype(np.float32) / max(1, n2 - 1) - 1.0,
            )
        ).astype(np.float32)

        gamma = np.zeros((n_obs, K), dtype=np.float32)
        omega = np.zeros((n_obs, K), dtype=np.float32)
        bands = choose_frequency_bands(K, self.freq_mode, self.freq_band_min, self.freq_band_max, n_obs, rng)
        for point_id, freq_idx in enumerate(bands):
            i = int(rows_i[point_id])
            j = int(cols_j[point_id])
            gamma[point_id, freq_idx] = H[i, j, freq_idx]
            omega[point_id, freq_idx] = 1.0

        return {
            "H": torch.from_numpy(H),
            "valid_mask": torch.from_numpy(valid_mask.astype(np.float32)),
            "I_mask": torch.from_numpy(valid_mask.reshape(-1)),
            "locs_norm": torch.from_numpy(locs_norm),
            "Gamma": torch.from_numpy(gamma),
            "Omega": torch.from_numpy(omega),
            "sample_grid_idx": torch.from_numpy(sample_idx),
            "sample_name": row["sample_name"],
        }


def collate_items(batch: list[dict]) -> list[dict]:
    return batch


def make_loader(dataset: Dataset, args: argparse.Namespace, pin_memory: bool, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_items,
    )


def make_dataset(
    dataset_root: Path,
    rows: list[dict],
    args: argparse.Namespace,
    *,
    seed: int,
) -> RadioSeerDpmIibtdDataset:
    return RadioSeerDpmIibtdDataset(
        dataset_root,
        rows,
        spatial_points_min=args.spatial_points_min,
        spatial_points_max=args.spatial_points_max,
        freq_mode=args.freq_mode,
        freq_band_min=args.freq_band_min,
        freq_band_max=args.freq_band_max,
        seed=seed,
    )


def run_item(model: DU_IIBTD, item: dict, grid: dict, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, dict]:
    H_true = item["H"].to(device=device, dtype=torch.float32, non_blocking=True)
    valid_mask = item["valid_mask"].to(device=device, dtype=torch.float32, non_blocking=True)
    I_flat = item["I_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
    locs = item["locs_norm"].to(device=device, dtype=torch.float32, non_blocking=True)
    gamma = item["Gamma"].to(device=device, dtype=torch.float32, non_blocking=True)
    omega = item["Omega"].to(device=device, dtype=torch.float32, non_blocking=True)
    sample_idx = item["sample_grid_idx"].to(device=device, dtype=torch.long, non_blocking=True)

    state = model.init_state(device=device, dtype=torch.float32)
    obs = append_observation(
        None,
        locs,
        gamma,
        omega,
        grid["grid_norm"],
        (model.M, model.N),
        args.kernel_bandwidth,
        I_flat=I_flat,
        sample_grid_idx=sample_idx,
    )
    updates = 0
    if obs["locs_norm"].shape[0] >= args.min_sensors_for_update:
        state = model(state, obs, grid)
        updates = 1

    H_hat = state["H_hat"]
    recon_loss = masked_nmse_loss(H_hat, H_true, valid_mask)
    obs_loss = observation_consistency_loss(H_hat, obs, model.N)
    loss = recon_loss + float(args.obs_loss_weight) * obs_loss

    with torch.no_grad():
        nmse = recon_loss.detach()
        mse = ((H_hat - H_true).square() * valid_mask.unsqueeze(-1)).sum()
        rmse = torch.sqrt(mse / (valid_mask.sum().clamp_min(1.0) * model.K))
        metrics = {
            "loss": float(loss.detach().cpu()),
            "nmse": float(nmse.cpu()),
            "rmse": float(rmse.detach().cpu()),
            "obs_loss": float(obs_loss.detach().cpu()),
            "updates": updates,
            "observation_count": int(locs.shape[0]),
            "observed_entries": int(torch.sum(omega > 0).detach().cpu().item()),
        }
    return loss, metrics


def run_epoch(
    model: DU_IIBTD,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    grid: dict,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    split_name: str,
) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {
        "loss": 0.0,
        "nmse": 0.0,
        "rmse": 0.0,
        "obs_loss": 0.0,
        "updates": 0.0,
        "observation_count": 0.0,
        "observed_entries": 0.0,
    }
    n_items = 0

    for batch_id, items in enumerate(loader, start=1):
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        losses = []
        batch_metrics = []
        with torch.set_grad_enabled(is_train):
            for item in items:
                loss, metrics = run_item(model, item, grid, args, device)
                losses.append(loss)
                batch_metrics.append(metrics)
            batch_loss = torch.stack(losses).mean()
            if is_train:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

        for metrics in batch_metrics:
            for key in totals:
                totals[key] += metrics[key]
            n_items += 1

        if args.log_interval > 0 and (batch_id % args.log_interval == 0 or batch_id == len(loader)):
            print(
                f"{split_name} epoch={epoch} batch={batch_id}/{len(loader)} "
                f"loss={totals['loss'] / max(1, n_items):.5g} "
                f"nmse={totals['nmse'] / max(1, n_items):.5g}",
                flush=True,
            )

    if n_items == 0:
        raise ValueError("Empty dataloader.")
    return {key: value / max(1, n_items) for key, value in totals.items()}


def save_training_curves(history: list[dict], output_dir: Path) -> None:
    if not history:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skip plot: {exc}", flush=True)
        return

    epochs = [int(row["epoch"]) for row in history]
    metrics = [
        ("loss", "Total loss"),
        ("nmse", "NMSE"),
        ("rmse", "RMSE"),
        ("obs_loss", "Observation consistency"),
    ]
    styles = {
        "train": {"marker": "o", "linestyle": "-", "color": "#1f77b4"},
        "val": {"marker": "s", "linestyle": "--", "color": "#d62728"},
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.reshape(-1)
    for ax, (metric_key, title) in zip(axes_flat, metrics):
        train_values = [float(row["train"][metric_key]) for row in history]
        ax.plot(epochs, train_values, linewidth=1.8, label=f"train {metric_key}", **styles["train"])
        val_rows = [row for row in history if row.get("val") is not None]
        if val_rows:
            val_epochs = [int(row["epoch"]) for row in val_rows]
            val_values = [float(row["val"][metric_key]) for row in val_rows]
            ax.plot(val_epochs, val_values, linewidth=1.8, label=f"val {metric_key}", **styles["val"])
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_key)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def resolve_output_dir(output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg)

    base = Path.cwd() / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = base
    suffix = 1
    while output_dir.exists():
        output_dir = base.with_name(f"{base.name}_{suffix:02d}")
        suffix += 1
    return output_dir


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train R=1 GPU DU-IIBTD on RadioSeerDPM PSD tensors.")
    parser.add_argument("--dataset-root", default=str(root / "RadioSeerDPM100PSD"))
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for run outputs. If omitted, create a timestamped directory under the current working directory.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--train-samples-per-epoch",
        type=int,
        default=1000,
        help="Number of training samples per mini-epoch. Use 0 to train on the full train split every epoch.",
    )
    parser.add_argument("--val-interval", type=int, default=10)
    parser.add_argument(
        "--val-max-samples",
        type=int,
        default=0,
        help="Maximum validation samples per validation pass. Use 0 to evaluate the full val split.",
    )

    parser.add_argument("--spatial-points-min", type=int, default=128)
    parser.add_argument("--spatial-points-max", type=int, default=256)
    parser.add_argument("--freq-mode", default="per_point_contiguous")
    parser.add_argument("--freq-band-min", type=int, default=2)
    parser.add_argument("--freq-band-max", type=int, default=6)
    parser.add_argument("--min-sensors-for-update", type=int, default=6)

    parser.add_argument("--epochs", "--epoch", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--obs-loss-weight", type=float, default=0.05)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--fixed-train-observation-sets", action="store_true")

    parser.add_argument("--T", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--kernel-bandwidth", type=float, default=0.2)
    parser.add_argument("--local-sr-update", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device was requested but CUDA is unavailable: {device}.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} was requested, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available."
            )
    if args.spatial_points_max < args.spatial_points_min:
        raise ValueError("--spatial-points-max must be >= --spatial-points-min.")
    if args.spatial_points_max < args.min_sensors_for_update:
        raise ValueError("--spatial-points-max must be >= --min-sensors-for-update.")
    dataset_root = Path(args.dataset_root)
    output_dir = resolve_output_dir(args.output_dir)
    args.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir={output_dir}", flush=True)

    rows = read_manifest(dataset_root)
    train_rows, val_rows, test_rows = split_by_base_crop(rows, args.val_ratio, args.test_ratio, args.seed, args.max_samples)
    if not train_rows or not val_rows or not test_rows:
        raise ValueError(f"Empty split: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}.")
    write_split_csv(output_dir / "splits" / "train.csv", train_rows)
    write_split_csv(output_dir / "splits" / "val.csv", val_rows)
    write_split_csv(output_dir / "splits" / "test.csv", test_rows)

    S0, Phi0, _ = load_npz_sample(dataset_root, train_rows[0])
    n1, n2 = S0.shape
    K = int(Phi0.shape[0])

    val_eval_rows = val_rows[: min(len(val_rows), args.val_max_samples)] if args.val_max_samples > 0 else val_rows
    train_ds = make_dataset(
        dataset_root,
        train_rows,
        args,
        seed=args.seed,
    )
    val_ds = make_dataset(
        dataset_root,
        val_eval_rows,
        args,
        seed=args.seed + 777,
    )
    test_ds = make_dataset(
        dataset_root,
        test_rows,
        args,
        seed=args.seed + 1554,
    )

    pin_memory = device.type == "cuda"
    val_loader = make_loader(val_ds, args, pin_memory, shuffle=False)
    test_loader = make_loader(test_ds, args, pin_memory, shuffle=False)

    model = DU_IIBTD(
        M=n1,
        N=n2,
        K=K,
        T=args.T,
        nu=args.nu,
        hidden=args.hidden,
        local_sr_update=args.local_sr_update,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    grid = {"grid_norm": make_grid_norm(n1, n2, device=device, dtype=torch.float32)}

    config = vars(args).copy()
    train_samples_per_epoch = int(args.train_samples_per_epoch)
    if train_samples_per_epoch <= 0 or train_samples_per_epoch >= len(train_ds):
        train_samples_per_epoch = len(train_ds)
    val_interval = max(1, int(args.val_interval))
    val_eval_count = len(val_ds)
    full_train_pass_epochs = math.ceil(len(train_ds) / max(1, train_samples_per_epoch))

    config.update(
        {
            "N1": n1,
            "N2": n2,
            "K": K,
            "R": 1,
            "observation_mode": "set",
            "effective_observation_points_min": int(args.spatial_points_min),
            "effective_observation_points_max": int(args.spatial_points_max),
            "effective_train_samples_per_epoch": train_samples_per_epoch,
            "full_train_pass_epochs": full_train_pass_epochs,
            "val_eval_samples": val_eval_count,
            "splits": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        }
    )
    (output_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"config={output_dir / 'train_config.json'}", flush=True)

    best_val_nmse = math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(0 if args.fixed_train_observation_sets else epoch)
        train_indices = epoch_subset_indices(len(train_ds), train_samples_per_epoch, epoch, args.seed + 24681357)
        if len(train_indices) < len(train_ds):
            train_epoch_ds = Subset(train_ds, train_indices)
            train_shuffle = False
        else:
            train_epoch_ds = train_ds
            train_shuffle = True
        train_loader = make_loader(train_epoch_ds, args, pin_memory, shuffle=train_shuffle)
        train_metrics = run_epoch(model, train_loader, optimizer, grid, args, device, epoch, "train")
        should_validate = epoch == 1 or epoch == args.epochs or epoch % val_interval == 0
        if should_validate:
            with torch.no_grad():
                val_ds.set_epoch(0)
                val_metrics = run_epoch(model, val_loader, None, grid, args, device, epoch, "val")
        else:
            val_metrics = None

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "train_samples": len(train_indices),
            "train_pass_fraction": len(train_indices) / max(1, len(train_ds)),
        }
        history.append(row)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_training_curves(history, output_dir)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "train": train_metrics,
            "val": val_metrics,
        }
        torch.save(checkpoint, checkpoint_dir / "latest.pth")
        if val_metrics is not None and val_metrics["nmse"] < best_val_nmse:
            best_val_nmse = val_metrics["nmse"]
            torch.save(checkpoint, checkpoint_dir / "best_nmse.pth")

        summary = (
            f"epoch {epoch}/{args.epochs} train_samples={len(train_indices)} "
            f"train_loss={train_metrics['loss']:.5g} train_nmse={train_metrics['nmse']:.5g}"
        )
        if val_metrics is not None:
            summary += f" val_loss={val_metrics['loss']:.5g} val_nmse={val_metrics['nmse']:.5g}"
        print(summary, flush=True)

    best_path = checkpoint_dir / "best_nmse.pth"
    if not best_path.exists():
        best_path = checkpoint_dir / "latest.pth"
    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    with torch.no_grad():
        test_ds.set_epoch(0)
        test_metrics = run_epoch(model, test_loader, None, grid, args, device, int(best_checkpoint["epoch"]), "test")

    test_summary = {
        "checkpoint_epoch": int(best_checkpoint["epoch"]),
        "val": best_checkpoint["val"],
        "test": test_metrics,
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(test_summary, indent=2), encoding="utf-8")
    print(
        f"test checkpoint_epoch={test_summary['checkpoint_epoch']} "
        f"test_loss={test_metrics['loss']:.5g} test_nmse={test_metrics['nmse']:.5g}",
        flush=True,
    )


if __name__ == "__main__":
    main(parse_args())
