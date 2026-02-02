#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="t-SNE plot for CHEESE ESPSIM embeddings (blind + known actives/inactives).")
    parser.add_argument(
        "--blind-npz",
        type=str,
        default="9add_2026_screening_challenge/embeddings/cheese_espsim_blind_set.npz",
        help="Path to blind-set embeddings npz.",
    )
    parser.add_argument(
        "--known-npz",
        type=str,
        default="9add_2026_screening_challenge/embeddings/cheese_espsim_known_all_class.npz",
        help="Path to known-set embeddings npz (must contain activity_class).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="9add_2026_screening_challenge/outputs/cheese_espsim_tsne.png",
        help="Output image path.",
    )
    parser.add_argument("--perplexity", type=float, default=50.0, help="t-SNE perplexity (default: 50).")
    parser.add_argument("--seed", type=int, default=920261, help="Random seed (default: 920261).")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI (default: 200).")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    blind_npz = Path(args.blind_npz)
    known_npz = Path(args.known_npz)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blind = np.load(blind_npz, allow_pickle=True)
    known = np.load(known_npz, allow_pickle=True)

    blind_emb = blind["embedding"].astype(np.float32, copy=False)
    known_emb = known["embedding"].astype(np.float32, copy=False)
    activity_class = known["activity_class"].astype(np.int8, copy=False)

    if blind_emb.ndim != 2 or known_emb.ndim != 2:
        raise RuntimeError(f"Unexpected embedding dims: blind={blind_emb.shape}, known={known_emb.shape}")
    if known_emb.shape[0] != activity_class.shape[0]:
        raise RuntimeError(f"known embedding rows != activity_class rows: {known_emb.shape[0]} != {activity_class.shape[0]}")

    # Labels: 0=blind, 1=known_inactive, 2=known_active
    labels = np.concatenate(
        [
            np.zeros(blind_emb.shape[0], dtype=np.int8),
            (activity_class == 0).astype(np.int8) * 1 + (activity_class == 1).astype(np.int8) * 2,
        ]
    )
    X = np.vstack([blind_emb, known_emb])

    n = X.shape[0]
    perplexity = float(args.perplexity)
    if perplexity <= 1:
        raise ValueError("perplexity must be > 1")
    if perplexity >= (n - 1) / 3:
        # sklearn TSNE constraint: perplexity < n_samples
        # Practical constraint: keep it comfortably smaller.
        perplexity = max(5.0, min(30.0, (n - 1) / 3 - 1))

    from sklearn.manifold import TSNE

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=int(args.seed),
        max_iter=1500,
        verbose=1,
    )
    xy = tsne.fit_transform(X)

    import matplotlib.pyplot as plt

    # Plot order: blind first (faint), then inactives, then actives on top.
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("CHEESE ESPSIM t-SNE (blind set vs known actives/inactives)")

    m0 = labels == 0
    m1 = labels == 1
    m2 = labels == 2

    ax.scatter(xy[m0, 0], xy[m0, 1], s=6, c="#9aa0a6", alpha=0.35, linewidths=0, label=f"Blind set (n={m0.sum()})")
    ax.scatter(xy[m1, 0], xy[m1, 1], s=10, c="#1f77b4", alpha=0.75, linewidths=0, label=f"Known inactives (n={m1.sum()})")
    ax.scatter(xy[m2, 0], xy[m2, 1], s=12, c="#d62728", alpha=0.85, linewidths=0, label=f"Known actives (n={m2.sum()})")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=True, markerscale=2, loc="best")
    ax.grid(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=int(args.dpi))

    # Also save coordinates for downstream analysis.
    coords_path = out_path.with_suffix(".csv")
    with coords_path.open("w", encoding="utf-8") as f:
        f.write("set,label,tsne1,tsne2\n")
        for i in range(blind_emb.shape[0]):
            f.write(f"blind,blind,{xy[i,0]:.6f},{xy[i,1]:.6f}\n")
        offset = blind_emb.shape[0]
        for i in range(known_emb.shape[0]):
            lbl = "active" if activity_class[i] == 1 else "inactive"
            f.write(f"known,{lbl},{xy[offset+i,0]:.6f},{xy[offset+i,1]:.6f}\n")

    print(f"Wrote: {out_path}")
    print(f"Wrote: {coords_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
