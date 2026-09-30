#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Mark known correct answers on t-SNE plots with crosses: "
            "green cross if also in CHEESE ShapeSim top-100, red cross otherwise."
        )
    )
    p.add_argument(
        "--correct-answers-txt",
        type=str,
        default="9add_2026_screening_challenge/data/correct_answers.txt",
        help="Path to correct answers (one compound ID per line).",
    )
    p.add_argument(
        "--shapesim-top100-txt",
        type=str,
        default="9add_2026_screening_challenge/submissions/cheese_shapesim_top100_meancos.txt",
        help="Path to CHEESE ShapeSim top-100 IDs (one compound ID per line).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="9add_2026_screening_challenge/outputs/correct_answers",
        help="Output directory for annotated plots.",
    )

    # ShapeSim-embedding t-SNE inputs
    p.add_argument(
        "--shapesim-blind-npz",
        type=str,
        default="9add_2026_screening_challenge/embeddings/cheese_shapesim_blind_set.npz",
        help="Blind-set CHEESE ShapeSim embedding npz.",
    )
    p.add_argument(
        "--shapesim-known-npz",
        type=str,
        default="9add_2026_screening_challenge/embeddings/cheese_shapesim_known_all_class.npz",
        help="Known-set CHEESE ShapeSim embedding npz (must contain activity_class).",
    )

    # Morgan-fingerprint t-SNE inputs (SMILES + activity_class are taken from ESPSIM npz files)
    p.add_argument(
        "--morgan-blind-npz",
        type=str,
        default="9add_2026_screening_challenge/embeddings/cheese_espsim_blind_set.npz",
        help="Blind-set npz providing SMILES + compound_id for Morgan fingerprints.",
    )
    p.add_argument(
        "--morgan-known-npz",
        type=str,
        default="9add_2026_screening_challenge/embeddings/cheese_espsim_known_all_class.npz",
        help="Known-set npz providing SMILES + activity_class for Morgan fingerprints.",
    )
    p.add_argument("--morgan-radius", type=int, default=2, help="Morgan fingerprint radius (default: 2).")
    p.add_argument("--morgan-nbits", type=int, default=2048, help="Morgan fingerprint bit length (default: 2048).")

    # t-SNE params (match export_presentation_assets defaults)
    p.add_argument("--perplexity", type=float, default=50.0, help="t-SNE perplexity (default: 50).")
    p.add_argument("--seed", type=int, default=920261, help="Random seed for t-SNE (default: 920261).")
    p.add_argument("--dpi", type=int, default=250, help="Output DPI (default: 250).")
    return p.parse_args()


def _read_id_list(path: Path) -> list[str]:
    raw: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        raw.append(s)

    seen: set[str] = set()
    ordered: list[str] = []
    for cid in raw:
        if cid in seen:
            continue
        seen.add(cid)
        ordered.append(cid)
    return ordered


def _compute_tsne_xy(*, X: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    if X.ndim != 2:
        raise RuntimeError(f"Expected X to be 2D, got {X.shape}")
    n = int(X.shape[0])
    p = float(perplexity)
    if p <= 1:
        raise ValueError("perplexity must be > 1")
    if p >= (n - 1) / 3:
        p = max(5.0, min(30.0, (n - 1) / 3 - 1))

    from sklearn.manifold import TSNE

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=p,
        random_state=int(seed),
        max_iter=1500,
        verbose=1,
    )
    return tsne.fit_transform(X.astype(np.float32, copy=False))


def _morgan_fingerprints_from_smiles(smiles: np.ndarray, radius: int, nbits: int) -> np.ndarray:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except Exception as e:  # pragma: no cover
        raise RuntimeError("RDKit is required to compute Morgan fingerprints (failed to import rdkit).") from e

    fps = np.zeros((int(smiles.shape[0]), int(nbits)), dtype=np.float32)
    for i, smi in enumerate(smiles.tolist()):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise RuntimeError(f"Invalid SMILES at index {i}: {smi!r}")
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(nbits))
        DataStructs.ConvertToNumpyArray(bv, fps[i])
    return fps


def _plot_tsne_with_marks(
    *,
    blind_xy: np.ndarray,
    known_inactive_xy: np.ndarray,
    known_active_xy: np.ndarray,
    out_path: Path,
    title: str,
    dpi: int,
    highlight_xy: np.ndarray | None = None,
    highlight_label: str | None = None,
    correct_green_xy: np.ndarray,
    correct_red_xy: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title)

    ax.scatter(
        blind_xy[:, 0],
        blind_xy[:, 1],
        s=6,
        c="#9aa0a6",
        alpha=0.35,
        linewidths=0,
        label=f"Blind set (n={blind_xy.shape[0]})",
        zorder=1,
    )
    ax.scatter(
        known_inactive_xy[:, 0],
        known_inactive_xy[:, 1],
        s=10,
        c="#1f77b4",
        alpha=0.75,
        linewidths=0,
        label=f"Known inactives (n={known_inactive_xy.shape[0]})",
        zorder=2,
    )
    ax.scatter(
        known_active_xy[:, 0],
        known_active_xy[:, 1],
        s=12,
        c="#d62728",
        alpha=0.85,
        linewidths=0,
        label=f"Known actives (n={known_active_xy.shape[0]})",
        zorder=3,
    )

    if highlight_xy is not None:
        lbl = highlight_label or f"Highlighted (n={highlight_xy.shape[0]})"
        ax.scatter(
            highlight_xy[:, 0],
            highlight_xy[:, 1],
            s=26,
            c="#2ca02c",
            alpha=0.95,
            linewidths=0,
            label=lbl,
            zorder=4,
        )

    if correct_green_xy.size:
        ax.scatter(
            correct_green_xy[:, 0],
            correct_green_xy[:, 1],
            s=90,
            marker="x",
            c="#2ca02c",
            alpha=0.98,
            linewidths=2.2,
            label=f"Correct answers (in top-100) (n={correct_green_xy.shape[0]})",
            zorder=10,
        )
    if correct_red_xy.size:
        ax.scatter(
            correct_red_xy[:, 0],
            correct_red_xy[:, 1],
            s=90,
            marker="x",
            c="#ff0000",
            alpha=0.98,
            linewidths=2.2,
            label=f"Correct answers (missed by top-100) (n={correct_red_xy.shape[0]})",
            zorder=10,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=True, markerscale=1.4, loc="best")
    ax.grid(False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _warn_missing_ids(*, label: str, wanted: list[str], available: np.ndarray) -> None:
    available_set = set(available.tolist())
    missing = [cid for cid in wanted if cid not in available_set]
    if missing:
        print(f"WARNING: {label}: {len(missing)} IDs not found: first few: {missing[:10]}")


def main() -> int:
    args = _parse_args()

    correct_ids = _read_id_list(Path(args.correct_answers_txt))
    shapesim_top100_ids = _read_id_list(Path(args.shapesim_top100_txt))
    correct_set = set(correct_ids)
    top100_set = set(shapesim_top100_ids)

    print(f"Correct answers: {len(correct_ids)}")
    print(f"CHEESE ShapeSim top-100 IDs: {len(shapesim_top100_ids)}")
    print(f"Correct ∩ top-100: {len(correct_set & top100_set)}")

    # (1) CHEESE ShapeSim-embedding t-SNE (with ShapeSim top-100 overlay)
    shapesim_blind = np.load(Path(args.shapesim_blind_npz), allow_pickle=True)
    shapesim_known = np.load(Path(args.shapesim_known_npz), allow_pickle=True)
    shapesim_blind_ids = shapesim_blind["compound_id"].astype(object, copy=False)
    shapesim_blind_emb = shapesim_blind["embedding"].astype(np.float32, copy=False)
    shapesim_known_emb = shapesim_known["embedding"].astype(np.float32, copy=False)
    shapesim_activity_class = shapesim_known["activity_class"].astype(np.int8, copy=False)

    _warn_missing_ids(label="ShapeSim blind set / correct answers", wanted=correct_ids, available=shapesim_blind_ids)

    shapesim_X = np.vstack([shapesim_blind_emb, shapesim_known_emb])
    shapesim_xy = _compute_tsne_xy(X=shapesim_X, perplexity=float(args.perplexity), seed=int(args.seed))

    b_n = shapesim_blind_emb.shape[0]
    s_blind_xy = shapesim_xy[:b_n, :]
    s_known_xy = shapesim_xy[b_n:, :]
    s_inactive_xy = s_known_xy[shapesim_activity_class == 0, :]
    s_active_xy = s_known_xy[shapesim_activity_class == 1, :]

    s_highlight = np.isin(shapesim_blind_ids, np.asarray(shapesim_top100_ids, dtype=object))
    s_correct = np.isin(shapesim_blind_ids, np.asarray(correct_ids, dtype=object))
    s_correct_green = s_correct & s_highlight
    s_correct_red = s_correct & ~s_highlight

    out_dir = Path(args.out_dir)
    shapesim_out = out_dir / "cheese_shapesim_tsne_shapesim_top100_correct_answers.png"
    _plot_tsne_with_marks(
        blind_xy=s_blind_xy,
        known_inactive_xy=s_inactive_xy,
        known_active_xy=s_active_xy,
        out_path=shapesim_out,
        title="CHEESE ShapeSim t-SNE (blind vs known) + CHEESE ShapeSim top-100",
        dpi=int(args.dpi),
        highlight_xy=s_blind_xy[s_highlight, :],
        highlight_label=f"CHEESE ShapeSim top-100 (n={int(s_highlight.sum())})",
        correct_green_xy=s_blind_xy[s_correct_green, :],
        correct_red_xy=s_blind_xy[s_correct_red, :],
    )
    print(f"Wrote: {shapesim_out}")

    shapesim_clean_out = out_dir / "cheese_shapesim_tsne_shapesim_top100_correct_answers_clean.png"
    _plot_tsne_with_marks(
        blind_xy=s_blind_xy,
        known_inactive_xy=s_inactive_xy,
        known_active_xy=s_active_xy,
        out_path=shapesim_clean_out,
        title="CHEESE ShapeSim t-SNE (blind vs known)",
        dpi=int(args.dpi),
        highlight_xy=None,
        highlight_label=None,
        correct_green_xy=s_blind_xy[s_correct_green, :],
        correct_red_xy=s_blind_xy[s_correct_red, :],
    )
    print(f"Wrote: {shapesim_clean_out}")

    # (2) Morgan fingerprint t-SNE (with ShapeSim top-100 overlay)
    morgan_blind = np.load(Path(args.morgan_blind_npz), allow_pickle=True)
    morgan_known = np.load(Path(args.morgan_known_npz), allow_pickle=True)
    morgan_blind_ids = morgan_blind["compound_id"].astype(object, copy=False)
    morgan_blind_smiles = morgan_blind["smiles"].astype(object, copy=False)
    morgan_known_smiles = morgan_known["smiles"].astype(object, copy=False)
    morgan_activity_class = morgan_known["activity_class"].astype(np.int8, copy=False)

    _warn_missing_ids(label="Morgan blind set / correct answers", wanted=correct_ids, available=morgan_blind_ids)

    mb = _morgan_fingerprints_from_smiles(
        morgan_blind_smiles, radius=int(args.morgan_radius), nbits=int(args.morgan_nbits)
    )
    mk = _morgan_fingerprints_from_smiles(
        morgan_known_smiles, radius=int(args.morgan_radius), nbits=int(args.morgan_nbits)
    )
    morgan_X = np.vstack([mb, mk])
    morgan_xy = _compute_tsne_xy(X=morgan_X, perplexity=float(args.perplexity), seed=int(args.seed))

    b_n = mb.shape[0]
    m_blind_xy = morgan_xy[:b_n, :]
    m_known_xy = morgan_xy[b_n:, :]
    m_inactive_xy = m_known_xy[morgan_activity_class == 0, :]
    m_active_xy = m_known_xy[morgan_activity_class == 1, :]

    m_highlight = np.isin(morgan_blind_ids, np.asarray(shapesim_top100_ids, dtype=object))
    m_correct = np.isin(morgan_blind_ids, np.asarray(correct_ids, dtype=object))
    m_correct_green = m_correct & m_highlight
    m_correct_red = m_correct & ~m_highlight

    morgan_out = out_dir / f"morgan_r{int(args.morgan_radius)}_b{int(args.morgan_nbits)}_tsne_shapesim_top100_correct_answers.png"
    _plot_tsne_with_marks(
        blind_xy=m_blind_xy,
        known_inactive_xy=m_inactive_xy,
        known_active_xy=m_active_xy,
        out_path=morgan_out,
        title=f"Morgan r={int(args.morgan_radius)} b={int(args.morgan_nbits)} t-SNE (blind vs known) + CHEESE ShapeSim top-100",
        dpi=int(args.dpi),
        highlight_xy=m_blind_xy[m_highlight, :],
        highlight_label=f"CHEESE ShapeSim top-100 (n={int(m_highlight.sum())})",
        correct_green_xy=m_blind_xy[m_correct_green, :],
        correct_red_xy=m_blind_xy[m_correct_red, :],
    )
    print(f"Wrote: {morgan_out}")

    morgan_clean_out = out_dir / f"morgan_r{int(args.morgan_radius)}_b{int(args.morgan_nbits)}_tsne_shapesim_top100_correct_answers_clean.png"
    _plot_tsne_with_marks(
        blind_xy=m_blind_xy,
        known_inactive_xy=m_inactive_xy,
        known_active_xy=m_active_xy,
        out_path=morgan_clean_out,
        title=f"Morgan r={int(args.morgan_radius)} b={int(args.morgan_nbits)} t-SNE (blind vs known)",
        dpi=int(args.dpi),
        highlight_xy=None,
        highlight_label=None,
        correct_green_xy=m_blind_xy[m_correct_green, :],
        correct_red_xy=m_blind_xy[m_correct_red, :],
    )
    print(f"Wrote: {morgan_clean_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
