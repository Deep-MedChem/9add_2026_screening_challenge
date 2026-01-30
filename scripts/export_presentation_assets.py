#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export presentation assets for the 9add 2026 screening challenge: "
            "(1) t-SNE plot copy, (2) t-SNE plot with ShapeSim top-100 overlay, "
            "(3) ShapeSim-embedding t-SNE (base + overlay), "
            "(4) Morgan-fingerprint t-SNE (base + overlay), "
            "(5) SDF subset for ShapeSim top-100."
        )
    )
    p.add_argument(
        "--selection-txt",
        type=str,
        default="9add_2026_screening_challenge/submissions/cheese_shapesim_top100_meancos.txt",
        help="Path to 100-line submission file (one compound ID per line).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="9add_2026_screening_challenge/outputs/presentation",
        help="Output directory for exported assets.",
    )

    # t-SNE inputs (reuse existing CHEESE ESPSIM t-SNE coordinates)
    p.add_argument(
        "--tsne-png",
        type=str,
        default="9add_2026_screening_challenge/outputs/cheese_espsim_tsne.png",
        help="Existing t-SNE PNG to copy (used as the 'current' plot).",
    )
    p.add_argument(
        "--tsne-csv",
        type=str,
        default="9add_2026_screening_challenge/outputs/cheese_espsim_tsne.csv",
        help="Existing t-SNE CSV coordinates written by cheese_espsim_tsne_plot.py.",
    )
    p.add_argument(
        "--blind-npz",
        type=str,
        default="9add_2026_screening_challenge/embeddings/cheese_espsim_blind_set.npz",
        help="Blind-set embeddings npz containing compound_id in the same order as the t-SNE CSV blind rows.",
    )
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
    p.add_argument("--dpi", type=int, default=250, help="Output DPI for generated plots.")
    p.add_argument("--perplexity", type=float, default=50.0, help="t-SNE perplexity (default: 50).")
    p.add_argument("--seed", type=int, default=920261, help="Random seed for t-SNE (default: 920261).")
    p.add_argument(
        "--lgbm-consensus-txt",
        type=str,
        default="9add_2026_screening_challenge/submissions/lgbm_cheese_consensus.txt",
        help="Path to ID list to highlight (LGBM CHEESE consensus).",
    )

    # SDF export
    p.add_argument(
        "--blind-sdf",
        type=str,
        default="9add_2026_screening_challenge/data/blind_set/blind_set.sdf",
        help="Input blind-set SDF to subset.",
    )
    return p.parse_args()


def _read_id_list(path: Path) -> list[str]:
    raw = []
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


def _load_tsne_coords(tsne_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blind_xy: list[tuple[float, float]] = []
    known_inactive_xy: list[tuple[float, float]] = []
    known_active_xy: list[tuple[float, float]] = []

    with tsne_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"set", "label", "tsne1", "tsne2"}
        if set(reader.fieldnames or []) != expected:
            raise RuntimeError(f"Unexpected t-SNE CSV columns: {reader.fieldnames} (expected {sorted(expected)})")
        for row in reader:
            s = row["set"]
            lbl = row["label"]
            x = float(row["tsne1"])
            y = float(row["tsne2"])
            if s == "blind":
                blind_xy.append((x, y))
            elif s == "known":
                if lbl == "inactive":
                    known_inactive_xy.append((x, y))
                elif lbl == "active":
                    known_active_xy.append((x, y))
                else:
                    raise RuntimeError(f"Unexpected known label in t-SNE CSV: {lbl!r}")
            else:
                raise RuntimeError(f"Unexpected set in t-SNE CSV: {s!r}")

    return (
        np.asarray(blind_xy, dtype=np.float32),
        np.asarray(known_inactive_xy, dtype=np.float32),
        np.asarray(known_active_xy, dtype=np.float32),
    )


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


def _plot_tsne(
    *,
    blind_xy: np.ndarray,
    known_inactive_xy: np.ndarray,
    known_active_xy: np.ndarray,
    out_path: Path,
    title: str,
    dpi: int,
    highlight_xy: np.ndarray | None = None,
    highlight_label: str | None = None,
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
    )
    ax.scatter(
        known_inactive_xy[:, 0],
        known_inactive_xy[:, 1],
        s=10,
        c="#1f77b4",
        alpha=0.75,
        linewidths=0,
        label=f"Known inactives (n={known_inactive_xy.shape[0]})",
    )
    ax.scatter(
        known_active_xy[:, 0],
        known_active_xy[:, 1],
        s=12,
        c="#d62728",
        alpha=0.85,
        linewidths=0,
        label=f"Known actives (n={known_active_xy.shape[0]})",
    )

    if highlight_xy is not None:
        lbl = highlight_label or f"Highlighted (n={highlight_xy.shape[0]})"
        ax.scatter(highlight_xy[:, 0], highlight_xy[:, 1], s=26, c="#2ca02c", alpha=0.95, linewidths=0, label=lbl)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=True, markerscale=1.6, loc="best")
    ax.grid(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _plot_tsne_with_highlight(
    *,
    blind_xy: np.ndarray,
    known_inactive_xy: np.ndarray,
    known_active_xy: np.ndarray,
    blind_compound_ids: np.ndarray,
    highlight_ids: list[str],
    out_path: Path,
    dpi: int,
) -> None:
    highlight_set = set(highlight_ids)
    m_highlight = np.isin(blind_compound_ids, np.asarray(list(highlight_set), dtype=object))
    n_highlight = int(m_highlight.sum())

    missing = [cid for cid in highlight_ids if cid not in set(blind_compound_ids.tolist())]
    if missing:
        print(f"WARNING: {len(missing)} highlight IDs not found in blind embeddings/SDF: first few: {missing[:10]}")
    _plot_tsne(
        blind_xy=blind_xy,
        known_inactive_xy=known_inactive_xy,
        known_active_xy=known_active_xy,
        out_path=out_path,
        title="CHEESE ESPSIM t-SNE (blind vs known) + CHEESE ShapeSim top-100",
        dpi=dpi,
        highlight_xy=blind_xy[m_highlight, :],
        highlight_label=f"CHEESE ShapeSim top-100 (n={n_highlight})",
    )


def _morgan_fingerprints_from_smiles(smiles: np.ndarray, radius: int, nbits: int) -> np.ndarray:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    fps = np.zeros((int(smiles.shape[0]), int(nbits)), dtype=np.float32)
    for i, smi in enumerate(smiles.tolist()):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise RuntimeError(f"Invalid SMILES at index {i}: {smi!r}")
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(nbits))
        DataStructs.ConvertToNumpyArray(bv, fps[i])
    return fps


def _mask_for_highlights(blind_ids: np.ndarray, highlight_ids: list[str], *, label: str) -> np.ndarray:
    if not highlight_ids:
        return np.zeros(blind_ids.shape[0], dtype=bool)
    highlight_set = set(highlight_ids)
    mask = np.isin(blind_ids, np.asarray(list(highlight_set), dtype=object))
    missing = [cid for cid in highlight_ids if cid not in set(blind_ids.tolist())]
    if missing:
        print(f"WARNING: {label}: {len(missing)} highlight IDs not found in blind IDs: first few: {missing[:10]}")
    return mask


def _export_sdf_subset(*, sdf_in: Path, ordered_ids: list[str], out_sdf: Path) -> None:
    try:
        from rdkit import Chem
    except Exception as e:  # pragma: no cover
        raise RuntimeError("RDKit is required to export the SDF subset (failed to import rdkit).") from e

    sel = set(ordered_ids)
    found: dict[str, "Chem.Mol"] = {}

    suppl = Chem.SDMolSupplier(str(sdf_in), sanitize=False, removeHs=False)
    for mol in suppl:
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        if name in sel:
            found[name] = mol

    missing = [cid for cid in ordered_ids if cid not in found]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} IDs from SDF {sdf_in}. "
            f"First few missing: {missing[:10]}"
        )

    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(out_sdf))
    for cid in ordered_ids:
        w.write(found[cid])
    w.close()


def main() -> int:
    args = _parse_args()

    selection_txt = Path(args.selection_txt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    highlight_ids = _read_id_list(selection_txt)
    if not highlight_ids:
        raise RuntimeError(f"No IDs found in selection file: {selection_txt}")

    # (a) t-SNE plots
    tsne_png = Path(args.tsne_png)
    if tsne_png.exists():
        shutil.copy2(tsne_png, out_dir / "cheese_espsim_tsne.png")
        print(f"Wrote: {out_dir / 'cheese_espsim_tsne.png'} (copied from {tsne_png})")
    else:
        print(f"WARNING: {tsne_png} not found; skipping copy of the 'current' plot.")

    blind_npz = Path(args.blind_npz)
    blind = np.load(blind_npz, allow_pickle=True)
    blind_compound_ids = blind["compound_id"].astype(object, copy=False)

    blind_xy, known_inactive_xy, known_active_xy = _load_tsne_coords(Path(args.tsne_csv))
    if blind_xy.shape[0] != blind_compound_ids.shape[0]:
        raise RuntimeError(
            "t-SNE CSV blind rows do not match blind embeddings compound_id length: "
            f"{blind_xy.shape[0]} != {blind_compound_ids.shape[0]}"
        )

    overlay_png = out_dir / "cheese_espsim_tsne_shapesim_top100.png"
    _plot_tsne_with_highlight(
        blind_xy=blind_xy,
        known_inactive_xy=known_inactive_xy,
        known_active_xy=known_active_xy,
        blind_compound_ids=blind_compound_ids,
        highlight_ids=highlight_ids,
        out_path=overlay_png,
        dpi=int(args.dpi),
    )
    print(f"Wrote: {overlay_png}")

    # (extra) LGBM consensus overlays (CHEESE ESPSIM t-SNE + Morgan t-SNE)
    lgbm_ids: list[str] = []
    lgbm_path = Path(args.lgbm_consensus_txt)
    if lgbm_path.exists():
        lgbm_ids = _read_id_list(lgbm_path)
    else:
        print(f"WARNING: {lgbm_path} not found; skipping LGBM consensus highlight plots.")

    if lgbm_ids:
        espsim_base_png = out_dir / "cheese_espsim_tsne_lgbm.png"
        _plot_tsne(
            blind_xy=blind_xy,
            known_inactive_xy=known_inactive_xy,
            known_active_xy=known_active_xy,
            out_path=espsim_base_png,
            title="CHEESE ESPSIM t-SNE (blind set vs known actives/inactives)",
            dpi=int(args.dpi),
        )
        print(f"Wrote: {espsim_base_png}")

        espsim_lgbm_overlay_png = out_dir / "cheese_espsim_tsne_lgbm_highlight.png"
        m_lgbm = _mask_for_highlights(blind_compound_ids, lgbm_ids, label="ESPSIM/LGBM")
        _plot_tsne(
            blind_xy=blind_xy,
            known_inactive_xy=known_inactive_xy,
            known_active_xy=known_active_xy,
            out_path=espsim_lgbm_overlay_png,
            title="CHEESE ESPSIM t-SNE (blind vs known) + LGBM CHEESE consensus",
            dpi=int(args.dpi),
            highlight_xy=blind_xy[m_lgbm, :],
            highlight_label=f"LGBM CHEESE consensus (n={int(m_lgbm.sum())})",
        )
        print(f"Wrote: {espsim_lgbm_overlay_png}")

    # (a, comparison) ShapeSim-embedding t-SNE (base + overlay)
    shapesim_blind = np.load(Path(args.shapesim_blind_npz), allow_pickle=True)
    shapesim_known = np.load(Path(args.shapesim_known_npz), allow_pickle=True)
    shapesim_blind_ids = shapesim_blind["compound_id"].astype(object, copy=False)
    shapesim_blind_emb = shapesim_blind["embedding"].astype(np.float32, copy=False)
    shapesim_known_emb = shapesim_known["embedding"].astype(np.float32, copy=False)
    shapesim_activity_class = shapesim_known["activity_class"].astype(np.int8, copy=False)

    shapesim_X = np.vstack([shapesim_blind_emb, shapesim_known_emb])
    shapesim_xy = _compute_tsne_xy(X=shapesim_X, perplexity=float(args.perplexity), seed=int(args.seed))

    b_n = shapesim_blind_emb.shape[0]
    s_blind_xy = shapesim_xy[:b_n, :]
    s_known_xy = shapesim_xy[b_n:, :]
    s_inactive_xy = s_known_xy[shapesim_activity_class == 0, :]
    s_active_xy = s_known_xy[shapesim_activity_class == 1, :]

    shapesim_base_png = out_dir / "cheese_shapesim_tsne.png"
    _plot_tsne(
        blind_xy=s_blind_xy,
        known_inactive_xy=s_inactive_xy,
        known_active_xy=s_active_xy,
        out_path=shapesim_base_png,
        title="CHEESE ShapeSim t-SNE (blind set vs known actives/inactives)",
        dpi=int(args.dpi),
    )
    print(f"Wrote: {shapesim_base_png}")

    shapesim_overlay_png = out_dir / "cheese_shapesim_tsne_shapesim_top100.png"
    s_highlight = np.isin(shapesim_blind_ids, np.asarray(highlight_ids, dtype=object))
    _plot_tsne(
        blind_xy=s_blind_xy,
        known_inactive_xy=s_inactive_xy,
        known_active_xy=s_active_xy,
        out_path=shapesim_overlay_png,
        title="CHEESE ShapeSim t-SNE (blind vs known) + CHEESE ShapeSim top-100",
        dpi=int(args.dpi),
        highlight_xy=s_blind_xy[s_highlight, :],
        highlight_label=f"CHEESE ShapeSim top-100 (n={int(s_highlight.sum())})",
    )
    print(f"Wrote: {shapesim_overlay_png}")

    # (a, comparison) Morgan fingerprint t-SNE (base + overlay)
    morgan_blind = np.load(Path(args.morgan_blind_npz), allow_pickle=True)
    morgan_known = np.load(Path(args.morgan_known_npz), allow_pickle=True)
    morgan_blind_ids = morgan_blind["compound_id"].astype(object, copy=False)
    morgan_blind_smiles = morgan_blind["smiles"].astype(object, copy=False)
    morgan_known_smiles = morgan_known["smiles"].astype(object, copy=False)
    morgan_activity_class = morgan_known["activity_class"].astype(np.int8, copy=False)

    mb = _morgan_fingerprints_from_smiles(morgan_blind_smiles, radius=int(args.morgan_radius), nbits=int(args.morgan_nbits))
    mk = _morgan_fingerprints_from_smiles(morgan_known_smiles, radius=int(args.morgan_radius), nbits=int(args.morgan_nbits))
    morgan_X = np.vstack([mb, mk])
    morgan_xy = _compute_tsne_xy(X=morgan_X, perplexity=float(args.perplexity), seed=int(args.seed))

    b_n = mb.shape[0]
    m_blind_xy = morgan_xy[:b_n, :]
    m_known_xy = morgan_xy[b_n:, :]
    m_inactive_xy = m_known_xy[morgan_activity_class == 0, :]
    m_active_xy = m_known_xy[morgan_activity_class == 1, :]

    morgan_base_png = out_dir / f"morgan_r{int(args.morgan_radius)}_b{int(args.morgan_nbits)}_tsne.png"
    _plot_tsne(
        blind_xy=m_blind_xy,
        known_inactive_xy=m_inactive_xy,
        known_active_xy=m_active_xy,
        out_path=morgan_base_png,
        title=f"Morgan r={int(args.morgan_radius)} b={int(args.morgan_nbits)} t-SNE (blind set vs known actives/inactives)",
        dpi=int(args.dpi),
    )
    print(f"Wrote: {morgan_base_png}")

    morgan_overlay_png = out_dir / f"morgan_r{int(args.morgan_radius)}_b{int(args.morgan_nbits)}_tsne_shapesim_top100.png"
    m_highlight = np.isin(morgan_blind_ids, np.asarray(highlight_ids, dtype=object))
    _plot_tsne(
        blind_xy=m_blind_xy,
        known_inactive_xy=m_inactive_xy,
        known_active_xy=m_active_xy,
        out_path=morgan_overlay_png,
        title=f"Morgan r={int(args.morgan_radius)} b={int(args.morgan_nbits)} t-SNE (blind vs known) + CHEESE ShapeSim top-100",
        dpi=int(args.dpi),
        highlight_xy=m_blind_xy[m_highlight, :],
        highlight_label=f"CHEESE ShapeSim top-100 (n={int(m_highlight.sum())})",
    )
    print(f"Wrote: {morgan_overlay_png}")

    if lgbm_ids:
        morgan_base_lgbm_png = out_dir / f"morgan_r{int(args.morgan_radius)}_b{int(args.morgan_nbits)}_tsne_lgbm.png"
        _plot_tsne(
            blind_xy=m_blind_xy,
            known_inactive_xy=m_inactive_xy,
            known_active_xy=m_active_xy,
            out_path=morgan_base_lgbm_png,
            title=f"Morgan r={int(args.morgan_radius)} b={int(args.morgan_nbits)} t-SNE (blind set vs known actives/inactives)",
            dpi=int(args.dpi),
        )
        print(f"Wrote: {morgan_base_lgbm_png}")

        morgan_lgbm_overlay_png = out_dir / f"morgan_r{int(args.morgan_radius)}_b{int(args.morgan_nbits)}_tsne_lgbm_highlight.png"
        m_lgbm = _mask_for_highlights(morgan_blind_ids, lgbm_ids, label="Morgan/LGBM")
        _plot_tsne(
            blind_xy=m_blind_xy,
            known_inactive_xy=m_inactive_xy,
            known_active_xy=m_active_xy,
            out_path=morgan_lgbm_overlay_png,
            title=f"Morgan r={int(args.morgan_radius)} b={int(args.morgan_nbits)} t-SNE (blind vs known) + LGBM CHEESE consensus",
            dpi=int(args.dpi),
            highlight_xy=m_blind_xy[m_lgbm, :],
            highlight_label=f"LGBM CHEESE consensus (n={int(m_lgbm.sum())})",
        )
        print(f"Wrote: {morgan_lgbm_overlay_png}")

    # (b) SDF selection
    out_sdf = out_dir / "cheese_shapesim_top100_meancos.sdf"
    _export_sdf_subset(sdf_in=Path(args.blind_sdf), ordered_ids=highlight_ids, out_sdf=out_sdf)
    print(f"Wrote: {out_sdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
