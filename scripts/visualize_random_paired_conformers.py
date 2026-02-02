#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PairInfo:
    hit_id: str
    chembl_id: str
    shape_sim: Optional[float]
    esp_sim: Optional[float]
    ic50_nM: Optional[float]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pick random aligned conformer pairs and render interactive 3D visualizations with optional surfaces."
    )
    p.add_argument(
        "--pairs-dir",
        type=str,
        default="outputs/paired_conformers_cheese_shapesim/conformer_pairs",
        help="Directory containing aligned pair SDFs (2 mols each).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs/paired_conformers_cheese_shapesim/visualizations_random10",
        help="Output directory for HTML visualizations.",
    )
    p.add_argument("--n", type=int, default=10, help="How many random pairs to pick (default: 10).")
    p.add_argument("--seed", type=int, default=920261, help="RNG seed for reproducible sampling.")
    p.add_argument(
        "--surface",
        action="store_true",
        help="Add transparent VDW surfaces for both molecules (may be slower in-browser).",
    )
    p.add_argument("--surface-opacity", type=float, default=0.30, help="Surface opacity (default: 0.30).")
    p.add_argument("--width", type=int, default=900, help="Viewer width in px (default: 900).")
    p.add_argument("--height", type=int, default=650, help="Viewer height in px (default: 650).")
    return p.parse_args()


def _read_pair_sdf(path: Path) -> tuple[list[object], PairInfo]:
    from rdkit import Chem

    suppl = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    mols = [m for m in suppl if m is not None]
    if len(mols) != 2:
        raise RuntimeError(f"Expected exactly 2 molecules in {path}, got {len(mols)}")

    hit = mols[0]
    act = mols[1]

    def _get_float_prop(mol, name: str) -> Optional[float]:
        if not mol.HasProp(name):
            return None
        v = mol.GetProp(name).strip()
        if not v:
            return None
        try:
            return float(v)
        except ValueError:
            return None

    info = PairInfo(
        hit_id=(hit.GetProp("_Name") if hit.HasProp("_Name") else path.stem),
        chembl_id=(act.GetProp("_Name") if act.HasProp("_Name") else ""),
        shape_sim=_get_float_prop(hit, "shapesim"),
        esp_sim=_get_float_prop(hit, "espsim"),
        ic50_nM=_get_float_prop(hit, "ic50_chembl_nM"),
    )
    return mols, info


def _write_html(
    *,
    out_path: Path,
    sdf_path: Path,
    mols: list[object],
    info: PairInfo,
    width: int,
    height: int,
    add_surface: bool,
    surface_opacity: float,
) -> None:
    import py3Dmol
    from rdkit import Chem

    v = py3Dmol.view(width=int(width), height=int(height))
    # Model 0: CHEESE hit (green), Model 1: ChEMBL active (magenta)
    for mol in mols:
        mb = Chem.MolToMolBlock(mol)
        v.addModel(mb, "mol")

    v.setBackgroundColor("white")
    v.setStyle({"model": 0}, {"stick": {"radius": 0.22, "color": "forestgreen"}, "sphere": {"scale": 0.22}})
    v.setStyle({"model": 1}, {"stick": {"radius": 0.22, "color": "magenta"}, "sphere": {"scale": 0.22}})

    if add_surface:
        op = float(surface_opacity)
        if op < 0 or op > 1:
            raise ValueError("--surface-opacity must be in [0, 1]")
        v.addSurface(py3Dmol.VDW, {"opacity": op, "color": "forestgreen"}, {"model": 0})
        v.addSurface(py3Dmol.VDW, {"opacity": op, "color": "magenta"}, {"model": 1})

    v.zoomTo()
    v.zoom(1.1)

    title = f"{info.hit_id} vs {info.chembl_id}".strip()
    subtitle_parts = []
    if info.shape_sim is not None:
        subtitle_parts.append(f"ShapeSim={info.shape_sim:.4f}")
    if info.esp_sim is not None:
        subtitle_parts.append(f"ESPSim={info.esp_sim:.4f}")
    if info.ic50_nM is not None:
        subtitle_parts.append(f"IC50={info.ic50_nM:g} nM")
    subtitle = " | ".join(subtitle_parts) if subtitle_parts else ""

    viewer_html = v._make_html()
    doc = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(title)}</title>
    <style>
      body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; }}
      .meta {{ margin: 10px 0 18px 0; color: #333; }}
      .path {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace; font-size: 13px; color: #555; }}
    </style>
  </head>
  <body>
    <h2>{html.escape(title)}</h2>
    <div class="meta">
      <div>{html.escape(subtitle)}</div>
      <div class="path">{html.escape(str(sdf_path))}</div>
      <div>Model 0 = hit (green), Model 1 = active (magenta)</div>
    </div>
    {viewer_html}
  </body>
</html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")


def main() -> int:
    args = _parse_args()

    pairs_dir = Path(args.pairs_dir)
    if not pairs_dir.exists():
        raise FileNotFoundError(f"Pairs dir not found: {pairs_dir}")
    sdf_files = sorted(pairs_dir.glob("*.sdf"))
    if not sdf_files:
        raise RuntimeError(f"No SDFs found in: {pairs_dir}")

    n = int(args.n)
    if n <= 0:
        raise ValueError("--n must be > 0")
    if n > len(sdf_files):
        raise ValueError(f"--n={n} is larger than available pairs ({len(sdf_files)})")

    random.seed(int(args.seed))
    picked = random.sample(sdf_files, n)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for sdf_path in picked:
        mols, info = _read_pair_sdf(sdf_path)
        out_path = out_dir / f"{sdf_path.stem}.html"
        _write_html(
            out_path=out_path,
            sdf_path=sdf_path,
            mols=mols,
            info=info,
            width=int(args.width),
            height=int(args.height),
            add_surface=bool(args.surface),
            surface_opacity=float(args.surface_opacity),
        )
        label = f"{info.hit_id} vs {info.chembl_id}".strip()
        index_rows.append((label, out_path.name))

    # Index page
    li = "\n".join(
        f'<li><a href="{html.escape(fname)}">{html.escape(label)}</a></li>' for (label, fname) in index_rows
    )
    index_html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Random Paired Conformer Visualizations</title>
    <style>
      body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; }}
      .meta {{ margin: 8px 0 16px 0; color: #333; }}
      .path {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace; font-size: 13px; color: #555; }}
    </style>
  </head>
  <body>
    <h2>Random paired conformers (n={n})</h2>
    <div class="meta">
      <div>Source: <span class="path">{html.escape(str(pairs_dir))}</span></div>
      <div>Seed: {int(args.seed)} | Surface: {str(bool(args.surface))} | Opacity: {float(args.surface_opacity):.2f}</div>
    </div>
    <ol>
      {li}
    </ol>
  </body>
</html>
"""
    (out_dir / "index.html").write_text(index_html, encoding="utf-8")
    (out_dir / "picked_files.txt").write_text("\n".join(str(p) for p in picked) + "\n", encoding="utf-8")

    print("Wrote:")
    print(f"- {out_dir / 'index.html'}")
    print(f"- {out_dir / 'picked_files.txt'}")
    print(f"- {out_dir}/*.html ({n} pair pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

