#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class Hit:
    compound_id: str
    smiles: str
    embedding: np.ndarray


@dataclass(frozen=True)
class Active:
    chembl_id: str
    smiles: str
    embedding: np.ndarray


@dataclass(frozen=True)
class PairRow:
    compound_id: str
    smiles: str
    chembl_id: str
    chembl_smiles: str
    shape_sim: float
    esp_sim: float
    ic50_nM: Optional[float]


def _read_id_list(path: Path) -> list[str]:
    raw = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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


def _load_hits(blind_npz: Path, selection_ids: list[str]) -> list[Hit]:
    data = np.load(blind_npz, allow_pickle=True)
    ids = data["compound_id"].astype(object, copy=False).tolist()
    smiles = data["smiles"].astype(object, copy=False).tolist()
    emb = data["embedding"].astype(np.float32, copy=False)
    if emb.ndim != 2:
        raise RuntimeError(f"Unexpected blind embedding shape: {emb.shape}")

    by_id: dict[str, int] = {cid: i for i, cid in enumerate(ids)}
    missing = [cid for cid in selection_ids if cid not in by_id]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} selection IDs from blind embeddings: first few: {missing[:10]}")

    hits: list[Hit] = []
    for cid in selection_ids:
        i = by_id[cid]
        hits.append(Hit(compound_id=cid, smiles=str(smiles[i]), embedding=emb[i]))
    return hits


def _load_actives(known_npz: Path) -> list[Active]:
    data = np.load(known_npz, allow_pickle=True)
    ids = data["chembl_id"].astype(object, copy=False).tolist()
    smiles = data["smiles"].astype(object, copy=False).tolist()
    activity_class = data["activity_class"].astype(np.int8, copy=False)
    emb = data["embedding"].astype(np.float32, copy=False)
    if emb.ndim != 2:
        raise RuntimeError(f"Unexpected known embedding shape: {emb.shape}")
    if activity_class.shape[0] != emb.shape[0]:
        raise RuntimeError(
            f"known activity_class length does not match embeddings: {activity_class.shape[0]} != {emb.shape[0]}"
        )

    actives: list[Active] = []
    for i, is_active in enumerate(activity_class.tolist()):
        if int(is_active) != 1:
            continue
        actives.append(Active(chembl_id=str(ids[i]), smiles=str(smiles[i]), embedding=emb[i]))

    if not actives:
        raise RuntimeError("No actives found in known embeddings (activity_class==1).")
    return actives


def _pick_most_similar_actives(hits: list[Hit], actives: list[Active]) -> dict[str, Active]:
    # Assumes normalized embeddings, so cosine == dot product.
    hit_X = np.stack([h.embedding for h in hits], axis=0).astype(np.float32, copy=False)
    active_X = np.stack([a.embedding for a in actives], axis=0).astype(np.float32, copy=False)
    sims = hit_X @ active_X.T
    best_idx = sims.argmax(axis=1)
    out: dict[str, Active] = {}
    for h, j in zip(hits, best_idx.tolist(), strict=True):
        out[h.compound_id] = actives[int(j)]
    return out


def _load_ic50_nM_by_chembl_id(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}

    import csv

    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            chembl_id = (row.get("Molecule ChEMBL ID") or "").strip()
            unit = (row.get("Standard Units") or "").strip()
            activity = (row.get("activity") or "").strip()
            if not chembl_id or not activity or unit != "nM":
                continue
            try:
                out[chembl_id] = float(activity)
            except ValueError:
                continue
    return out


def _require_rdkit():
    try:
        from rdkit import Chem  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("RDKit is required to generate conformers and write SDF files.") from e


def _require_espsim():
    try:
        import espsim  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing python package 'espsim'. Install with: python -m pip install espsim"
        ) from e


def _embed_mol_3d(smiles: str, *, n_confs: int, seed: int):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)

    n_confs = max(1, int(n_confs))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 0
    params.useRandomCoords = True

    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))
    if not conf_ids:
        # Fallback: try one conformer with a different seed.
        params.randomSeed = int(seed) + 1337
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params))
    if not conf_ids:
        raise RuntimeError("Failed to embed any conformers.")

    AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
    return mol, conf_ids


def _set_conf_positions(conf, xyz: np.ndarray) -> None:
    # RDKit 2022.09 lacks Conformer.SetPositions.
    for atom_idx, (x, y, z) in enumerate(xyz.tolist()):
        conf.SetAtomPosition(int(atom_idx), (float(x), float(y), float(z)))


def _pick_best_shape_pair(prb_mol, prb_conf_ids: list[int], ref_mol, ref_conf_ids: list[int]) -> tuple[int, int, float]:
    from espsim import GetShapeSim
    from rdkit.Chem import rdMolAlign, rdMolDescriptors

    prb_crippen = rdMolDescriptors._CalcCrippenContribs(prb_mol)
    ref_crippen = rdMolDescriptors._CalcCrippenContribs(ref_mol)

    best_shape = -1.0
    best_i = int(prb_conf_ids[0])
    best_j = int(ref_conf_ids[0])

    for i in prb_conf_ids:
        prb_conf = prb_mol.GetConformer(int(i))
        orig = prb_conf.GetPositions().copy()
        for j in ref_conf_ids:
            rdMolAlign.GetCrippenO3A(
                prb_mol,
                ref_mol,
                prb_crippen,
                ref_crippen,
                prbCid=int(i),
                refCid=int(j),
            ).Align()
            shape = float(GetShapeSim(prb_mol, ref_mol, prbCid=int(i), refCid=int(j)))
            if shape > best_shape:
                best_shape = shape
                best_i = int(i)
                best_j = int(j)
            _set_conf_positions(prb_conf, orig)

    return best_i, best_j, float(best_shape)


def _align_and_score(prb_mol, ref_mol, *, prb_cid: int, ref_cid: int) -> tuple[float, float]:
    from espsim import GetEspSim, GetShapeSim
    from rdkit.Chem import rdMolAlign, rdMolDescriptors

    prb_crippen = rdMolDescriptors._CalcCrippenContribs(prb_mol)
    ref_crippen = rdMolDescriptors._CalcCrippenContribs(ref_mol)
    rdMolAlign.GetCrippenO3A(
        prb_mol,
        ref_mol,
        prb_crippen,
        ref_crippen,
        prbCid=int(prb_cid),
        refCid=int(ref_cid),
    ).Align()

    shape = float(GetShapeSim(prb_mol, ref_mol, prbCid=int(prb_cid), refCid=int(ref_cid)))
    esp = float(
        GetEspSim(
            prb_mol,
            ref_mol,
            prbCid=int(prb_cid),
            refCid=int(ref_cid),
            partialCharges="gasteiger",
            integrate="gauss",
            nMC=1,
            marginMC=10,
        )
    )
    return shape, esp


def _mol_with_single_conformer(mol, *, cid: int):
    from rdkit import Chem

    out = Chem.Mol(mol)
    conf = mol.GetConformer(int(cid))
    out.RemoveAllConformers()
    out.AddConformer(Chem.Conformer(conf), assignId=True)
    return out


def _write_pair_sdf(
    *,
    out_path: Path,
    compound_id: str,
    chembl_id: str,
    hit_mol,
    active_mol,
    shape_sim: float,
    esp_sim: float,
    ic50_nM: Optional[float],
) -> None:
    from rdkit import Chem

    out_path.parent.mkdir(parents=True, exist_ok=True)

    hit = Chem.Mol(hit_mol)
    hit.SetProp("_Name", compound_id)
    hit.SetProp("role", "cheese_hit")
    hit.SetProp("paired_chembl_id", chembl_id)
    hit.SetProp("shapesim", f"{shape_sim:.6f}")
    hit.SetProp("espsim", f"{esp_sim:.6f}")
    if ic50_nM is not None and math.isfinite(ic50_nM):
        hit.SetProp("ic50_chembl_nM", f"{ic50_nM:g}")

    act = Chem.Mol(active_mol)
    act.SetProp("_Name", chembl_id)
    act.SetProp("role", "chembl_active")
    act.SetProp("paired_compound_id", compound_id)
    act.SetProp("shapesim", f"{shape_sim:.6f}")
    act.SetProp("espsim", f"{esp_sim:.6f}")
    if ic50_nM is not None and math.isfinite(ic50_nM):
        act.SetProp("ic50_chembl_nM", f"{ic50_nM:g}")

    w = Chem.SDWriter(str(out_path))
    w.write(hit)
    w.write(act)
    w.close()


def _safe_stem(s: str) -> str:
    return "".join(c if (c.isalnum() or c in {"-", "_", "."}) else "_" for c in s).strip("_") or "item"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pair CHEESE ShapeSim top hits with their most similar known actives, compute 3D similarity, and export aligned conformer pairs."
    )
    p.add_argument(
        "--selection",
        type=str,
        default="9add_2026_screening_challenge/submissions/cheese_shapesim_top100_meancos.txt",
        help="Text file with compound IDs (one per line).",
    )
    p.add_argument(
        "--blind-npz",
        type=str,
        default="9add_2026_screening_challenge/embeddings/cheese_shapesim_blind_set.npz",
        help="CHEESE ShapeSim blind-set embeddings (npz with compound_id, smiles, embedding).",
    )
    p.add_argument(
        "--known-npz",
        type=str,
        default="9add_2026_screening_challenge/embeddings/cheese_shapesim_known_all_class.npz",
        help="CHEESE ShapeSim known-set embeddings (npz with chembl_id, smiles, activity_class, embedding).",
    )
    p.add_argument(
        "--chembl-ic50",
        type=str,
        default="9add_2026_screening_challenge/data/known_compounds/chembl_IC50.smi",
        help="Optional TSV table mapping ChEMBL IDs to IC50 in nM.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs/paired_conformers_cheese_shapesim",
        help="Output directory (default: outputs/paired_conformers_cheese_shapesim).",
    )
    p.add_argument("--n-confs-hit", type=int, default=10, help="Number of conformers for each hit (default: 10).")
    p.add_argument(
        "--n-confs-active", type=int, default=10, help="Number of conformers for each active (default: 10)."
    )
    p.add_argument("--seed", type=int, default=920261, help="Random seed for 3D embedding (default: 920261).")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of pairs to process (0 = all selection IDs).",
    )
    return p.parse_args()


def _iter_pairs(selection_ids: list[str], hits: list[Hit], best_active_by_hit: dict[str, Active]) -> Iterable[tuple[Hit, Active]]:
    hit_by_id = {h.compound_id: h for h in hits}
    for cid in selection_ids:
        h = hit_by_id[cid]
        yield h, best_active_by_hit[cid]


def main() -> int:
    args = _parse_args()
    _require_rdkit()
    _require_espsim()

    selection_path = Path(args.selection)
    selection_ids = _read_id_list(selection_path)
    if not selection_ids:
        raise RuntimeError(f"No IDs found in selection file: {selection_path}")

    limit = int(args.limit)
    if limit < 0:
        raise ValueError("--limit must be >= 0")
    if limit:
        selection_ids = selection_ids[:limit]

    hits = _load_hits(Path(args.blind_npz), selection_ids)
    actives = _load_actives(Path(args.known_npz))
    best_active_by_hit = _pick_most_similar_actives(hits, actives)
    ic50_by_chembl = _load_ic50_nM_by_chembl_id(Path(args.chembl_ic50))

    out_dir = Path(args.out_dir)
    out_pairs_dir = out_dir / "conformer_pairs"
    out_csv = out_dir / "paired_conformers.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pairs_dir.mkdir(parents=True, exist_ok=True)

    cache_active_3d: dict[str, tuple[object, list[int]]] = {}

    rows: list[PairRow] = []
    total = len(selection_ids)
    for idx, (hit, active) in enumerate(_iter_pairs(selection_ids, hits, best_active_by_hit), start=1):
        prefix = f"[{idx:>3}/{total}]"
        print(f"{prefix} {hit.compound_id} -> {active.chembl_id}", flush=True)

        try:
            hit_mol, hit_conf_ids = _embed_mol_3d(
                hit.smiles, n_confs=int(args.n_confs_hit), seed=int(args.seed) + idx * 17
            )
            if active.chembl_id in cache_active_3d:
                active_mol, active_conf_ids = cache_active_3d[active.chembl_id]
            else:
                active_mol, active_conf_ids = _embed_mol_3d(
                    active.smiles,
                    n_confs=int(args.n_confs_active),
                    seed=int(args.seed) + 10_000 + idx * 31,
                )
                cache_active_3d[active.chembl_id] = (active_mol, active_conf_ids)

            best_i, best_j, _best_shape = _pick_best_shape_pair(
                hit_mol, hit_conf_ids, active_mol, active_conf_ids
            )
            shape_sim, esp_sim = _align_and_score(hit_mol, active_mol, prb_cid=best_i, ref_cid=best_j)

            hit_out = _mol_with_single_conformer(hit_mol, cid=best_i)
            active_out = _mol_with_single_conformer(active_mol, cid=best_j)

            ic50 = ic50_by_chembl.get(active.chembl_id)
            pair_path = out_pairs_dir / f"{_safe_stem(hit.compound_id)}__{_safe_stem(active.chembl_id)}.sdf"
            _write_pair_sdf(
                out_path=pair_path,
                compound_id=hit.compound_id,
                chembl_id=active.chembl_id,
                hit_mol=hit_out,
                active_mol=active_out,
                shape_sim=shape_sim,
                esp_sim=esp_sim,
                ic50_nM=ic50,
            )

            rows.append(
                PairRow(
                    compound_id=hit.compound_id,
                    smiles=hit.smiles,
                    chembl_id=active.chembl_id,
                    chembl_smiles=active.smiles,
                    shape_sim=float(shape_sim),
                    esp_sim=float(esp_sim),
                    ic50_nM=ic50,
                )
            )
        except Exception as e:
            print(f"{prefix} WARNING: failed for {hit.compound_id} -> {active.chembl_id}: {e}", file=sys.stderr)
            rows.append(
                PairRow(
                    compound_id=hit.compound_id,
                    smiles=hit.smiles,
                    chembl_id=active.chembl_id,
                    chembl_smiles=active.smiles,
                    shape_sim=float("nan"),
                    esp_sim=float("nan"),
                    ic50_nM=ic50_by_chembl.get(active.chembl_id),
                )
            )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "smiles", "id_chembl", "smiles_chembl", "shapesim", "espsim", "ic50_chembl"])
        for r in rows:
            ic50 = "" if r.ic50_nM is None else f"{r.ic50_nM:g}"
            shape = "" if not math.isfinite(r.shape_sim) else f"{r.shape_sim:.6f}"
            esp = "" if not math.isfinite(r.esp_sim) else f"{r.esp_sim:.6f}"
            w.writerow([r.compound_id, r.smiles, r.chembl_id, r.chembl_smiles, shape, esp, ic50])

    print("\nWrote:")
    print(f"- {out_csv}")
    print(f"- {out_pairs_dir} ({len(selection_ids)} expected SDF pairs; failures may reduce this count)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

