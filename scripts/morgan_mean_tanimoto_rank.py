#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


@dataclass(frozen=True)
class BlindCompound:
    compound_id: str
    smiles: str


@dataclass(frozen=True)
class KnownCompound:
    chembl_id: str
    smiles: str
    activity_class: int  # 0/1


def _read_blind_smi(path: Path) -> list[BlindCompound]:
    items: list[BlindCompound] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Unexpected blind_set line (expected 'SMILES ID'): {line!r}")
            smiles, compound_id = parts[0], parts[1]
            items.append(BlindCompound(compound_id=compound_id, smiles=smiles))
    ids = [x.compound_id for x in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return items


def _read_known_all_class(path: Path) -> list[KnownCompound]:
    items: list[KnownCompound] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"Smiles", "Molecule ChEMBL ID", "activity_class"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        for row in reader:
            smiles = (row.get("Smiles") or "").strip()
            chembl_id = (row.get("Molecule ChEMBL ID") or "").strip()
            activity_class_raw = (row.get("activity_class") or "").strip()
            if not smiles or not chembl_id or activity_class_raw not in {"0", "1"}:
                continue
            items.append(
                KnownCompound(
                    chembl_id=chembl_id,
                    smiles=smiles,
                    activity_class=int(activity_class_raw),
                )
            )
    ids = [x.chembl_id for x in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ChEMBL IDs in {path}")
    return items


def _mol_from_smiles(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Standardize a bit: add Hs? For Morgan fingerprints, heavy-atom graph is typical; leave as-is.
    return mol


def _morgan_fp(mol: Chem.Mol, *, radius: int, n_bits: int) -> DataStructs.ExplicitBitVect:
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank blind-set compounds by mean Morgan-fingerprint Tanimoto similarity to known actives."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="9add_2026_screening_challenge/data",
        help="Base data dir containing extracted datasets.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="9add_2026_screening_challenge",
        help="Base output dir (default: challenge folder).",
    )
    parser.add_argument("--radius", type=int, default=2, help="Morgan radius (default: 2).")
    parser.add_argument("--n-bits", type=int, default=2048, help="Morgan bit vector size (default: 2048).")
    parser.add_argument("--top-n", type=int, default=100, help="How many IDs to output (default: 100).")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    blind_path = data_dir / "blind_set" / "blind_set.smi"
    known_path = data_dir / "known_compounds" / "chembl_all_class.smi"
    if not blind_path.exists():
        raise FileNotFoundError(f"Missing extracted blind set: {blind_path}")
    if not known_path.exists():
        raise FileNotFoundError(f"Missing extracted known set: {known_path}")

    radius = int(args.radius)
    n_bits = int(args.n_bits)
    top_n = int(args.top_n)
    if radius <= 0:
        raise ValueError("--radius must be > 0")
    if n_bits <= 0:
        raise ValueError("--n-bits must be > 0")
    if top_n <= 0:
        raise ValueError("--top-n must be > 0")

    out_root = Path(args.out_dir)
    out_csv = out_root / "outputs" / f"morgan_r{radius}_b{n_bits}_blind_mean_tanimoto.csv"
    out_ids = out_root / "submissions" / f"morgan_r{radius}_b{n_bits}_top{top_n}_meantanimo.txt"

    blind = _read_blind_smi(blind_path)
    known = _read_known_all_class(known_path)
    actives = [k for k in known if k.activity_class == 1]
    if not actives:
        raise RuntimeError("No actives found in known set (activity_class==1).")

    active_fps: list[DataStructs.ExplicitBitVect] = []
    active_ids: list[str] = []
    n_active_bad = 0
    for a in actives:
        mol = _mol_from_smiles(a.smiles)
        if mol is None:
            n_active_bad += 1
            continue
        active_fps.append(_morgan_fp(mol, radius=radius, n_bits=n_bits))
        active_ids.append(a.chembl_id)
    if not active_fps:
        raise RuntimeError("All active SMILES failed to parse; cannot compute similarities.")

    print(f"Actives: {len(actives)} total, {len(active_fps)} fingerprinted, {n_active_bad} failed SMILES")
    print(f"Blind: {len(blind)} compounds")
    print(f"Morgan params: radius={radius} n_bits={n_bits}")

    mean_sims = np.empty(len(blind), dtype=np.float32)
    n_blind_bad = 0
    for i, b in enumerate(blind):
        mol = _mol_from_smiles(b.smiles)
        if mol is None:
            mean_sims[i] = -1.0
            n_blind_bad += 1
            continue
        fp = _morgan_fp(mol, radius=radius, n_bits=n_bits)
        sims = DataStructs.BulkTanimotoSimilarity(fp, active_fps)
        mean_sims[i] = float(sum(sims) / len(sims)) if sims else -1.0

    print(f"Blind SMILES failed: {n_blind_bad}")

    blind_ids = np.array([b.compound_id for b in blind], dtype=object)
    top_ids = blind_ids[np.argsort(-mean_sims)[:top_n]].tolist()

    out_ids.parent.mkdir(parents=True, exist_ok=True)
    out_ids.write_text("\n".join(top_ids) + "\n", encoding="utf-8")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["compound_id", "mean_tanimoto_to_actives"])
        for cid, s in zip(blind_ids.tolist(), mean_sims.tolist(), strict=True):
            w.writerow([cid, f"{s:.8f}"])

    print("Wrote:")
    print(f"- {out_ids} (top {top_n} by mean Tanimoto to actives)")
    print(f"- {out_csv} (all blind mean Tanimoto scores)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

