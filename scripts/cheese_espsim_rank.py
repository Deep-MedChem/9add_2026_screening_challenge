#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _ensure_transformers_import_works() -> None:
    # `transformers` may import audio helpers that try to import `soxr`.
    # In some envs `soxr` is present but binary-incompatible with NumPy 2.x,
    # which breaks text-only model loading. Stub it out.
    sys.modules.setdefault("soxr", types.ModuleType("soxr"))


def _ensure_sentence_transformers_import_works() -> None:
    # Compatibility shim: some `sentence-transformers` versions expect
    # `huggingface_hub.cached_download`, which was removed in newer hub versions.
    try:
        import huggingface_hub  # noqa: F401

        if not hasattr(huggingface_hub, "cached_download") and hasattr(huggingface_hub, "hf_hub_download"):
            huggingface_hub.cached_download = huggingface_hub.hf_hub_download  # type: ignore[attr-defined]
    except Exception:
        pass


def _default_model_path() -> Path:
    # repo_root/cheese_models/cheese-models/espsim
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "cheese_models" / "cheese-models" / "espsim"


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
    # Sanity: IDs should be unique
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
    # Sanity: IDs should be unique
    ids = [x.chembl_id for x in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ChEMBL IDs in {path}")
    return items


def _embed_smiles(model_path: Path, smiles: list[str], *, device: str, batch_size: int) -> np.ndarray:
    _ensure_sentence_transformers_import_works()
    _ensure_transformers_import_works()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path), device=device)
    emb = model.encode(
        smiles,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity == dot product
        batch_size=batch_size,
        show_progress_bar=True,
    )
    if emb.ndim != 2:
        raise RuntimeError(f"Unexpected embedding shape: {emb.shape}")
    return emb.astype(np.float32, copy=False)


def _write_id_list(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CHEESE ESPSIM embeddings and rank blind-set compounds by similarity to known actives."
    )
    parser.add_argument("--model", type=str, default=str(_default_model_path()), help="Path to local CHEESE model dir.")
    parser.add_argument(
        "--run-name",
        type=str,
        default="cheese_espsim",
        help="Name prefix for outputs (default: cheese_espsim).",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for SentenceTransformer (default: cpu).")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for embedding (default: 256).")
    parser.add_argument(
        "--metric",
        type=str,
        choices=["max", "mean", "both"],
        default="both",
        help="Which similarity aggregation to produce submissions for (default: both).",
    )
    parser.add_argument("--top-n", type=int, default=100, help="How many IDs to output (default: 100).")
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    data_dir = Path(args.data_dir)
    blind_path = data_dir / "blind_set" / "blind_set.smi"
    known_path = data_dir / "known_compounds" / "chembl_all_class.smi"
    if not blind_path.exists():
        raise FileNotFoundError(f"Missing extracted blind set: {blind_path}")
    if not known_path.exists():
        raise FileNotFoundError(f"Missing extracted known set: {known_path}")

    run_name = "".join(c if (c.isalnum() or c in {"-", "_", "."}) else "_" for c in args.run_name).strip("_")
    if not run_name:
        raise ValueError("--run-name must not be empty")

    out_root = Path(args.out_dir)
    emb_dir = out_root / "embeddings"
    out_csv = out_root / "outputs" / f"{run_name}_blind_scores.csv"
    top_max_path = out_root / "submissions" / f"{run_name}_top{args.top_n}_maxcos.txt"
    top_mean_path = out_root / "submissions" / f"{run_name}_top{args.top_n}_meancos.txt"

    blind = _read_blind_smi(blind_path)
    known = _read_known_all_class(known_path)
    actives = [k for k in known if k.activity_class == 1]
    if not actives:
        raise RuntimeError("No actives found in known set (activity_class==1).")

    print(f"Model: {model_path}")
    print(f"Blind set: {len(blind)} compounds")
    print(f"Known set: {len(known)} compounds ({len(actives)} actives)")

    blind_smiles = [x.smiles for x in blind]
    known_smiles = [x.smiles for x in known]

    print("\nEmbedding blind set…")
    blind_emb = _embed_smiles(model_path, blind_smiles, device=args.device, batch_size=args.batch_size)
    print("\nEmbedding known set…")
    known_emb = _embed_smiles(model_path, known_smiles, device=args.device, batch_size=args.batch_size)

    emb_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        emb_dir / f"{run_name}_blind_set.npz",
        compound_id=np.array([x.compound_id for x in blind], dtype=object),
        smiles=np.array(blind_smiles, dtype=object),
        embedding=blind_emb,
    )
    np.savez_compressed(
        emb_dir / f"{run_name}_known_all_class.npz",
        chembl_id=np.array([x.chembl_id for x in known], dtype=object),
        smiles=np.array(known_smiles, dtype=object),
        activity_class=np.array([x.activity_class for x in known], dtype=np.int8),
        embedding=known_emb,
    )

    active_mask = np.array([k.activity_class == 1 for k in known], dtype=bool)
    active_emb = known_emb[active_mask]

    print("\nScoring blind set vs actives (cosine)…")
    sims = blind_emb @ active_emb.T
    max_cos = sims.max(axis=1)
    mean_cos = sims.mean(axis=1)

    blind_ids = np.array([x.compound_id for x in blind], dtype=object)
    top_n = int(args.top_n)
    if top_n <= 0:
        raise ValueError("--top-n must be > 0")

    wrote_any = False
    if args.metric in {"max", "both"}:
        top_max_ids = blind_ids[np.argsort(-max_cos)[:top_n]].tolist()
        _write_id_list(top_max_path, top_max_ids)
        wrote_any = True
    if args.metric in {"mean", "both"}:
        top_mean_ids = blind_ids[np.argsort(-mean_cos)[:top_n]].tolist()
        _write_id_list(top_mean_path, top_mean_ids)
        wrote_any = True
    if not wrote_any:
        raise RuntimeError("Nothing to write (unexpected metric setting).")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["compound_id", "max_cos_to_actives", "mean_cos_to_actives"])
        for cid, a, b in zip(blind_ids.tolist(), max_cos.tolist(), mean_cos.tolist(), strict=True):
            w.writerow([cid, f"{a:.8f}", f"{b:.8f}"])

    print("\nWrote:")
    if args.metric in {"max", "both"}:
        print(f"- {top_max_path}  (top {top_n} by max cosine)")
    if args.metric in {"mean", "both"}:
        print(f"- {top_mean_path} (top {top_n} by mean cosine)")
    print(f"- {out_csv} (all blind scores)")
    print(f"- {emb_dir / f'{run_name}_blind_set.npz'} (embeddings)")
    print(f"- {emb_dir / f'{run_name}_known_all_class.npz'} (embeddings)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
