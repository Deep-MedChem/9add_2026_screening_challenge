#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScoreRow:
    compound_id: str
    mean_cos_to_actives: float


def _read_mean_cos_scores(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        required = {"compound_id", "mean_cos_to_actives"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

        scores: dict[str, float] = {}
        for row in reader:
            cid = (row.get("compound_id") or "").strip()
            val = (row.get("mean_cos_to_actives") or "").strip()
            if not cid or not val:
                continue
            scores[cid] = float(val)
    return scores


def _ranks_from_scores(scores: dict[str, float]) -> dict[str, int]:
    # Rank 1 is best (highest score).
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return {cid: i + 1 for i, (cid, _score) in enumerate(ordered)}


def _read_id_list(path: Path) -> list[str]:
    ids = [l.strip() for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return ids


def _write_id_list(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble ESPSIM + ShapeSim mean-cosine baselines.")
    parser.add_argument(
        "--espsim-scores",
        type=str,
        default="9add_2026_screening_challenge/outputs/cheese_espsim_blind_scores.csv",
        help="CSV with columns including mean_cos_to_actives (ESPSIM).",
    )
    parser.add_argument(
        "--shapesim-scores",
        type=str,
        default="9add_2026_screening_challenge/outputs/cheese_shapesim_blind_scores.csv",
        help="CSV with columns including mean_cos_to_actives (ShapeSim).",
    )
    parser.add_argument(
        "--espsim-top",
        type=str,
        default="9add_2026_screening_challenge/submissions/cheese_espsim_top100_meancos.txt",
        help="Top list used for voting (ESPSIM).",
    )
    parser.add_argument(
        "--shapesim-top",
        type=str,
        default="9add_2026_screening_challenge/submissions/cheese_shapesim_top100_meancos.txt",
        help="Top list used for voting (ShapeSim).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="9add_2026_screening_challenge/submissions",
        help="Output directory for submission files.",
    )
    parser.add_argument("--top-n", type=int, default=100, help="How many IDs to output (default: 100).")
    parser.add_argument(
        "--w-espsim",
        type=float,
        default=1.0,
        help="Weight for ESPSIM rank in weighted-rank ensemble (default: 1.0).",
    )
    parser.add_argument(
        "--w-shapesim",
        type=float,
        default=1.0,
        help="Weight for ShapeSim rank in weighted-rank ensemble (default: 1.0).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    top_n = int(args.top_n)
    if top_n <= 0:
        raise ValueError("--top-n must be > 0")

    espsim_scores = _read_mean_cos_scores(Path(args.espsim_scores))
    shapesim_scores = _read_mean_cos_scores(Path(args.shapesim_scores))

    if espsim_scores.keys() != shapesim_scores.keys():
        # Expect same blind set. If not, still proceed with intersection for safety.
        common = espsim_scores.keys() & shapesim_scores.keys()
        if not common:
            raise RuntimeError("No overlapping compound IDs between ESPSIM and ShapeSim score files.")
        espsim_scores = {k: espsim_scores[k] for k in common}
        shapesim_scores = {k: shapesim_scores[k] for k in common}

    espsim_rank = _ranks_from_scores(espsim_scores)
    shapesim_rank = _ranks_from_scores(shapesim_scores)

    # (a) Mean rank ensemble over the full blind set.
    mean_rank_items = []
    for cid in espsim_scores.keys():
        r = (espsim_rank[cid] + shapesim_rank[cid]) / 2.0
        # tie-break by average raw score, then best individual rank
        mean_score = (espsim_scores[cid] + shapesim_scores[cid]) / 2.0
        best_rank = min(espsim_rank[cid], shapesim_rank[cid])
        mean_rank_items.append((r, -mean_score, best_rank, cid))
    mean_rank_items.sort()
    mean_rank_top = [cid for *_rest, cid in mean_rank_items[:top_n]]

    # (a2) Weighted-rank ensemble over the full blind set.
    w_espsim = float(args.w_espsim)
    w_shapesim = float(args.w_shapesim)
    if w_espsim < 0 or w_shapesim < 0:
        raise ValueError("Weights must be >= 0")
    if (w_espsim + w_shapesim) == 0:
        raise ValueError("At least one weight must be > 0")
    w_sum = w_espsim + w_shapesim
    w_espsim /= w_sum
    w_shapesim /= w_sum

    weighted_rank_items = []
    for cid in espsim_scores.keys():
        r = w_espsim * espsim_rank[cid] + w_shapesim * shapesim_rank[cid]
        weighted_score = w_espsim * espsim_scores[cid] + w_shapesim * shapesim_scores[cid]
        best_rank = min(espsim_rank[cid], shapesim_rank[cid])
        weighted_rank_items.append((r, -weighted_score, best_rank, cid))
    weighted_rank_items.sort()
    weighted_rank_top = [cid for *_rest, cid in weighted_rank_items[:top_n]]

    # (b) Voting ensemble: prioritize compounds appearing in both top lists,
    # then one-vote compounds ordered by mean rank.
    espsim_top = _read_id_list(Path(args.espsim_top))
    shapesim_top = _read_id_list(Path(args.shapesim_top))
    vote_count: dict[str, int] = {}
    for cid in espsim_top:
        vote_count[cid] = vote_count.get(cid, 0) + 1
    for cid in shapesim_top:
        vote_count[cid] = vote_count.get(cid, 0) + 1

    voting_items = []
    big_rank = len(espsim_rank) + 10_000
    for cid, votes in vote_count.items():
        r1 = espsim_rank.get(cid, big_rank)
        r2 = shapesim_rank.get(cid, big_rank)
        avg_rank = (r1 + r2) / 2.0
        best_rank = min(r1, r2)
        voting_items.append((-votes, avg_rank, best_rank, cid))
    voting_items.sort()
    voting_top = [cid for *_rest, cid in voting_items[:top_n]]

    out_dir = Path(args.out_dir)
    out_mean_rank = out_dir / f"ensemble_espsim_shapesim_top{top_n}_meanrank.txt"
    out_weighted_rank = (
        out_dir / f"ensemble_espsim_shapesim_top{top_n}_weightedrank_wshape{args.w_shapesim:g}_wesp{args.w_espsim:g}.txt"
    )
    out_voting = out_dir / f"ensemble_espsim_shapesim_top{top_n}_voting.txt"
    _write_id_list(out_mean_rank, mean_rank_top)
    _write_id_list(out_weighted_rank, weighted_rank_top)
    _write_id_list(out_voting, voting_top)

    print("Wrote:")
    print(f"- {out_mean_rank}")
    print(f"- {out_weighted_rank}")
    print(f"- {out_voting}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
