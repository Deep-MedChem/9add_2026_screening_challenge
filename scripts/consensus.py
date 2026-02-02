"""Consensus ranking combining LightGBM and CHEESE similarity scores."""

from __future__ import annotations

import os

import pandas as pd

LGBM_SCORES = "data/lgbm_blind_scores.csv"
CHEESE_SCORES = "data/cheese_shapesim_blind_scores.csv"
OUTPUT = "submissions/consensus_avg_rank.txt"

CHEESE_WEIGHT = 53
LGBM_WEIGHT = 33


def main():
    lgbm = pd.read_csv(LGBM_SCORES)
    cheese = pd.read_csv(CHEESE_SCORES)

    # Rank (1 = best)
    lgbm["lgbm_rank"] = lgbm["pred_prob"].rank(ascending=False).astype(int)
    cheese["cheese_rank"] = cheese["mean_cos_to_actives"].rank(ascending=False).astype(int)

    merged = lgbm[["compound_id", "lgbm_rank"]].merge(
        cheese[["compound_id", "cheese_rank"]], on="compound_id", how="inner"
    )

    merged["consensus_score"] = CHEESE_WEIGHT * merged["cheese_rank"] + LGBM_WEIGHT * merged["lgbm_rank"]
    merged = merged.sort_values("consensus_score").reset_index(drop=True)

    top100 = merged.head(100)["compound_id"].tolist()

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        for cid in top100:
            f.write(f"{cid}\n")
    print(f"Saved consensus top-100 to {OUTPUT}")

    # Summary
    lgbm_top100 = set(lgbm.nsmallest(100, "lgbm_rank")["compound_id"])
    cheese_top100 = set(cheese.nsmallest(100, "cheese_rank")["compound_id"])
    consensus_top100 = set(top100)

    print(f"\nOverlap: LGBM ∩ CHEESE = {len(lgbm_top100 & cheese_top100)}")
    print(f"         LGBM ∩ Consensus = {len(lgbm_top100 & consensus_top100)}")
    print(f"         CHEESE ∩ Consensus = {len(cheese_top100 & consensus_top100)}")

    print("\nTop-10 consensus compounds:")
    for i, row in merged.head(10).iterrows():
        print(f"  {i + 1:3d}. {row['compound_id']}  "
              f"(cheese_rank={row['cheese_rank']}, lgbm_rank={row['lgbm_rank']}, "
              f"score={row['consensus_score']})")


if __name__ == "__main__":
    main()
