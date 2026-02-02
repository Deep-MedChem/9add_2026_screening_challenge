"""Train LightGBM classifier for DYRK1B virtual screening.

Features: Morgan fingerprints (ECFP4, 2048 bits) + 9 RDKit descriptors.
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED, rdFingerprintGenerator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")

# Constants
FP_RADIUS = 2
FP_NBITS = 2048
DESCRIPTOR_NAMES = [
    "MolLogP", "TPSA", "MolWt", "NumHAcceptors", "NumHDonors",
    "NumRotatableBonds", "NumAromaticRings", "HeavyAtomCount", "qed",
]

DESCRIPTOR_FUNCS = {}
for _name in DESCRIPTOR_NAMES:
    if _name == "qed":
        DESCRIPTOR_FUNCS[_name] = QED.qed
    else:
        DESCRIPTOR_FUNCS[_name] = getattr(Descriptors, _name)


# Feature generation 

def smiles_to_features(smiles_list: list[str]) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    """Compute Morgan FP matrix + descriptor matrix for a list of SMILES.

    Returns (fp_matrix, desc_matrix, valid_mask) where valid_mask[i] is True
    if SMILES[i] could be parsed.
    """
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)

    fps, descs, valid = [], [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid.append(False)
            continue
        valid.append(True)

        # Fingerprint → dense array
        fp = fpgen.GetFingerprint(mol)
        arr = np.zeros(FP_NBITS, dtype=np.uint8)
        for bit in fp.GetOnBits():
            arr[bit] = 1
        fps.append(arr)

        # Descriptors
        descs.append([func(mol) for func in DESCRIPTOR_FUNCS.values()])

    fp_matrix = np.vstack(fps) if fps else np.empty((0, FP_NBITS))
    desc_matrix = np.array(descs, dtype=np.float64) if descs else np.empty((0, len(DESCRIPTOR_NAMES)))
    return fp_matrix, desc_matrix, valid


def build_feature_matrix(fp_matrix: np.ndarray, desc_matrix: np.ndarray) -> np.ndarray:
    return np.hstack([fp_matrix, desc_matrix])


# Data loading

def load_known_compounds(path: str = "data/known_compounds.zip") -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        df = pd.read_csv(io.TextIOWrapper(z.open("chembl_all_class.smi")), sep="\t")
    df.columns = ["smiles", "chembl_id", "is_active"]
    return df


def load_blind_set(path: str = "data/blind_set.zip") -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        df = pd.read_csv(
            io.TextIOWrapper(z.open("blind_set.smi")),
            sep="\t",
            header=None,
            names=["smiles", "compound_id"],
        )
    return df


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="submissions/lgbm_submission.txt",
                        help="Path for top-100 compound IDs output")
    args = parser.parse_args()

    # Step 1: Load data
    print("Loading known compounds...")
    known = load_known_compounds()
    print(f"  {len(known)} compounds ({known['is_active'].sum()} active, "
          f"{(~known['is_active'].astype(bool)).sum()} inactive)")

    print("Loading blind set...")
    blind = load_blind_set()
    print(f"  {len(blind)} compounds")

    # Step 2: Generate features
    print("Generating features for known compounds...")
    fp_known, desc_known, valid_known = smiles_to_features(known["smiles"].tolist())
    known_valid = known[valid_known].reset_index(drop=True)
    X_known = build_feature_matrix(fp_known, desc_known)
    y_known = known_valid["is_active"].values
    print(f"  Valid: {len(known_valid)}/{len(known)}, features shape: {X_known.shape}")

    print("Generating features for blind set...")
    fp_blind, desc_blind, valid_blind = smiles_to_features(blind["smiles"].tolist())
    blind_valid = blind[valid_blind].reset_index(drop=True)
    X_blind = build_feature_matrix(fp_blind, desc_blind)
    print(f"  Valid: {len(blind_valid)}/{len(blind)}, features shape: {X_blind.shape}")

    # Step 3: Train LightGBM with 5-fold stratified CV
    print("\nTraining LightGBM with 5-fold stratified CV...")
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": 42,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(X_known))
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_known, y_known)):
        X_tr, X_val = X_known[train_idx], X_known[val_idx]
        y_tr, y_val = y_known[train_idx], y_known[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = probs
        auc = roc_auc_score(y_val, probs)
        fold_aucs.append(auc)
        print(f"  Fold {fold + 1}: AUC = {auc:.4f}")

    cv_auc = roc_auc_score(y_known, oof_probs)
    print(f"\n  CV ROC-AUC (OOF): {cv_auc:.4f}")
    print(f"  Mean fold AUC:    {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    # Step 4: Train final model on all data and score blind set
    print("\nTraining final model on all data...")
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X_known, y_known)

    blind_probs = final_model.predict_proba(X_blind)[:, 1]
    blind_valid["pred_prob"] = blind_probs
    blind_ranked = blind_valid.sort_values("pred_prob", ascending=False).reset_index(drop=True)

    # Step 5: Save blind set scores
    scores_path = "data/lgbm_blind_scores.csv"
    blind_ranked[["compound_id", "pred_prob"]].to_csv(scores_path, index=False)
    print(f"\nSaved blind set scores to {scores_path}")

    # Step 6: Save top-100 compound IDs
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    top100 = blind_ranked.head(100)["compound_id"].tolist()
    with open(args.output, "w") as f:
        for cid in top100:
            f.write(f"{cid}\n")
    print(f"\nSaved top-100 compound IDs to {args.output}")

    # Print top-10 preview
    print("\nTop-10 predicted compounds:")
    for i, row in blind_ranked.head(10).iterrows():
        print(f"  {i + 1:3d}. {row['compound_id']}  (p={row['pred_prob']:.4f})")


if __name__ == "__main__":
    main()
