# 9add_2026_screening_challenge

Task description (challenge web app): https://shiny.imtm.cz/apps/polishp/9add/

## Simplified task overview

This is a **virtual screening** challenge against **DYRK1B** (PDB ID **8C2Z**).

Goal: pick **100** compounds from the blind set, trying to maximize the number of **true actives** (the leaderboard score is **recall**; ties are broken by submission time). Each team may submit up to **10** submissions.

Submission format: **exactly 100 unique compound IDs** (one ID per line). If you submit fewer than 100 picks, the remaining lines may be filled with random IDs.

**“Active” definition (for the provided labeled/known compounds):**
- Active if **IC50 / Ki / Kd ≤ 1 µM** OR **inhibition ≥ 90%**
- Compounds with conflicting activity annotations were removed

The provided docking grid is centered on the reference ligand **AZ191** (with an additional **6 Å** margin mentioned in the task text).

## Data files in `data/`

### Protein + docking grid
- `data/8C2Z_protein.pdb`: prepared protein structure (source format; includes hydrogens/protonation states).
- `data/8C2Z_protein.pdbqt`: docking-ready protein format (compatible with Vina/Smina/Gnina-family tools).
- `data/grid.txt`: docking box parameters (as provided):
  - `size_x = 26.34`, `size_y = 21.74`, `size_z = 22.18`
  - `center_x = -19.56`, `center_y = -27.76`, `center_z = 1.86`

### Blind set (the screening library you must rank)
- `data/blind_set.zip`
  - `blind_set.smi` (3,365 rows): **2 columns**: `SMILES<TAB>ID` where IDs look like `MOL_1`, `MOL_2`, …
  - `blind_set.sdf` (3,365 molecules): SDF where the **record name line is the ID** (e.g. `MOL_1`); no extra SD tags are present

### Known compounds (labeled actives/inactives and activity measurements)
- `data/known_compounds.zip` (ChEMBL-derived; datasets may overlap)
  - `chembl_all_class.smi` (1,891 rows): columns:
    - `Smiles`, `Molecule ChEMBL ID`, `activity_class` (0 = inactive, 1 = active)
  - `chembl_all_class.sdf` (1,891 molecules): SDF where the **record name line is the ChEMBL ID**; no extra SD tags are present
  - `chembl_IC50.smi` (745 rows), `chembl_Ki.smi` (225 rows), `chembl_Kd.smi` (41 rows), `chembl_Inhibition.smi` (585 rows): columns:
    - `Smiles`, `Molecule ChEMBL ID`, `Standard Type`, `activity`, `Standard Units`, `activity_sd`, `p_activity`
  - matching `.sdf` files for each subset (same molecule ID convention as above; no extra SD tags)

### Note on `chembl_all_class.zip`
- `data/chembl_all_class.zip` is **not** a valid zip in this repo (it’s an HTML document). The actual `chembl_all_class.smi/.sdf` are inside `data/known_compounds.zip`.

## CHEESE (ESPSIM) similarity baselines

This repo includes a small script to embed SMILES with the local CHEESE ESPSIM model and rank the blind set by cosine similarity to the **known actives**.

Run:
- `python 9add_2026_screening_challenge/scripts/cheese_espsim_rank.py --device cpu`

Outputs:
- `9add_2026_screening_challenge/submissions/cheese_espsim_top100_maxcos.txt:1` (top-100 by **max** cosine similarity to any active)
- `9add_2026_screening_challenge/submissions/cheese_espsim_top100_meancos.txt:1` (top-100 by **mean** cosine similarity to actives)
- `9add_2026_screening_challenge/outputs/cheese_espsim_blind_scores.csv:1` (all blind compounds scored)
- `9add_2026_screening_challenge/embeddings/cheese_espsim_blind_set.npz:1` / `9add_2026_screening_challenge/embeddings/cheese_espsim_known_all_class.npz:1` (saved embeddings)

Additional baselines:
- CHEESE ShapeSim (mean-cosine): `9add_2026_screening_challenge/submissions/cheese_shapesim_top100_meancos.txt:1`
- Morgan fingerprints (mean Tanimoto to actives): `9add_2026_screening_challenge/submissions/morgan_r2_b2048_top100_meantanimo.txt:1`

Ensembles (ESPSIM + ShapeSim mean-cos):
- Mean-rank ensemble: `9add_2026_screening_challenge/submissions/ensemble_espsim_shapesim_top100_meanrank.txt:1`
- Weighted-rank ensemble (ShapeSim weighted higher): `9add_2026_screening_challenge/submissions/ensemble_espsim_shapesim_top100_weightedrank_wshape0.53_wesp0.4.txt:1`
- Voting ensemble (union of top-100 lists): `9add_2026_screening_challenge/submissions/ensemble_espsim_shapesim_top100_voting.txt:1`


## Results of submitted baselines
- Random selection: 0.07 hit rate
- CHEESE maxcos: 0.2 hit rate
- CHEESE meancos: 0.4 hit rate
