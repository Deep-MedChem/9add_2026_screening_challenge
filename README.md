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


## Hosted CHEESE REAL searches

The resumable Jobs API workflow searches ENAMINE-REAL using ESP and shape similarity,
with `accurate` quality and 1,000 neighbors per input molecule. It requires Python
with `requests` and RDKit, plus tmux for the launchers. Authentication uses
`CHEESE_API_KEY` from the environment, or the configured 1Password credential via
`with-keyring --profile personal op read`.

From this repository directory, launch the October Enamine search with:

```bash
bash scripts/start_cheese_enamine_tmux.sh
tmux attach -t cheese-enamine-oct2026
```

This uses `data/enamine_all_r0_top100_oct2026.smi`, stores resumable state under
`results/cheese_enamine_all_r0_top100_oct2026_run/`, and automatically validates and
exports `results/cheese_enamine_all_r0_top100_oct2026.zip` after success. The archive
contains two gzipped CSVs, the original input, and an `INFO.md`. The run directory's
`exit_code` records the combined search/export outcome (zero means success).

Check progress without attaching:

```bash
python scripts/search_cheese_real.py --out results/cheese_enamine_all_r0_top100_oct2026_run --status
```

To resume an interrupted run directly, including automatic export:

```bash
bash scripts/start_cheese_enamine_tmux.sh --worker
```

The September ChEMBL launcher is `bash scripts/start_cheese_real_tmux.sh`; it uses
`data/chembl33_all_r0_top100_sep2026.smi` and writes directly under `results/`.
For other inputs, use `scripts/search_cheese_real.py --input PATH --out DIRECTORY`
with a separate output directory for each input/settings combination. The standalone
packager accepts `scripts/export_cheese_search.py --run DIRECTORY --input PATH
--package PACKAGE_DIRECTORY` and creates `PACKAGE_DIRECTORY.zip`.

Small query inputs are versioned. Generated search payloads, checkpoints, logs,
and delivery archives under `results/` are ignored by Git.

## Results of submitted baselines
- Random selection: 0.07 hit rate
- CHEESE maxcos: 0.2 hit rate
- CHEESE meancos: 0.4 hit rate

## Final submission results

| Submission ID | Score | Private Note | Submission file |
| ---: | ---: | --- | --- |
| 1 | 0.07 | Random Baseline / Test Submission | [9add_2026_screening_challenge/submissions/random_baseline_seed_920261.txt](submissions/random_baseline_seed_920261.txt) |
| 2 | 0.67 | LGBM+CHEESE consensus | [9add_2026_screening_challenge/submissions/lgbm_cheese_consensus.txt](submissions/lgbm_cheese_consensus.txt) |
| 3 | 0.2 | cheese_espsim_top100_maxcos | [9add_2026_screening_challenge/submissions/cheese_espsim_top100_maxcos.txt](submissions/cheese_espsim_top100_maxcos.txt) |
| 4 | 0.4 | cheese_espsim_top100_meancos | [9add_2026_screening_challenge/submissions/cheese_espsim_top100_meancos.txt](submissions/cheese_espsim_top100_meancos.txt) |
| 5 | 0.53 | cheese_shapesim_top100_meancos | [9add_2026_screening_challenge/submissions/cheese_shapesim_top100_meancos.txt](submissions/cheese_shapesim_top100_meancos.txt) |
| 6 | 0.07 | Mean Tanimoto | [9add_2026_screening_challenge/submissions/morgan_r2_b2048_top100_meantanimo.txt](submissions/morgan_r2_b2048_top100_meantanimo.txt) |
| 7 | 0.47 | Mean Rank ESP + Shapesim | [9add_2026_screening_challenge/submissions/ensemble_espsim_shapesim_top100_meanrank.txt](submissions/ensemble_espsim_shapesim_top100_meanrank.txt) |
| 8 | 0.47 | Voting ensemble ESP + Shapesim | [9add_2026_screening_challenge/submissions/ensemble_espsim_shapesim_top100_voting.txt](submissions/ensemble_espsim_shapesim_top100_voting.txt) |
| 9 | 0 | Glide Docking top100 | [9add_2026_screening_challenge/submissions/glide_top100_IDs.txt](submissions/glide_top100_IDs.txt) |
| 10 | 0.33 | LGBM | _N/A (submission file not in repo)_ |
