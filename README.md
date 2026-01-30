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
