# Prompt Sensitivity and Label Budgets in Vision-Language Models for Waste Classification

Code and experimental evidence for the paper accepted at **ACLing 2026**, for publication in Procedia Computer Science.

We study whether selecting prompts is a useful way to spend limited annotation budgets. Experiments compare frozen OpenAI CLIP, SigLIP, and OpenCLIP checkpoints on RealWaste and TrashNet, distinguishing changes caused by prompt wording from the benefit of searching for a prompt using labeled examples.

## Main findings

- Wording effects depend on the checkpoint, metric, and manually authored prompt variants.
- Under matched annotation budgets, target-specific selection improves macro-F1 over pooled descriptions in **2 of 24** primary comparisons and performs worse in **1**, after Holm correction. The remaining comparisons do not establish equivalence.
- Visual-only nearest-centroid classification is competitive in several RealWaste settings.

The budget study uses 4, 8, 16, or 32 labels per class, 30 paired source-pool draws, and eight comparison recipes.

## Verify saved results on CPU

Use Python 3.10–3.13. Run these commands from the repository root; a GPU and model downloads are not required for this path.

```bash
python -m venv .venv
source .venv/bin/activate
python -m zipfile -e evidence.zip .
python scripts/restore_measurements.py
python -m pip install -r requirements-audit.txt
export PYTHONPATH=src
python -m pytest -q
python scripts/summarize_budget_experiment.py --results measurements/budget/budget_study --out reanalysis/budget
```

On Windows PowerShell, replace the activation command with `.venv\Scripts\Activate.ps1` and the `export` command with `$env:PYTHONPATH = "src"`.

`evidence.zip` contains the measured results, predictions, split manifests, prompts, frozen features, reference tables, audit records, and optional input templates. Extract it once before running tests or experiments. The restoration script verifies and expands the large `.csv.gz` files without overwriting modified evidence. Extracted evidence directories are ignored by Git; keep `evidence.zip` committed.

### Expected outputs

The summary command writes:

- `reanalysis/budget/method_summary.csv`: means and standard deviations for every recipe and setting.
- `reanalysis/budget/paired_comparisons.csv`: paired differences, pointwise intervals, and adjusted primary p-values.

Filter the second table to `primary=True`: it contains 24 comparisons, with two positive and one negative differences having `primary_holm_p < 0.05`.

### Full consistency checks

After completing setup, run the following Bash loop to check budgets, selections, cross-validation calculations, and test predictions for both datasets:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

for dataset in realwaste trashnet
do
    python scripts/audit_budget_results.py --results measurements/budget/budget_study/$dataset
    python scripts/audit_budget_features.py --results measurements/budget/budget_study/$dataset --features measurements/budget/feature_audit
    python scripts/audit_budget_test_features.py --results measurements/budget/budget_study/$dataset --features measurements/budget/feature_audit
done
```

Full audits may take several minutes. They reconstruct **720 CV grids, 5,760 method runs, and 4,196,160 test decisions** from saved features and predictions. This checks arithmetic and accounting; it does not independently repeat image encoding. Audit reports are written beside the measurements.

## Run fresh experiments

Fresh inference requires the original images, model dependencies, and access to download pretrained weights. A GPU is recommended.

- [Budget-study notebook](notebooks/Budget_Experiment_Colab.ipynb): run all three checkpoints with shared labeled pools and inner four-fold selection. Upload the repository ZIP, set the two dataset paths, and choose a fresh output directory.
- [Historical experiment notebook](notebooks/Reproduce_Paper_Colab.ipynb): reproduce the earlier linguistic controls, fixed-family checkpoint comparisons, and CLIP few-shot experiments.

Both notebooks unpack `evidence.zip` during setup. For command-line inference, install `requirements.txt` and follow the [experiment protocol](docs/PROTOCOL.md#budget-study). Use a new output directory when changing settings; historical and budget results must remain separate.

## Repository layout

| Path | Purpose |
|---|---|
| `src/wastevlm/` | Prompt inventories, encoders, scoring, sampling, and adaptation |
| `scripts/` | Experiment runners, statistical summaries, and audits |
| `tests/` | Offline protocol and numerical checks |
| `notebooks/` | Colab workflows for budget and historical experiments |
| `docs/PROTOCOL.md` | Detailed methods, historical commands, and validation scope |
| `evidence.zip` | Canonical evidence and input archive; extract before use |
| `CITATION.cff` | Citation metadata and author information |

After extraction, `measurements/` holds recorded runs, `reference/` holds historical reference inputs and summaries, and `verification/` and `revision_analysis/` hold derived checks. `research_inputs/` contains blank templates for optional future evaluations.

## Data and limitations

Download images from [RealWaste (UCI)](https://archive.ics.uci.edu/dataset/908/realwaste) and [TrashNet](https://github.com/garythung/trashnet). Dataset pixels and pretrained weights are not distributed here.

Historical and budget TrashNet analyses use different image memberships despite equal partition counts; the budget study groups known exact-pixel duplicates. All budget recipes share the same labeled images, but search capacity and ensemble sizes differ. Thirty source-pool draws reuse fixed test images and do not represent independent datasets. Perceptual-hash candidates remain unconfirmed, and independent-author or external-domain results are not included. Recognition scores do not establish safe disposal decisions. See the [protocol and validation details](docs/PROTOCOL.md#validation-status).

## Citation

Please cite **Prompt Sensitivity and Label Budgets in Vision-Language Models for Waste Classification**, accepted at ACLing 2026, when using this code or evidence. Author and software citation metadata are in [CITATION.cff](CITATION.cff). Publication details will be added when available.

## Authors

Salama Alsenaidi, Wadhha Alsulaimani, Shamsa Howaireb, Mahra Al Qubaisi, Mariam Alkatheeri, Abdelrhman Gaber, and Youssef Ibrahim. **The first five authors contributed equally.**

Computer Information Science, Higher Colleges of Technology, Abu Dhabi, United Arab Emirates.

Corresponding author: **Youssef Ibrahim** — [yibrahim@hct.ac.ae](mailto:yibrahim@hct.ac.ae).

## License

Code is distributed under the [MIT license](LICENSE). Original datasets, pretrained weights, and third-party packages retain their own licenses.
