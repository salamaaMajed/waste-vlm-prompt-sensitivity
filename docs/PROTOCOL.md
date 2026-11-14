# Experiment protocols and validation

Extract `evidence.zip` and restore compressed measurements as described in the root README before running the commands below.

## Budget study

#### Status and research question

The completed RealWaste and TrashNet budget measurements are included under `measurements/budget/`. The protocol compares checkpoint-specific prompt/cache selection with alternatives using the same sampled labeled images. Frozen-feature audits reconstruct 720 CV grids and 5,760 method runs. Image encoding was not independently rerun; see `PROTOCOL.md`, validation status for scope.

#### Frozen default design

- Source datasets: RealWaste retains its historical seed-42 partition. TrashNet uses the resolved exact-pixel-group partition in `measurements/budget/budget_study/resolved_splits/trashnet.csv`. Historical and budget TrashNet memberships differ. Validation images are unused.
- Backbones: CLIP, SigLIP, OpenCLIP.
- Labels per class B: 4, 8, 16, 32; 30 sampled pools per budget. Pools are shared by all methods and checkpoints.
- Primary selection/reporting metric: macro-F1; accuracy is secondary.
- Primary contrast: target_selected_adapter minus pooled_ensemble_adapter.
- Selection: four stratified inner folds inside the B-per-class pool. Pool out-of-fold predictions before computing macro-F1. Then build the final cache from all B images per class.
- Grid: alpha=0 (text-only), or alpha in 1,5,10,20,50 with beta in 1,5,10,20; fixed text scale 100.
- Ties: sorted prompt ID, then text-only, then beta/positive-alpha grid order.
- Target selection jointly chooses among the nine fixed family/variant configurations and cache settings. Fixed-prompt baselines tune only cache settings in the identical pool.

The benchmark sampler knows class labels to form balanced episodes. This is an episodic label budget, not a model of the annotation effort needed to find rare classes in an unlabeled stream. The selection algorithm receives exactly B times the class count distinct image labels and no historical validation labels. Cross-validation reuses those labels; it does not introduce new ones.

#### Eight methods

1. class_name_adapter: one fixed class-name template.
2. name_ensemble_adapter: fixed name-preserving variant 1.
3. object_ensemble_adapter: fixed object-expression variant 1.
4. pooled_ensemble_adapter: the fixed, deduplicated union of all candidate prompt strings, encoded and averaged as one prototype per class. This is a strong within-pool text ensemble; it is not a claimed reproduction of Funk et al.'s exact 79-template/5-synonym setup.
5. clip_prompt_transferred_adapter: CLIP chooses a prompt ID using the same B labels; the target checkpoint retunes cache settings within those same labels. Extra model computation is recorded separately from labeling.
6. target_selected_adapter: jointly select the prompt and cache settings for the target checkpoint.
7. target_selected_text: select a fixed text classifier within the label budget, without a cache branch. This selection uses labels and is not label-free zero-shot selection.
8. nearest_centroid: use all B labeled image features per class, without a text prior or hyperparameter search.

The pooled ensemble uses more text strings than a fixed-family configuration; text and compute costs differ even though labeled-image budgets are equal. All strings and counts are released. The CLIP transfer method equals target selection when CLIP itself is the target, so that comparison is a sanity check, not independent evidence.

#### Run and inspect

Open Budget_Experiment_Colab.ipynb. Set dataset paths and a fresh study name; keep the primary metric and default budgets fixed for the first run. The notebook freezes settings/code/prompt hashes, screens image overlap, prepares image/text features, runs both datasets, audits the outputs, and exports the results. A prespecified cleaning rule regroups exact source pixel duplicates with seed 42 when they cross partitions, records the resolved split hash, and preserves the old manifests. Source/external exact overlap stops the run. Perceptual candidates remain unconfirmed until reviewed. Content-bound feature bundles can be reused only when image bytes and prompts match. Resolved Hugging Face revisions are recorded when exposed by the backend; a null revision is not a bitwise weight guarantee.

CLI example from this package root:

```bash
PYTHONPATH=src python scripts/run_budget_experiment.py --dataset realwaste --data /path/to/RealWaste --reference-split reference/splits/realwaste.csv --out budget_results/study_v1/realwaste
python scripts/audit_budget_results.py --results budget_results/study_v1/realwaste
python scripts/summarize_budget_experiment.py --results budget_results/study_v1 --out budget_results/study_v1/summary
```

The runner refuses nonempty output directories. After a failed partial run, keep its outputs, choose a new study name, and reuse matching feature bundles. Do not silently overwrite or combine partial runs. A smoke run with fewer draws is a software check, not the planned research experiment.

#### Output and uncertainty

The release logs the sampled labels and folds, all CV candidate scores, selected configurations, selected out-of-fold predictions, test predictions, per-run metrics, prompts, model environment and content hashes. audit_budget_results.py reconstructs budgets, choices, and metrics; inference cannot be independently regenerated without images and features/weights.

Summary intervals are pointwise 95% t intervals across paired support draws, conditional on fixed test images. Primary p values are Holm-adjusted across the supplied primary dataset/checkpoint/budget/evaluation cells; do not call the intervals multiplicity-adjusted. Historical benchmark extensions remain retrospective. Do not pool different source splits or author studies into the same summary or treat overlapping splits as independent datasets.

#### Optional prompt and external-data inputs

The three blank author forms and external manifests are in research_inputs/. Follow AUTHOR_BRIEF.md and EXTERNAL_DATA_BRIEF.md. These templates are blank; no independent-author or external-data results are included. Set PROMPTS or EXTERNAL manifests in the notebook only once the inputs are complete and their provenance is established.

Run scripts/audit_image_overlap.py on the real pixels. It screens every image pair using exact file/pixel hashes and a coarse dHash; perceptual neighbors require manual review and the screen is not exhaustive for scene similarity. Fill TRUE/FALSE for each candidate, then optionally create a new group split:

```bash
python scripts/make_group_split.py --hashes audit/hash_manifest.csv --reviewed-candidates audit/reviewed_candidate_pairs.csv --out group_split_v1.csv
```

The new split uses three training folds, one unused validation fold, and one test fold. Group-aware budget draws choose one image per sampled group per class; inner folds keep groups together. Every class needs enough distinct groups for its chosen budget. Supply this split in the notebook and use a new study name. This is a changed retrospective partition, not independent external validation.

#### Recheck CV arithmetic without model downloads

The notebook exports source feature bundles under feature_audit/ with relative image IDs. After extracting its output archive, use:

```bash
python scripts/audit_budget_features.py --results extracted/budget_study/realwaste --features extracted/feature_audit --first-draw-only
```

Omit --first-draw-only to recompute every CV grid. This checks the feature-level selection arithmetic, not the origin of the features or the correctness of the underlying image annotations. The complete CV and test output audit remains available through audit_budget_results.py.

## Historical experimental protocol

This protocol describes the historical reference experiments. For the current three-checkpoint total-budget study, see [BUDGET_PROTOCOL.md](PROTOCOL.md#budget-study).

- **Datasets:** RealWaste (9 labels, 4,752 images) and TrashNet (6 labels, 2,527 images). Source images are not included. Image class = parent folder. Image paths sorted lexicographically before splitting.
- **Partition:** `train_test_split(indices, test_size=.4, stratify=y, random_state=42)` followed by `train_test_split(rest, test_size=.5, stratify=y[rest], random_state=42)`. Train / validation / test = 60 / 20 / 20 approximately. Exact per-image manifests are in `reference/splits`.
- **Zero-shot:** OpenAI CLIP ViT-B/16, Google SigLIP Base Patch16-224, OpenCLIP ViT-B/16 LAION-2B. Nine fixed prompt sets: 3 manually authored sets for each of 3 families. Per-prompt L2 normalization; mean within class; L2 normalization of class prototype; image features L2 normalized. Class = argmax image/text cosine. Report all prompt sets, not selected best.
- **Few-shot:** Only OpenAI CLIP ViT-B/16. Historical text baseline: `a photo of {class} waste`. AE-Tip: 3 object-level expressions per class from `DESCRIPTIONS`, prefixed `a photo of `. Visual cache is identical. Cache score `exp(-beta*(1-cosine)) @ support_one_hot`; logits = `100 * image @ text_prototype.T + alpha * cache`. Search beta in [1,5,10,20], alpha in [1,5,10,20,50] using **baseline** validation accuracy, transfer the selected pair to AE-Tip. This is a matched, baseline-tuned intervention, not independently optimized comparison.
- **Support sampling:** 30 draws at each of 1,2,4,8,16 shots per class. RNG `default_rng(42 + 1009*repeat + 97*shots)`, classwise sampling from the fixed training pool. All support IDs recorded.
- **Uncertainty:** Paired 95% t intervals across 30 support draws, fixed held-out test images. These intervals are **unadjusted** for multiple comparisons and do not estimate between-dataset generalization.
- **Interpretation:** Object-oriented expressions are NOT strict semantic paraphrases of broad material categories. Backbones differ in pretraining data, architecture and objectives; this study cannot isolate causal sources of prompt ranking differences.
- **Leakage:** Original study screened 25 high-cosine cross-partition pairs, 8 involving held-out test images. A conservative zero-shot sensitivity analysis removed those 8; the observed accuracy gain persisted. That targeted audit does not establish global duplicate-free datasets. The code here reproduces the main experiments, not the original embedding-neighbor screening.
- **Important:** No external-paper numerical comparison should be called fair without matched splits, class ontology, and evaluation protocol. Reproducing this paper's tables does not prove environmental benefit or recycling-pathway correctness.

#### Historical linguistic controls and sensitivity analysis

- `scripts/run_controls.py --protocol experiment8` reproduces the original five-way controls (repeated class name, generic, attributes, alternative expressions, shuffled attributes). It uses the original `ATTRIBUTES` and `ALTERNATIVES` dictionaries, which are intentionally **different** from the later Experiment 10 object-expression lists.
- `scripts/run_controls.py --protocol linguistic` reproduces the later three-way control (repeated class name, class-name-preserving templates, alternative category phrases), the per-class prototype cosine similarities, and paired image bootstrap (5,000 resamples).
- `scripts/sensitivity_analysis.py` excludes only the **eight test images flagged by the original embedding-neighbor screen**. It is conservative screening, not confirmed duplicate removal. No new image labels or image-level annotations are invented.

## Historical reproduction

These instructions reproduce the historical linguistic controls, fixed-family checkpoint comparisons, and CLIP few-shot experiments. They use the historical seed-42 partitions in `reference/splits/`. The current total-budget study covers all three checkpoints and uses a rebuilt TrashNet partition; see [BUDGET_PROTOCOL.md](PROTOCOL.md#budget-study).

Install dependencies using the root README before running the commands below.

#### 1. Get the datasets

Download the original datasets from their official sources, and extract them **locally** or in Google Drive. Respect the individual dataset licenses and citation requirements.

- **RealWaste**: [UCI Machine Learning Repository, dataset 908](https://archive.ics.uci.edu/dataset/908/realwaste) (9 categories, 4,752 images).
- **TrashNet**: [garythung/trashnet](https://github.com/garythung/trashnet) (6 categories, 2,527 images). The images may be packaged separately from the source repository.

Your extracted directories must contain the original category folders, for example:

```text
data/
  RealWaste/
    Cardboard/  Food Organics/  Glass/  Metal/  Miscellaneous Trash/
    Paper/  Plastic/  Textile Trash/  Vegetation/
  dataset-resized/
    cardboard/ glass/ metal/ paper/ plastic/ trash/
```

Nested archive wrappers are fine; pass the path of the directory **containing the class folders**. `discover()` recursively searches beneath it. **Do not rename, remove, deduplicate or rearrange the images** if your goal is to reproduce the published test partitions. The scripts validate reference split membership and fail if it differs.

#### 3. Reproduce the zero-shot cross-backbone study

**OpenAI CLIP** (Experiment 10):

```bash
python scripts/run_zero_shot.py --dataset realwaste --data data/RealWaste \
  --backbone clip --reference-split reference/splits/realwaste.csv \
  --out results/zero_shot/realwaste/clip
```

**SigLIP** (Experiment 11):

```bash
python scripts/run_zero_shot.py --dataset realwaste --data data/RealWaste \
  --backbone siglip --reference-split reference/splits/realwaste.csv \
  --out results/zero_shot/realwaste/siglip
```

**OpenCLIP** (Experiment 12):

```bash
python scripts/run_zero_shot.py --dataset realwaste --data data/RealWaste \
  --backbone openclip --reference-split reference/splits/realwaste.csv \
  --out results/zero_shot/realwaste/openclip
```

Repeat with `--dataset trashnet --data data/dataset-resized --reference-split reference/splits/trashnet.csv` and a corresponding output path. Or run the entire protocol with:

```bash
bash scripts/run_all.sh data/RealWaste data/dataset-resized
```

Use `--cache-dir results/cache/<dataset>` in each run (as `run_all.sh` does) to reuse checkpoint-checked CLIP features across the zero-shot, few-shot, and linguistic-control experiments.

Each zero-shot run writes `family_summary.csv`, `metrics_by_prompt_set.csv`, `predictions.csv`, `all_prompts.csv`, `split_manifest.csv`, `metadata.json` and cached image features. **All nine prompt sets are evaluated and reported**. The test labels are never used to choose prompts.

#### 4. Reproduce the original linguistic controls

The manuscript also reports the original five-condition Experiment 8 and a three-condition follow-up. These use **different** prompt definitions from the later nine-set backbone study:

```bash
python scripts/run_controls.py --protocol experiment8 --dataset realwaste --data data/RealWaste --reference-split reference/splits/realwaste.csv --out results/controls/realwaste/experiment8 --bootstrap 2000
python scripts/run_controls.py --protocol linguistic --dataset realwaste --data data/RealWaste --reference-split reference/splits/realwaste.csv --out results/controls/realwaste/linguistic --bootstrap 5000
```

Repeat both commands with `--dataset trashnet --data data/dataset-resized` and output paths under `results/controls/trashnet/`. The reference archive includes the original metrics and the exact follow-up prompts. For the conservative image-overlap sensitivity analysis, run:

```bash
python scripts/sensitivity_analysis.py --predictions results/controls/realwaste/linguistic/predictions.csv --dataset realwaste --out paper_outputs/sensitivity_realwaste.csv
python scripts/sensitivity_analysis.py --predictions results/controls/trashnet/linguistic/predictions.csv --dataset trashnet --out paper_outputs/sensitivity_trashnet.csv
```

These remove **flagged candidates**, not confirmed duplicates. The audit does not establish absence of other overlapping images.

#### 5. Reproduce the 30-draw few-shot comparison

```bash
python scripts/run_few_shot.py --dataset realwaste --data data/RealWaste \
  --reference-split reference/splits/realwaste.csv \
  --shots 1 2 4 8 16 --repeats 30 --seed 42 \
  --out results/few_shot/realwaste

python scripts/run_few_shot.py --dataset trashnet --data data/dataset-resized \
  --reference-split reference/splits/trashnet.csv \
  --shots 1 2 4 8 16 --repeats 30 --seed 42 \
  --out results/few_shot/trashnet
```

This matches the original Experiment 9 implementation: the **same** support set and baseline-tuned alpha/beta are used for both adapters; the only change is the text prototype. Each run saves `support_set_ids.csv`, `matched_hyperparameters.csv`, per-image `predictions.csv`, `metrics_by_run.csv`, `summary.csv`, and `paired_draw_intervals.csv`.

**Do not treat the 30 draws as 30 independent test datasets.** The paired confidence intervals quantify support-set sampling variability on one fixed test partition and are unadjusted for multiplicity.

#### 6. Regenerate paper tables and figures

```bash
python scripts/make_paper_tables.py --results results --out paper_outputs
python scripts/verify_reference.py --results results --reference reference
```

The table generator writes:

- `table_cross_backbone.csv` (all models, datasets, linguistic families; accuracy and macro-F1)
- `table_fewshot_paired.csv` (paired gains and unadjusted intervals)
- `figure_prompt_families_realwaste.png`, `figure_prompt_families_trashnet.png`
- `figure_fewshot_paired.png`

`verify_reference.py` compares your fresh output against the archived summaries in `reference/`, using a configurable numerical tolerance. The default permits minor library/hardware floating-point variation; it does **not** guarantee bit-for-bit equivalence. For stricter checking pass `--tolerance 0.0001`.

##### Archived result checkpoints (descriptive means)

| Model | Best accuracy prompt family, RealWaste | Best accuracy prompt family, TrashNet |
|---|---|---|
| OpenAI CLIP ViT-B/16 | Object-oriented (60.25%) | Object-oriented (72.92%) |
| SigLIP Base Patch16-224 | Name-preserving (63.62%) | Name-preserving (78.99%) |
| OpenCLIP ViT-B/16 LAION-2B | Name-preserving (58.32%) | Name-preserving (76.02%) |

These are **mean accuracies over three predefined authored prompt sets**, not accuracy of a single selected best prompt. For OpenCLIP, object-oriented prompts have higher mean **macro-F1** despite lower mean accuracy. Full archived tables are included.

#### 7. Reproducibility details and limitations

The full protocol is documented in [PROTOCOL.md](PROTOCOL.md), and what has actually been tested in this release is stated in [REPRODUCTION_STATUS.md](PROTOCOL.md#validation-status). In particular:

- The exact split is sorted image paths, seed 42, stratified 60/20/20. The original split CSVs are included and checked at run time.
- Text and image features are L2-normalized. Text prompts are embedded individually, averaged per class, and normalized again.
- The historical few-shot text prototypes are **not** identical to the nine-family zero-shot prompt grid. They reproduce the original Experiment 9 design.
- Checkpoint names are fixed, but floating-point precision, library versions, upstream model revisions and device can change numerical results slightly. For archival-grade replication, record package versions and model revision hashes.
- Images and weights are excluded due to size and redistribution restrictions.
- The original near-duplicate screen was limited to flagged candidate pairs; the paper reports a zero-shot sensitivity analysis excluding eight flagged test images. This repository reproduces the primary classification and adapter experiments, not the original duplicate-screening pipeline or exploratory Experiments 1–7. The original Experiment 8 controls are included.
- Reproducing the code does not establish that material classification alone provides valid recycling, donation, or repair recommendations.


#### Secondary core analyses

The archived core outputs include shared and independently tuned CLIP adapters, validation-selected prompt transfers, and paired checkpoint contrasts. Historical validation selection uses 950 RealWaste or 505 TrashNet labels in addition to support labels; these results are separate from the current total-budget study. Intervals are exploratory and conditional on the historical split and wording.

Run from the repository root after restoring measurements:

```bash
python scripts/audit_upgrade_results.py
python scripts/analyze_selection_transfer.py
python scripts/analyze_cross_backbone.py --results measurements/seed_42 --dataset realwaste --out reanalysis/bootstrap_realwaste
python scripts/analyze_cross_backbone.py --results measurements/seed_42 --dataset trashnet --out reanalysis/bootstrap_trashnet
```

Canonical archived paired intervals and their metadata are in `measurements/seed_42/paired_analysis/`. These checks reconstruct saved results; they do not independently repeat image encoding or establish the original validation-grid optimum.

## Validation status

Completed real-data budget outputs from ACLing_budget_results.zip are archived in measurements/budget/. Independent local reanalysis reconstructs all 720 inner-CV score grids and 5,760 method runs, including 4,196,160 per-image test predictions. The raw summary tables reproduce exactly, including the Holm correction over 24 primary comparisons. All four frozen experiment-code hashes match the current inference code. See verification/budget/INDEPENDENT_AUDIT.json.

No historical validation labels enter budget selection. Pools, inner folds, chosen CV scores, fixed test inventories, selected parameters and budget counts pass the accounting audit. Pixel-hash groups in the uploaded image manifest do not cross the budget partitions. Three exact TrashNet duplicate pairs (two historically cross-partition) triggered a group split; RealWaste retained its original split. Perceptual-hash candidates still require human review. Historical and budget TrashNet results use different images, despite equal split counts.

Two positive and one negative target-selection-versus-pooling macro-F1 contrasts survive Holm correction: RealWaste CLIP at B=16 and SigLIP at B=32 favor selection; SigLIP at B=4 favors pooling. The remaining 21 do not reject zero. Non-significance does not imply equivalence. Unadjusted support-draw intervals condition on fixed test data and prompts. Independent human-authored prompts and external-domain evaluation remain incomplete.

The recorded core and budget inference occurred outside this local audit environment. Frozen-feature reconstruction does not independently verify image pixels, model encoding, pretraining contamination or weight provenance. Hugging Face revisions are recorded; OpenCLIP has no resolved revision. Earlier core validation records remain in verification/.

Canonical budget summaries are in `measurements/budget/budget_study/summary/`. Historical paired checkpoint intervals are in `measurements/seed_42/paired_analysis/`.
