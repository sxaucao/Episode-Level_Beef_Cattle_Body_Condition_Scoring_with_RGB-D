# Academic editor revision experiments

These additions address the Academic Editor's two remaining concerns for Animals manuscript animals-4512356: separating the dorsal-angle contribution from two quality statistics, and directly evaluating episode-level BCS predictions. They supplement the existing 01–08 workflow. The earlier scripts and their results are not replaced.

## What is added

| File | Purpose |
|---|---|
| `revision_core.py` | Validated common cohort, fixed-capacity feature masking, reproducible frame sampling, model and metrics. |
| `09_component_and_episode_training.py` | Matched component/aggregation training; saves best checkpoints, histories, predictions, protocol and split audits. |
| `10_episode_prediction_reliability.py` | Held-out repeat-sampling and disjoint-half prediction assessment using frozen models. |
| `11_summarize_editor_revision.py` | Fold/seed summaries, paired animal-cluster confidence intervals, calibration and prediction-stability tables. |

Keep all files beside `04_train_full_ablation.py` and `project_paths.py`. The recovered 01–08 scripts are included for completeness and retain their numerical methods. No real dataset, trained experimental model or fabricated experimental result is included.

## Environment

Use Python 3.12 and the project's environment. A proposed installation is PyTorch 2.7.1 with torchvision 0.22.1 and CUDA 12.8 wheels, plus `requirements.txt`. Official installation source: https://pytorch.org/get-started/previous-versions/ . The new scripts require PyTorch 2.7.1-compatible APIs. RTX 4090 with 24 GB VRAM, or a newer compatible NVIDIA GPU with adequate VRAM, is the intended training hardware. GPU execution has not been tested in this revision workspace.

Windows PowerShell, from the code directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_environment.py --require-cuda
```

Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python check_environment.py --require-cuda
```

In commands below, `python` means this environment's interpreter. On Windows, replace it with `.\.venv\Scripts\python.exe`; on Linux, `.venv/bin/python`.

## Inputs and preflight

Use your prepared `dataset/2`, `3`, `4`, `5`, `6` episode directories containing paired RGB images and numerical depth `.npy` arrays. The public source is https://doi.org/10.5061/dryad.tqjq2bw4s . Raw `.bag` recordings must first be extracted; the included numbered scripts do not perform that extraction.

Stages 01, 02 and 03 must have produced:

```text
02_make_longitudinal_protocol_splits/02A_animalwise_3fold.csv
03_quality_controlled_dorsal_angle/03_qc_frame_dorsal_angle.csv
03_quality_controlled_dorsal_angle/03_qc_bag_dorsal_angle.csv
03_quality_controlled_dorsal_angle/03_profile_fit_quality.csv
```

Stage 03 must process all indexed frames. Retain unique frame keys and correct animal identities. Stage 09 checks outer animal separation, episode metadata consistency, unique frame pairs, QC/index consistency, good-frame count/ratio agreement and file existence. It fails on missing or inconsistent inputs instead of substituting zeros for missing images. Existing CSVs containing another computer's paths must be regenerated locally.

```bash
python 09_component_and_episode_training.py --audit-only
```

Inspect `09_editor_revision_results/split_manifest.csv`, `split_class_counts.csv`, `eligible_episodes.csv` and `all_episodes.csv`. The original data should ordinarily yield 111 eligible episodes out of 151, but the script checks your data rather than hard-coding those counts. Resolve unexplained count differences before training. Inspect rare-class support and confirm `Cow_5`, `Cow_5_1`, etc. are assigned to the same real animal as intended.

## Default full experiment

```bash
python 09_component_and_episode_training.py
python 10_episode_prediction_reliability.py
python 11_summarize_editor_revision.py
```

Default stage 09 runs six conditions × three animal-wise folds × three training seeds = **54 fitted models**, each for 70 epochs. Stage 10 reuses saved models and does not retrain. Run the commands sequentially. Stage 09 fails if CUDA is unavailable; `--device cpu` is intended for tiny implementation checks.

| Condition | Angle input | Ratio and count inputs | Visual pooling | Frames |
|---|---|---|---|---|
| V | Masked | Masked | Attention | 12 |
| VA | Included | Masked | Attention | 12 |
| VQ | Masked | Included | Attention | 12 |
| VAQ | Included | Included | Attention | 12 |
| V_single | Masked | Masked | Single frame | 1 |
| V_mean | Masked | Masked | Uniform mean | 12 |

The four component models retain the same three-input feature encoder and fusion dimensions. Features are standardized using training data, then excluded coordinates are zeroed. V has a constant-input branch, so the comparison controls architecture capacity while removing varying geometric information. All conditions retain the same geometry-based eligibility rule; the new results are conditional on that common cohort. This is not a comparison with a geometry-free acquisition pipeline.

The V_single control is trained with one sampled good frame per episode. Its deterministic validation/test frame is the middle frame in natural frame-key order. V versus V_single assesses visual aggregation without full-episode geometric features; V versus V_mean evaluates attention versus uniform pooling. The default main model VAQ retains auxiliary losses. The optional auxiliary-loss control adds nine fitted models:

```bash
python 09_component_and_episode_training.py --variants VAQ_noaux --resume
python 11_summarize_editor_revision.py
```

Use the same seeds as the other conditions. The optional auxiliary comparison is generated automatically if both models are available.

## Run controls and reproducibility

Stage 09 defaults: seeds 2026/2027/2028; fixed validation split seed 2026; epochs 70; image size 160; batch size 8; workers 4; AdamW learning rate 0.0002; weight decay 0.0001; crop ratio 0.70; minimum good frames 3; validation fraction 0.22; auxiliary weights 0.45/0.45; CUDA AMP; gradient clipping 5; cosine schedule.

The main five-class and management Macro-F1 definitions use fixed label sets. This corrects a comparability issue when a fold lacks classes, but means new validation scores need not match the old script exactly. Undefined evaluation QWK is missing; its contribution is set to zero only for checkpoint selection. Existing experiment values must not be relabeled as the new matched experiment.

Frame sampling is deterministic per episode and epoch, with the same weighted sampler generator and frame draws across paired variants. A repeated appearance of the same episode within one epoch uses the same sampled bag. Training seeds change initialization/sampling without changing validation animals. cuDNN benchmarking is disabled and deterministic cuDNN behavior is requested, but exact equality across hardware/software is not guaranteed.

For Windows data-loader issues, set `--num-workers 0` consistently from the audit onward. Each output directory has a fixed `protocol.json`; changing training settings, code hashes or input CSV hashes requires a new output directory. To resume exactly the same protocol:

```bash
python 09_component_and_episode_training.py --resume
```

Completed folders are skipped only after checking output hashes. An interrupted incomplete fold is retrained from epoch 1; optimizer-level resumption is not implemented. GPU/epoch runtime is not benchmarked. Existing stage 04 checkpoints are not interchangeable with the new masked-capacity models.

A staged run with the same settings is also supported:

```bash
python 09_component_and_episode_training.py --variants V VA VQ VAQ --resume
python 09_component_and_episode_training.py --variants V_single V_mean --resume
```

A one-seed pilot can detect input/runtime issues before the full run, but should use a **separate** `--output` directory. Do not report it as a three-seed analysis. All stages offer `--help`. Stage 10 and 11 accept `--results PATH` for a nondefault stage 09 directory.

## Prediction-level analyses

Stage 10 defaults to VAQ, V, V_single and V_mean, with 50 repeated draws for each bag size 1/3/6/12. The same frame draws are used for models and seeds. Good-frame embeddings are cached in evaluation mode. Repeated predictions are not an ensemble used to improve the reported standard prediction: they diagnose sensitivity. The default reference is each fitted model's deterministic episode prediction.

1. **Visual sampling sensitivity:** the complete episode's geometric vector remains fixed. Report reference-prediction agreement, modal consistency, mean absolute class change, expected-BCS standard deviation, probability total variation, repeat accuracy and the number of unique frames. Drawing 12 slots from 3 good frames does not create 12 independent observations. A one-frame VAQ test still contains full-episode geometry and must not be called a true one-frame model.
2. **Disjoint halves:** split all indexed frames, including QC failures, by natural frame-key order. Recompute the angle from pooled valid cross-sectional angles and the ratio/count within each half. Preserve original frame-level QC decisions; apply the same minimum of three good frames independently. Ineligible halves abstain. Compare halves only when both qualify, and disclose the reduced denominator. Full-episode accuracy on this same paired subset is also exported. Verify frame keys reflect acquisition order before describing halves as chronological.
3. **Accuracy and probability outputs:** the standard OOF predictions provide reference-label performance, per-class support/recall, multiclass Brier score, NLL and ten-bin ECE. ECE is descriptive in this small sample; these calculations do not calibrate the model.

The three `.csv` QC/index inputs are hashed at training time; the profile CSV is separately hashed by stage 10 and its good-frame counts and episode medians are cross-checked. Image bytes are not individually hashed. Preserve the actual prepared dataset and curation/extraction records for stronger provenance.

## Tables and interpretation

All new outputs are under `09_editor_revision_results/`, separate from the original results.

| Output in `11_revision_tables/` | Manuscript use |
|---|---|
| `component_fold_mean_sd_by_seed.csv` | Three-fold mean and SD for each seed; supplement to Table S2. |
| `component_pooled_seed_mean_sd.csv` | Pooled OOF metrics summarized across seeds; Table S2. |
| `paired_animal_cluster_contrasts.csv` | Table S3; primary Acc5 contrast VAQ minus VQ, secondary contrasts and metric-scale interaction. |
| `absolute_animal_cluster_intervals.csv` | Conditional uncertainty of each fitted model's pooled OOF performance. |
| `episode_stability_summary.csv` | Table S4 repeat-sampling summaries and animal-cluster intervals. |
| `disjoint_half_summary.csv` | Table S4 eligibility, paired agreement and matched full/half accuracy. |
| `calibration_by_seed.csv` and `calibration_bins.csv` | Probability-output diagnostics. |
| `per_class_support_recall.csv` | Class counts and recall; especially important for rare BCS 2 episodes. |
| `analysis_record.json` | Provenance, sample sizes, bootstrap settings and analysis status. |

The bootstrap resamples animals within test folds, keeping all their episodes together. The same sampled indices are applied to every model and seed. It summarizes the mean of **separate seed-level metrics**, not predictions from a seed ensemble. The 95% percentile intervals condition on the fitted models and fixed folds; they do not measure uncertainty from new training cohorts or external farms. No formal multiple-comparison-adjusted significance claim is produced. Positive MAE differences indicate deterioration.

Do not select a favorable test seed or discard unfavorable metrics. If VAQ minus VQ is negligible, negative or imprecise, qualify the claimed independent contribution of the angle. High stability with low accuracy is consistent error, not reliable BCS assessment. Half-episode agreement is conditional on both halves passing; it is not longitudinal same-animal agreement across visits with potentially different true BCS labels.

## Validation performed for this delivery

A small synthetic fixture (24 episodes, 12 animals, 192 paired images) exercised all six default models across three folds for one epoch, stage 10 including both accepted and abstained half episodes, and stage 11. Separate assertions checked masked-feature invariance, equal parameter counts for the four factorial models, cached/direct inference equivalence, fixed-class metrics and whole-animal bootstrap grouping. A multiworker runtime check was blocked by workspace IPC permissions; module-level collate serialization was checked separately. Windows and multiworker runtime support still require deployment-machine verification. These are implementation tests only. No actual cattle dataset was available, and no scientific result was estimated from synthetic data for the manuscript.

The manuscript drafts contain marked result placeholders. Run the real experiments, inspect the exported tables and fill those placeholders before submission. Do not submit a draft containing `RESULTS REQUIRED`, `Pending` or `AUTHOR COMPLETION NOTE`.
