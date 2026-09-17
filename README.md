# Quality-Gated Episode-Level RGB-D Body Condition Scoring in Beef Cattle

Code and execution guide for the manuscript **Quality-Gated Episode-Level RGB-D Body Condition Scoring in Beef Cattle**.

Repository: [Episode-Level Beef Cattle Body Condition Scoring with RGB-D](https://github.com/sxaucao/Episode-Level_Beef_Cattle_Body_Condition_Scoring_with_RGB-D).

This package documents the supplied numbered analysis scripts and makes their paths independent of the terminal's working directory. Script names below are exact and case-sensitive. Remove upload suffixes such as `(1)` and `(3)`; use the supplied normalized copies together with `project_paths.py`.

## 1. Hardware and software

- Target platforms: 64-bit Windows 10/11 and Linux (for example, Ubuntu 22.04/24.04).
- Python: **3.12** in a dedicated virtual environment.
- Training hardware: **NVIDIA RTX 4090 (24 GB VRAM), or a newer suitable NVIDIA GPU with at least 24 GB VRAM**. This is the intended deployment configuration, not a benchmark-verified minimum. Future GPU architectures may need newer PyTorch wheels.
- Install an NVIDIA driver compatible with the selected CUDA wheel and confirm that `nvidia-smi` detects the GPU. CPU execution is possible for preprocessing; full training should use CUDA.
- Suggested system memory: 32 GB or more. Reserve space for the original dataset, extracted RGB/depth pairs, environments and outputs.

The installation below uses PyTorch **2.7.1**, torchvision **0.22.1** and CUDA **12.8** wheels. This pair is listed for Windows and Linux in the [official PyTorch installation archive](https://pytorch.org/get-started/previous-versions/). PyTorch 2.7 introduced Blackwell support with CUDA 12.8; see the [release announcement](https://pytorch.org/blog/pytorch-2-7/). Check a real CUDA operation with the supplied environment checker before training on a newer GPU.

These dependency pins define a proposed deployment environment. They are not a recovered package lockfile from the original manuscript experiments. Windows deployment and full GPU training have not been revalidated as part of this packaging revision.

## 2. Data source and required organization

Download the public dataset: Winkler and Boucheron, **Labeled RGB and depth images for cattle body condition score prediction**, [Dryad, DOI: 10.5061/dryad.tqjq2bw4s](https://doi.org/10.5061/dryad.tqjq2bw4s). Follow the dataset's terms and cite it when using the data.

The Dryad record provides raw `.bag` recordings and a separate DGE image archive. **These scripts expect already extracted, paired RGB images and numerical depth `.npy` arrays. They do not read raw `.bag` recordings, and DGE composite images are not a replacement for the two input modalities.** Raw-recording extraction and the manuscript's manual frame inclusion decisions are prerequisites and are not implemented by the supplied 01–08 scripts.

Place the prepared dataset in `dataset` beside the scripts. Its five class folders must be named `2`, `3`, `4`, `5`, and `6`. Within each class folder, keep a separate directory for each recording episode. Example file paths:

```text
dataset/2/Cow_5/rgb/rgb_0001.png
dataset/2/Cow_5/depth/depth_0001.npy
dataset/2/Cow_5/rgb/rgb_0002.png
dataset/2/Cow_5/depth/depth_0002.npy
dataset/3/Cow_5_1/rgb/rgb_0001.png
dataset/3/Cow_5_1/depth/depth_0001.npy
dataset/4/Cow_12/rgb/rgb_0001.png
dataset/4/Cow_12/depth/depth_0001.npy
dataset/5/Cow_18/rgb/rgb_0001.png
dataset/5/Cow_18/depth/depth_0001.npy
dataset/6/Cow_21/rgb/rgb_0001.png
dataset/6/Cow_21/depth/depth_0001.npy
```

The examples illustrate naming only; use the true animal IDs, episode identities and BCS labels. Preserve one consistent animal ID across all recordings and classes. Under the default identity parser, `Cow_5` and `Cow_5_1` both belong to animal `Cow_5`. Do not place frames directly inside a class folder: the indexer would combine them into a single `root` episode.

RGB files may use `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif` or `.tiff`. Depth files must be numerical `.npy` arrays with the original depth information, not colorized depth previews. Within each episode, the indexer pairs filenames using a normalized key, usually their **last numeric group**. Use a unique frame number for every pair. Duplicate normalized keys can silently select the wrong depth file; remove duplicates before indexing. Do not mix preview images with RGB frames. Unpaired frames are omitted, so inspect the indexer output and inventory.

The manuscript describes 3815 retained RGB–depth pairs, 151 episodes and 52 animals after curation. These are reference counts, not values enforced by the scripts. Matching them requires the same recordings, extraction and frame inclusion decisions. The exact original inclusion manifest is not bundled here.

By default, no path editing is necessary. An optional external dataset directory can be set before running:

Windows PowerShell:

```powershell
$env:CATTLE_DATASET_DIR = "D:\cattle_data\dataset"
```

Linux:

```bash
export CATTLE_DATASET_DIR="/data/cattle/dataset"
```

Relative overrides are resolved against the repository root. Generated outputs always stay under the repository root. Index CSVs contain local image paths: **rerun stages 01 and 02, then downstream stages, after moving the project or dataset to another machine**.

## 3. Install on Windows

Open PowerShell in the repository directory. Python 3.12 must already be installed.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
nvidia-smi
.\.venv\Scripts\python.exe check_environment.py --require-cuda
```

Using the virtual environment's Python directly avoids PowerShell activation-policy issues. Do not use a different Python executable to launch training.

## 4. Install on Linux

Open a terminal in the repository directory. Install Python 3.12 and its `venv` support using your distribution's supported method first.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
nvidia-smi
.venv/bin/python check_environment.py --require-cuda
```

The checker imports all dependencies, records versions and GPU information in `environment_report.json`, and performs a small CUDA matrix multiplication. A nonzero exit status means the environment needs attention. It does not validate model accuracy or dataset content.

## 5. Script inventory and execution order

| Stage | Exact script filename | Role / principal output |
|---|---|---|
| 01 | `01_build_new_rgb_depth_index_and_folds.py` | Pair RGB/depth files; create frame, episode and animal inventories and a preliminary five-fold split under `01_build_new_rgb_depth_index_and_folds/`. |
| 02 | `02_make_longitudinal_protocol_splits.py` | Recover animal identities; generate animal-wise and episode-wise three-/five-fold split CSVs under `02_make_longitudinal_protocol_splits/`. |
| 03 | `03_quality_controlled_dorsal_angle.py` | Frame-level dorsal-angle QC, episode aggregation and fit diagnostics under `03_quality_controlled_dorsal_angle/`. |
| 03B | `03B_pretraining_supplementary_analyses.py` | QC exclusions, threshold coverage, within-episode consistency and group composition. Despite its filename, this does not pretrain a neural network. |
| 03C | `03C_fit_quality_analysis.py` | Fit-quality distribution and additional filtering sensitivity analyses. |
| 04 | `04_train_full_ablation.py` | Main seven-variant, three-fold ablation training; results under `04_full_ablation_results/`. |
| 04B | `04B_threshold_performance.py` | Post hoc threshold filtering of existing final-model predictions. |
| 05 | `05_train_rgb_depth_comil_lowhigh_animalwise3fold.py` | Separate pretrained ResNet18-based RGB–depth experiment under `05_rgb_depth_comil_lowhigh_animalwise3fold/`. |
| 06 | `06_make_tables_and_figures.py` | Generate initial data/statistical/model tables and figures under `06_tables_and_figures_outputs/`. |
| 07 | `07_update_tables_with_ablation.py` | Update final model, ablation, extreme-error and contextual tables using stage 04; write `07_final_tables_figures/`. |
| 08 | `08_collect_animals_manuscript_artifacts.py` | Collect available manuscript assets into `08_manuscript_package/` and record missing/manual items. |

Downstream model analyses use **`02_make_longitudinal_protocol_splits/02A_animalwise_3fold.csv`**, not stage 01's preliminary five-fold output or the episode-wise diagnostic split.

Preview all commands without running:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --dry-run
```

```bash
.venv/bin/python run_pipeline.py --dry-run
```

Run the complete supplied sequence, including stage 05:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

```bash
.venv/bin/python run_pipeline.py
```

The runner uses the same interpreter for all stages, executes them in the order shown above, and stops on a failed stage. It records commands, Python/platform information, source hashes, package versions and completed stages under `run_logs/`. It does not automatically resume or infer missing dependencies.

For staged execution, use the following commands after activating the environment, or replace `python` with the explicit executable shown above:

```bash
python run_pipeline.py --stages 01 02
python run_pipeline.py --stages 03 03B 03C
python run_pipeline.py --stages 04 04B 05
python run_pipeline.py --stages 06 07 08
```

Inspect stage 02's identity and fold audits before starting training. Selecting later stages assumes their input files already exist. A complete stage 04 run trains seven variants across three test folds; stage 05 adds a separate three-fold experiment. Runtime depends on hardware and data and has not been benchmarked here.

## 6. Main training configuration

Stage 04 is the source for the final ablation comparisons. Its supplied defaults are retained:

| Setting | Default |
|---|---|
| Outer evaluation | Animal-wise three-fold cross-validation |
| Split construction seed, stages 01/02 | 42 |
| Stage 04 training seed | 2026, with fold-specific offsets |
| Validation fraction | 0.22 of training-side animals |
| Epochs | 70 |
| RGB/depth input size | 160 × 160 |
| Sampled frames per episode | 12 |
| Batch size / data loader workers | 8 / 4 |
| Optimizer | AdamW |
| Learning rate / weight decay | 0.0002 / 0.0001 |
| Scheduler | CosineAnnealingLR over the configured epochs |
| Gradient clipping | Maximum norm 5 |
| AMP | Enabled on CUDA unless `--no-amp` is supplied |
| Depth center-crop ratio | 0.70 |
| Quality gate | At least 3 good-angle frames and a valid episode angle |
| Auxiliary Low/High loss weights | 0.45 / 0.45 where enabled |

The five output classes represent BCS 2–6. The management mapping is Lean = BCS 2–3, Ideal = BCS 4–5, High = BCS 6.

The seven stage 04 variant names are:

```text
rgb_only_lowhigh
depth_only_lowhigh
rgb_depth_comil
rgb_depth_lowhigh
qg_rgb_depth_lowhigh_no_angle
qg_rgb_depth_lowhigh_with_angle
angle_only_mlp
```

Example: train only the final model with explicit defaults:

```bash
python 04_train_full_ablation.py --variants qg_rgb_depth_lowhigh_with_angle --epochs 70 --img-size 160 --frames-per-bag 12 --batch-size 8 --num-workers 4 --lr 0.0002 --weight-decay 0.0001 --depth-crop-ratio 0.70 --min-good-frames 3 --val-fraction 0.22 --seed 2026
```

Run all variants for complete stage 07 tables. A final-model-only run cannot supply the missing ablation rows. Use `python 04_train_full_ablation.py --help` for supported options. If Windows worker startup or memory use is problematic, try `--num-workers 0`. Changing batch size, seeds or other training settings produces a new experiment and should be reported.

`--skip-existing` reuses files without verifying their settings. Use it only when the data, code and configuration match. Output directories have fixed names and reruns can replace previous results; archive them or use a separate project copy for each experiment.

Stage 05 has different defaults: pretrained ResNet18, 224 × 224 input, batch size 12, learning rate 0.00003, 70 epochs, seed 42 and validation fraction 0.20. Its first execution may download pretrained torchvision weights. Its parameters are constants near the top of its script. **Stage 05 is not interchangeable with stage 04's `rgb_depth_lowhigh` ablation.** Final comparative tables are updated by stage 07 from stage 04 outputs.

## 7. Outputs and reproducibility records

For each stage 04 variant, inspect `fold_metrics.csv`, `mean_std.csv`, `predictions.csv`, aggregate confusion matrices and classification reports. The combined results are `04_ablation_summary_table.csv` and `04_ablation_extreme_error_table.csv`. Stage 04 selects the best validation state in memory but does not export `.pth` checkpoints; stage 05 does export checkpoints.

Stage 07 provides the updated Tables 3, 4, 5 and 7 and performance/extreme-error figures. Stage 08 uses these updated assets together with stage 06 data/statistical assets and stage 04 confusion matrices. Inspect `08_manuscript_package/00_manifest_copied_files.csv` and `00_missing_files.csv`. A completed collector does not mean all manuscript assets exist: workflow/architecture diagrams and example panels may require manual preparation. The separate 03B/03C/04B directories should also be retained when sharing supplementary analyses; stage 08 is not an exhaustive archive of all outputs.

Keep the generated split/identity audits, frame/QC inventories, prediction CSVs, environment report and run logs with the experiment. For direct script execution, additionally record the exact command and `python -m pip freeze` output. Preserve the data extraction and inclusion manifest where available.

Random seeds do not guarantee bitwise equality across operating systems, GPUs or library versions. The supplied training code allows CUDA/cuDNN nondeterminism. Do not label newly generated scores as the manuscript's original scores without checking their provenance.

## 8. Troubleshooting

| Symptom | Action |
|---|---|
| `ModuleNotFoundError: project_paths` | Keep `project_paths.py` beside all numbered scripts. Copy the complete package. |
| CUDA unavailable or no compatible kernel | Check `nvidia-smi`, the active interpreter, CUDA-enabled PyTorch installation and GPU architecture support; rerun the environment checker. |
| `torchvision` import/operator error | Reinstall the matching torch/torchvision pair in the same clean environment. |
| No RGB/depth pairs or unexpectedly few frames | Check `dataset/2–6/<episode>/`, supported extensions, unique normalized frame keys and the indexer inventory. |
| Missing index/QC/results CSV | Run its preceding stages; verify exact filenames and case on Linux. |
| Paths point to another computer | Rebuild stage 01/02 indices and downstream outputs locally. |
| GPU or worker memory exhausted | Close competing GPU jobs; use stage 04 `--num-workers 0`; document any batch-size change. |
| Stage 07 fails after a partial variant run | Run all stage 04 variants before generating full comparison tables. |
| Stage 08 reports missing assets | Consult the missing-file CSV; supply manual figures and rerun the appropriate generation/collection stage. |

## 9. Package revision record

The numbered scientific scripts retain their original computational settings. Changes comprise normalized filenames, repository-root-based path definitions, the optional dataset environment variable, an outdated stage-reference message correction, and new installation/execution documentation and helpers. `source_manifest.json` records the input and delivered script hashes. No dataset, trained model, experimental result or software licence grant is added by this package.
