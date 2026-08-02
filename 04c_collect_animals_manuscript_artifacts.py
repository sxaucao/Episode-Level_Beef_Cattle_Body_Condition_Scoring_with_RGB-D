from pathlib import Path
import shutil
import csv
import textwrap
from datetime import datetime


BASE_PATH = Path(r"../dataset")

SRC_11 = BASE_PATH / "11_animals_paper_outputs"
SRC_12 = BASE_PATH / "12_full_ablation_results"
SRC_13 = BASE_PATH / "13_animals_final_tables_figures"

FINAL_VARIANT_DIR = SRC_12 / "qg_rgb_depth_lowhigh_with_angle"

OUT_DIR = BASE_PATH / "14_animals_manuscript_package"
FIG_MAIN_DIR = OUT_DIR / "01_figures_main"
TAB_MAIN_DIR = OUT_DIR / "02_tables_main"
SUPP_DIR = OUT_DIR / "03_supplementary"
MISS_DIR = OUT_DIR / "04_missing_or_manual_items"

for d in [OUT_DIR, FIG_MAIN_DIR, TAB_MAIN_DIR, SUPP_DIR, MISS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

manifest_rows = []
missing_rows = []


def copy_file(src, dst_dir, dst_name=None, category="", note="", required=True):

    src = Path(src)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if dst_name is None:
        dst_name = src.name

    dst = dst_dir / dst_name

    if src.exists():
        shutil.copy2(src, dst)
        manifest_rows.append({
            "category": category,
            "status": "copied",
            "source": str(src),
            "destination": str(dst),
            "note": note,
        })
        print(f"[Copied] {dst}")
        return dst

    status = "missing_required" if required else "missing_optional"
    missing_rows.append({
        "category": category,
        "status": status,
        "source": str(src),
        "destination": str(dst),
        "note": note,
    })
    print(f"[Missing] {src}")
    return None


def copy_both_ext(src_dir, stem, dst_dir, out_stem, category="", note="", required=True):

    copied_any = False
    for ext in [".png", ".pdf"]:
        p = Path(src_dir) / f"{stem}{ext}"
        out_name = f"{out_stem}{ext}"
        out = copy_file(p, dst_dir, out_name, category=category, note=note, required=(required and ext == ".png"))
        if out is not None:
            copied_any = True
    return copied_any


def write_csv(path, rows, fieldnames):
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def collect_main_figures():

    copy_both_ext(
        SRC_11 / "figures",
        "Figure1_overall_workflow",
        FIG_MAIN_DIR,
        "Figure01_overall_workflow",
        category="main_figure",
        note="Manual workflow diagram if available; otherwise draw separately.",
        required=False,
    )

    # Figure 2 good/bad frame examples: manual item unless user later provides it.
    copy_both_ext(
        SRC_11 / "figures",
        "Figure2_good_bad_frame_examples",
        FIG_MAIN_DIR,
        "Figure02_good_bad_frame_examples",
        category="main_figure",
        note="Good/bad frame example panel if available; otherwise assemble from QC debug images.",
        required=False,
    )

    # Figure 3 data distribution / coverage
    copy_both_ext(
        SRC_11 / "figures",
        "Figure3A_dataset_distribution_by_bcs",
        FIG_MAIN_DIR,
        "Figure03A_dataset_distribution_by_BCS",
        category="main_figure",
        note="Dataset bag/frame distribution by BCS.",
        required=True,
    )

    copy_both_ext(
        SRC_11 / "figures",
        "Figure3B_good_frame_ratio_by_bcs",
        FIG_MAIN_DIR,
        "Figure03B_good_frame_ratio_by_BCS",
        category="main_figure",
        note="Good-frame ratio by BCS after anatomical quality control.",
        required=True,
    )

    # Figure 4 model architecture: manual item unless user later provides it.
    copy_both_ext(
        SRC_11 / "figures",
        "Figure4_model_architecture",
        FIG_MAIN_DIR,
        "Figure04_model_architecture",
        category="main_figure",
        note="Model architecture diagram if available; final prediction must come from ordinal head.",
        required=False,
    )

    # Figure 5 final confusion matrices from final variant in 12 results.
    copy_file(
        FINAL_VARIANT_DIR / "aggregate_cm5.png",
        FIG_MAIN_DIR,
        "Figure05A_final_model_5class_confusion_matrix.png",
        category="main_figure",
        note="Final model = qg_rgb_depth_lowhigh_with_angle; copied from 12 ablation result.",
        required=True,
    )

    copy_file(
        FINAL_VARIANT_DIR / "aggregate_cm3.png",
        FIG_MAIN_DIR,
        "Figure05B_final_model_3class_management_confusion_matrix.png",
        category="main_figure",
        note="Final ordinal-derived Lean/Ideal/High confusion matrix from 12 ablation result.",
        required=True,
    )

    # Figure 6 dorsal angle statistical analysis
    copy_both_ext(
        SRC_11 / "figures",
        "Figure6A_QC_dorsal_angle_by_BCS",
        FIG_MAIN_DIR,
        "Figure06A_QC_dorsal_angle_by_BCS",
        category="main_figure",
        note="QC dorsal angle distribution by BCS; Spearman association.",
        required=True,
    )

    copy_both_ext(
        SRC_11 / "figures",
        "Figure6B_QC_dorsal_angle_by_management",
        FIG_MAIN_DIR,
        "Figure06B_QC_dorsal_angle_by_management_class",
        category="main_figure",
        note="QC dorsal angle distribution by Lean/Ideal/High with significance annotations.",
        required=True,
    )

    # Figure 7 and 8 from 13 final tables/figures.
    copy_both_ext(
        SRC_13 / "figures",
        "Figure7_final_model_performance",
        FIG_MAIN_DIR,
        "Figure07_final_model_performance_comparison",
        category="main_figure",
        note="Updated using 12_full_ablation_results.",
        required=True,
    )

    copy_both_ext(
        SRC_13 / "figures",
        "Figure8_final_extreme_error_comparison",
        FIG_MAIN_DIR,
        "Figure08_final_extreme_error_comparison",
        category="main_figure",
        note="Updated using 12_full_ablation_results.",
        required=True,
    )



def collect_main_tables():
    """
    Main manuscript tables.
    """
    # Tables from 11: dataset, fold, angle stats.
    table11 = SRC_11 / "tables"
    table13 = SRC_13 / "tables"

    # Table 1 dataset summary
    for ext in [".csv", ".xlsx"]:
        copy_file(
            table11 / f"Table1_dataset_summary{ext}",
            TAB_MAIN_DIR,
            f"Table01_dataset_summary{ext}",
            category="main_table",
            note="Dataset summary.",
            required=(ext == ".csv"),
        )

    # Table 2 fold distribution and QC coverage
    for ext in [".csv", ".xlsx"]:
        copy_file(
            table11 / f"Table2_fold_distribution_and_qc_coverage{ext}",
            TAB_MAIN_DIR,
            f"Table02_fold_distribution_and_QC_coverage{ext}",
            category="main_table",
            note="Animal-wise fold distribution and QC coverage.",
            required=(ext == ".csv"),
        )

    # Table 3 final main results from 13
    for ext in [".csv", ".xlsx"]:
        copy_file(
            table13 / f"Table3_final_main_results{ext}",
            TAB_MAIN_DIR,
            f"Table03_final_main_results{ext}",
            category="main_table",
            note="Final main results using 12 ablation source of truth.",
            required=(ext == ".csv"),
        )

    # Table 4 full ablation
    for ext in [".csv", ".xlsx"]:
        copy_file(
            table13 / f"Table4_full_ablation_study{ext}",
            TAB_MAIN_DIR,
            f"Table04_full_ablation_study{ext}",
            category="main_table",
            note="Complete ablation; final model qg_rgb_depth_lowhigh_with_angle.",
            required=(ext == ".csv"),
        )

    # Table 5 extreme error comparison
    for ext in [".csv", ".xlsx"]:
        copy_file(
            table13 / f"Table5_final_extreme_error_comparison{ext}",
            TAB_MAIN_DIR,
            f"Table05_final_extreme_error_comparison{ext}",
            category="main_table",
            note="Extreme error comparison updated with 12 results.",
            required=(ext == ".csv"),
        )

    # Table 6 dorsal angle summary/statistics
    for ext in [".csv", ".xlsx"]:
        copy_file(
            table11 / f"Table6_QC_dorsal_angle_summary_by_BCS{ext}",
            TAB_MAIN_DIR,
            f"Table06A_QC_dorsal_angle_summary_by_BCS{ext}",
            category="main_table",
            note="Dorsal angle descriptive statistics by BCS.",
            required=(ext == ".csv"),
        )

        copy_file(
            table11 / f"Table6B_QC_dorsal_angle_summary_by_management{ext}",
            TAB_MAIN_DIR,
            f"Table06B_QC_dorsal_angle_summary_by_management_class{ext}",
            category="main_table",
            note="Dorsal angle descriptive statistics by management class.",
            required=False,
        )

        copy_file(
            table11 / f"Table6C_QC_dorsal_angle_statistical_tests{ext}",
            TAB_MAIN_DIR,
            f"Table06C_QC_dorsal_angle_statistical_tests{ext}",
            category="main_table",
            note="Spearman/Kruskal-Wallis/Mann-Whitney statistics.",
            required=(ext == ".csv"),
        )

    # Table 7 contextual reference comparison
    for ext in [".csv", ".xlsx"]:
        copy_file(
            table13 / f"Table7_contextual_reference_comparison{ext}",
            TAB_MAIN_DIR,
            f"Table07_contextual_reference_comparison{ext}",
            category="main_table",
            note="Contextual reference comparison; not strict head-to-head.",
            required=(ext == ".csv"),
        )


# ============================================================
# Supplementary materials
# ============================================================

def collect_supplementary():
    """
    Copy useful supplementary files.
    """
    supp_tables_dir = SUPP_DIR / "tables"
    supp_figs_dir = SUPP_DIR / "figures"
    supp_preds_dir = SUPP_DIR / "predictions_and_reports"

    for d in [supp_tables_dir, supp_figs_dir, supp_preds_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Complete ablation raw files.
    for name in [
        "12_ablation_summary_table.csv",
        "12_ablation_extreme_error_table.csv",
        "12_ablation_main_metrics.png",
        "12_ablation_extreme_errors.png",
    ]:
        dst_dir = supp_figs_dir if name.lower().endswith((".png", ".pdf", ".jpg")) else supp_tables_dir
        copy_file(
            SRC_12 / name,
            dst_dir,
            name,
            category="supplementary",
            note="Raw 12-ablation output.",
            required=False,
        )

    # Final variant reports.
    for name in [
        "fold_metrics.csv",
        "mean_std.csv",
        "predictions.csv",
        "aggregate_confusion_report.txt",
        "aggregate_cm5.png",
        "aggregate_cm3.png",
    ]:
        dst_dir = supp_preds_dir if not name.lower().endswith((".png", ".pdf", ".jpg")) else supp_figs_dir
        copy_file(
            FINAL_VARIANT_DIR / name,
            dst_dir,
            f"final_qg_angle_{name}",
            category="supplementary",
            note="Final model detailed report/predictions from 12 ablation.",
            required=False,
        )

    # Pairwise angle tests.
    copy_file(
        SRC_11 / "tables" / "Supplementary_pairwise_BCS_angle_tests.csv",
        supp_tables_dir,
        "Supplementary_pairwise_BCS_angle_tests.csv",
        category="supplementary",
        note="Pairwise BCS-level dorsal angle tests.",
        required=False,
    )

    copy_file(
        SRC_11 / "tables" / "Supplementary_pairwise_BCS_angle_tests.xlsx",
        supp_tables_dir,
        "Supplementary_pairwise_BCS_angle_tests.xlsx",
        category="supplementary",
        note="Pairwise BCS-level dorsal angle tests.",
        required=False,
    )

    # QC good frame index and bag-level dorsal angle table.
    copy_file(
        BASE_PATH / "08e_quality_controlled_dorsal_angle" / "08e_qc_bag_dorsal_angle.csv",
        supp_tables_dir,
        "Supplementary_QC_bag_dorsal_angle.csv",
        category="supplementary",
        note="Bag-level QC dorsal angle features.",
        required=False,
    )

    copy_file(
        BASE_PATH / "08e_quality_controlled_dorsal_angle" / "08e_good_frame_index.csv",
        supp_tables_dir,
        "Supplementary_QC_good_frame_index.csv",
        category="supplementary",
        note="Good-frame index generated by anatomical quality control.",
        required=False,
    )


# ============================================================
# Manual item checklist
# ============================================================

def write_manual_checklist():
    """
    Figure 1, Figure 2, Figure 4 are usually custom diagrams/panels.
    This checklist reminds what to prepare manually if not already present.
    """
    checklist = textwrap.dedent("""
    Manual / visual items still recommended for the manuscript
    =========================================================

    Figure 1. Overall workflow diagram
    ----------------------------------
    Suggested content:
      RGB-depth video episode
      -> animal-wise episode indexing
      -> anatomical quality gate
      -> good-frame selection
      -> QC dorsal angle extraction
      -> RGB-depth MIL feature extraction
      -> angle-aware fusion
      -> ordinal BCS head
      -> Lean / Ideal / High management class

    Important labels:
      Final model = Quality-gated RGB-depth LowHigh CO-MIL with QC dorsal angle.
      Final class comes from ordinal BCS prediction.
      Low/high branches are auxiliary binary risk-learning heads.

    Figure 2. Good/bad frame quality-control examples
    -------------------------------------------------
    Suggested panel:
      Row 1: good frame, complete dorsal trunk view
      Row 2: bad frame with head/neck/horn interference
      Row 3: bad frame with incomplete trunk / oblique posture / rail interference

    Recommended columns:
      RGB | depth/closeness map | body mask / trunk rows | dorsal profile / angle

    Figure 4. Model architecture diagram
    ------------------------------------
    Suggested modules:
      RGB encoder
      Depth encoder
      frame-level fusion
      attention MIL pooling
      QC dorsal angle feature encoder
      bag-level feature fusion
      ordinal BCS head
      low-condition auxiliary branch
      high-condition auxiliary branch

    Avoid:
      Do NOT draw the auxiliary branches as the final three-class classifier.
      Do NOT include auxiliary branch-only confusion matrix in the main paper.
    """).strip()

    p = MISS_DIR / "manual_figure_checklist.txt"
    p.write_text(checklist, encoding="utf-8")
    print(f"[Checklist] {p}")


# ============================================================
# Manifest / README
# ============================================================

def write_manifest_and_readme():
    fieldnames = ["category", "status", "source", "destination", "note"]

    manifest_path = OUT_DIR / "00_manifest_copied_files.csv"
    write_csv(manifest_path, manifest_rows, fieldnames)
    print(f"[Manifest] {manifest_path}")

    missing_path = OUT_DIR / "00_missing_files.csv"
    write_csv(missing_path, missing_rows, fieldnames)
    print(f"[Missing list] {missing_path}")

    readme = f"""
Animals manuscript package
==========================

Generated at:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Base path:
{BASE_PATH}

Output package:
{OUT_DIR}

Final result rule
-----------------
The main manuscript result is based on:
  12_full_ablation_results/qg_rgb_depth_lowhigh_with_angle

Final model:
  Quality-gated RGB-depth LowHigh CO-MIL with QC dorsal angle

Do NOT use old 10_final_clean results as main manuscript results.

Folder guide
------------
01_figures_main/
  Main paper figures in manuscript order.
  Figure 1, Figure 2, and Figure 4 may still need manual design if missing.

02_tables_main/
  Main paper tables in manuscript order.

03_supplementary/
  Detailed model reports, predictions, supplementary angle tests, and raw ablation outputs.

04_missing_or_manual_items/
  Checklist for figures that usually require manual drawing/assembly.

Recommended main figures
------------------------
Figure01: Overall workflow diagram                     [manual if missing]
Figure02: Good/bad frame quality-control examples      [manual if missing]
Figure03: Dataset distribution and good-frame ratio
Figure04: Model architecture diagram                   [manual if missing]
Figure05: Final model confusion matrices
Figure06: QC dorsal angle statistics
Figure07: Model performance comparison
Figure08: Extreme error comparison

Recommended main tables
-----------------------
Table01: Dataset summary
Table02: Animal-wise fold distribution and QC coverage
Table03: Final main model results
Table04: Full ablation study
Table05: Extreme error comparison
Table06: QC dorsal angle statistics
Table07: Contextual comparison with reference LACO ViT

Check:
  00_manifest_copied_files.csv
  00_missing_files.csv
"""

    readme_path = OUT_DIR / "00_README_manifest.txt"
    readme_path.write_text(readme.strip(), encoding="utf-8")
    print(f"[README] {readme_path}")


def main():
    print("=" * 100)
    print("Collect Animals manuscript figures and tables")
    print(f"BASE_PATH: {BASE_PATH}")
    print(f"OUT_DIR:   {OUT_DIR}")
    print("=" * 100)

    collect_main_figures()
    collect_main_tables()
    collect_supplementary()
    write_manual_checklist()
    write_manifest_and_readme()

    print("\nFinished.")
    print(f"Package folder: {OUT_DIR}")
    print(f"Main figures:   {FIG_MAIN_DIR}")
    print(f"Main tables:    {TAB_MAIN_DIR}")
    print(f"Supplementary:  {SUPP_DIR}")
    print(f"Missing list:   {OUT_DIR / '00_missing_files.csv'}")


if __name__ == "__main__":
    main()
