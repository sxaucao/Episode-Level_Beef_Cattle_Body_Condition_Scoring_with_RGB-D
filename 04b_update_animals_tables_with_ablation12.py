from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_PATH = Path(r"../dataset")

ABL_DIR = BASE_PATH / "12_full_ablation_results"
SUMMARY_CSV = ABL_DIR / "12_ablation_summary_table.csv"
EXTREME_CSV = ABL_DIR / "12_ablation_extreme_error_table.csv"

OUT_DIR = BASE_PATH / "13_animals_final_tables_figures"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


LABELS = {
    "rgb_only_lowhigh": "RGB-only LowHigh CO-MIL",
    "depth_only_lowhigh": "Depth-only LowHigh CO-MIL",
    "rgb_depth_comil": "RGB-depth CO-MIL",
    "rgb_depth_lowhigh": "RGB-depth LowHigh CO-MIL",
    "qg_rgb_depth_lowhigh_no_angle": "Quality-gated RGB-depth LowHigh, no angle",
    "qg_rgb_depth_lowhigh_with_angle": "Final quality-gated angle-aware CO-MIL",
    "angle_only_mlp": "Angle-only MLP",
}

ORDER = [
    "rgb_only_lowhigh",
    "depth_only_lowhigh",
    "rgb_depth_comil",
    "rgb_depth_lowhigh",
    "qg_rgb_depth_lowhigh_no_angle",
    "qg_rgb_depth_lowhigh_with_angle",
    "angle_only_mlp",
]

MAIN_ORDER = [
    "rgb_depth_comil",
    "rgb_depth_lowhigh",
    "qg_rgb_depth_lowhigh_with_angle",
]


def read_inputs():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing: {SUMMARY_CSV}")
    if not EXTREME_CSV.exists():
        raise FileNotFoundError(f"Missing: {EXTREME_CSV}")

    summary = pd.read_csv(SUMMARY_CSV, encoding="utf-8-sig")
    extreme = pd.read_csv(EXTREME_CSV, encoding="utf-8-sig")

    if "Variant" not in summary.columns:
        raise ValueError(f"Variant column not found in {SUMMARY_CSV}")
    if "Variant" not in extreme.columns:
        raise ValueError(f"Variant column not found in {EXTREME_CSV}")

    summary["Display model"] = summary["Variant"].map(LABELS).fillna(summary["Variant"])
    extreme["Display model"] = extreme["Variant"].map(LABELS).fillna(extreme["Variant"])

    return summary, extreme


def save_table(df, filename):
    csv_path = TABLE_DIR / f"{filename}.csv"
    xlsx_path = TABLE_DIR / f"{filename}.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception:
        pass
    print(f"[Table] {csv_path}")


def save_fig(fig, filename):
    png = FIG_DIR / f"{filename}.png"
    pdf = FIG_DIR / f"{filename}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] {png}")


def select_and_order(df, variants):
    out = []
    for v in variants:
        hit = df[df["Variant"] == v]
        if len(hit) == 0:
            print(f"[Warning] variant not found: {v}")
            continue
        out.append(hit.iloc[0])
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out)


def make_table3(summary, extreme):
    main = select_and_order(summary, MAIN_ORDER)
    ext = extreme[["Variant", "Lean→High", "High→Lean", "BCS2→BCS6", "BCS6→BCS2/3"]].copy()
    main = main.merge(ext, on="Variant", how="left")

    cols = [
        "Display model",
        "Coverage",
        "Acc5",
        "Macro-F1 5",
        "Weighted-F1 5",
        "QWK5",
        "MAE",
        "Within-one Acc",
        "Macro-F1 3 final",
        "Lean recall final",
        "High recall final",
        "Lean→High",
        "High→Lean",
        "BCS2→BCS6",
        "BCS6→BCS2/3",
    ]
    cols = [c for c in cols if c in main.columns]
    table = main[cols].copy()

    rename = {
        "Display model": "Model",
        "Macro-F1 3 final": "Macro-F1 3",
        "Lean recall final": "Lean recall",
        "High recall final": "High recall",
    }
    table = table.rename(columns=rename)

    save_table(table, "Table3_final_main_results")
    return table


def make_table4(summary, extreme):
    ab = select_and_order(summary, ORDER)
    ext = extreme[["Variant", "Lean→High", "High→Lean", "BCS2→BCS6", "BCS6→BCS2/3"]].copy()
    ab = ab.merge(ext, on="Variant", how="left")

    # Add binary check marks for manuscript clarity.
    flags = {
        "rgb_only_lowhigh": dict(RGB="✓", Depth="", LowHigh="✓", QualityGate="", DorsalAngle=""),
        "depth_only_lowhigh": dict(RGB="", Depth="✓", LowHigh="✓", QualityGate="", DorsalAngle=""),
        "rgb_depth_comil": dict(RGB="✓", Depth="✓", LowHigh="", QualityGate="", DorsalAngle=""),
        "rgb_depth_lowhigh": dict(RGB="✓", Depth="✓", LowHigh="✓", QualityGate="", DorsalAngle=""),
        "qg_rgb_depth_lowhigh_no_angle": dict(RGB="✓", Depth="✓", LowHigh="✓", QualityGate="✓", DorsalAngle=""),
        "qg_rgb_depth_lowhigh_with_angle": dict(RGB="✓", Depth="✓", LowHigh="✓", QualityGate="✓", DorsalAngle="✓"),
        "angle_only_mlp": dict(RGB="", Depth="", LowHigh="", QualityGate="✓", DorsalAngle="✓"),
    }

    for c in ["RGB", "Depth", "LowHigh", "QualityGate", "DorsalAngle"]:
        ab[c] = ab["Variant"].apply(lambda v: flags.get(v, {}).get(c, ""))

    cols = [
        "Display model",
        "RGB", "Depth", "LowHigh", "QualityGate", "DorsalAngle",
        "Coverage",
        "Acc5",
        "Macro-F1 5",
        "Weighted-F1 5",
        "QWK5",
        "MAE",
        "Within-one Acc",
        "Macro-F1 3 final",
        "Lean recall final",
        "High recall final",
        "Lean→High",
        "High→Lean",
        "BCS2→BCS6",
        "BCS6→BCS2/3",
    ]
    cols = [c for c in cols if c in ab.columns]
    table = ab[cols].copy()

    table = table.rename(columns={
        "Display model": "Variant",
        "QualityGate": "Quality gate",
        "DorsalAngle": "Dorsal angle",
        "Macro-F1 3 final": "Macro-F1 3",
        "Lean recall final": "Lean recall",
        "High recall final": "High recall",
    })

    save_table(table, "Table4_full_ablation_study")
    return table


def make_table5(extreme):
    ext = select_and_order(extreme, [
        "rgb_depth_lowhigh",
        "qg_rgb_depth_lowhigh_no_angle",
        "qg_rgb_depth_lowhigh_with_angle",
    ])

    cols = [
        "Display model",
        "Coverage",
        "Lean→High",
        "High→Lean",
        "BCS2→BCS6",
        "BCS6→BCS2/3",
    ]
    cols = [c for c in cols if c in ext.columns]
    table = ext[cols].rename(columns={"Display model": "Model"}).copy()
    save_table(table, "Table5_final_extreme_error_comparison")
    return table


def make_contextual_table(summary):
    final = summary[summary["Variant"] == "qg_rgb_depth_lowhigh_with_angle"].iloc[0]
    full = summary[summary["Variant"] == "rgb_depth_lowhigh"].iloc[0]

    rows = [
        {
            "Study / Model": "Winkler et al. LACO ViT",
            "Unit": "Image",
            "Input": "DGE image",
            "Curation": "Manual image curation",
            "Split": "Leave-a-cow-out",
            "Coverage": 1.0000,
            "Acc5": 0.6400,
            "Macro-F1 5": 0.3800,
            "Weighted-F1 5": 0.6200,
            "Within-one Acc": 0.7800,
            "Notes": "Contextual reference only; not a strict head-to-head comparison.",
        },
        {
            "Study / Model": "Ours: RGB-depth LowHigh CO-MIL",
            "Unit": "Episode / bag",
            "Input": "RGB-depth video frames",
            "Curation": "Automatic bag aggregation",
            "Split": "Animal-wise",
            "Coverage": float(full["Coverage"]),
            "Acc5": float(full["Acc5"]),
            "Macro-F1 5": float(full["Macro-F1 5"]),
            "Weighted-F1 5": float(full["Weighted-F1 5"]),
            "Within-one Acc": float(full["Within-one Acc"]),
            "Notes": "Full-coverage model.",
        },
        {
            "Study / Model": "Ours: Final quality-gated angle-aware CO-MIL",
            "Unit": "Eligible episode / bag",
            "Input": "RGB-depth + QC dorsal angle",
            "Curation": "Automatic quality gate",
            "Split": "Animal-wise",
            "Coverage": float(final["Coverage"]),
            "Acc5": float(final["Acc5"]),
            "Macro-F1 5": float(final["Macro-F1 5"]),
            "Weighted-F1 5": float(final["Weighted-F1 5"]),
            "Within-one Acc": float(final["Within-one Acc"]),
            "Notes": "Deployment-oriented reliable prediction with abstention.",
        },
    ]

    table = pd.DataFrame(rows)
    save_table(table, "Table7_contextual_reference_comparison")
    return table


def plot_model_performance(summary):
    variants = [
        "rgb_depth_comil",
        "rgb_depth_lowhigh",
        "qg_rgb_depth_lowhigh_no_angle",
        "qg_rgb_depth_lowhigh_with_angle",
        "angle_only_mlp",
    ]
    df = select_and_order(summary, variants)

    metrics = ["Acc5", "Macro-F1 5", "QWK5", "Within-one Acc", "Macro-F1 3 final"]
    labels = ["Acc5", "Macro-F1 5", "QWK5", "Within-one", "Macro-F1 3"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(metrics))
    width = 0.15

    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[m] for m in metrics]
        ax.bar(x + (i - (len(df)-1)/2) * width, values, width, label=row["Display model"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model performance comparison")
    ax.legend(fontsize=8, ncol=1)
    save_fig(fig, "Figure7_final_model_performance")


def plot_extreme_errors(extreme):
    variants = [
        "rgb_depth_lowhigh",
        "qg_rgb_depth_lowhigh_no_angle",
        "qg_rgb_depth_lowhigh_with_angle",
        "angle_only_mlp",
    ]
    df = select_and_order(extreme, variants)

    metrics = ["Lean→High", "High→Lean", "BCS2→BCS6", "BCS6→BCS2/3"]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(metrics))
    width = 0.18

    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[m] for m in metrics]
        ax.bar(x + (i - (len(df)-1)/2) * width, values, width, label=row["Display model"])

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylabel("Number of extreme errors")
    ax.set_title("Extreme error comparison")
    ax.legend(fontsize=8)
    save_fig(fig, "Figure8_final_extreme_error_comparison")


def write_readme():
    p = OUT_DIR / "README.txt"
    p.write_text(
        "Final Animals manuscript tables/figures generated from 12_full_ablation_results.\n"
        "Final model is qg_rgb_depth_lowhigh_with_angle.\n"
        "Old 10_final_clean results are not used in these main result tables.\n",
        encoding="utf-8",
    )
    print(f"[README] {p}")


def main():
    print("=" * 100)
    print("Update Animals tables/figures using 12 ablation results as final source")
    print(f"Input:  {ABL_DIR}")
    print(f"Output: {OUT_DIR}")
    print("=" * 100)

    summary, extreme = read_inputs()

    make_table3(summary, extreme)
    make_table4(summary, extreme)
    make_table5(extreme)
    make_contextual_table(summary)
    plot_model_performance(summary)
    plot_extreme_errors(extreme)
    write_readme()

    print("\nFinished.")
    print(f"Tables:  {TABLE_DIR}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
