from pathlib import Path
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    from sklearn.metrics import confusion_matrix
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


BASE_PATH = Path(r"../dataset")

FRAME_CSV = BASE_PATH / "03A_animalwise_3fold.csv"

LOWHIGH_DIR = BASE_PATH / "06_rgb_depth_comil_lowhigh_animalwise3fold_depthcrop07_results"
LOWHIGH_MEAN_STD_CSV = LOWHIGH_DIR / "06_lowhigh_mean_std.csv"
LOWHIGH_PRED_CSV = LOWHIGH_DIR / "06_lowhigh_predictions.csv"

QC_DIR = BASE_PATH / "08e_quality_controlled_dorsal_angle"
QC_FRAME_CSV = QC_DIR / "08e_qc_frame_dorsal_angle.csv"
QC_BAG_CSV = QC_DIR / "08e_qc_bag_dorsal_angle.csv"
QC_BAD_REASON_CSV = QC_DIR / "08e_bad_reason_counts.csv"

FINAL_DIR = BASE_PATH / "10_final_clean_quality_gated_angle_aware_lowhigh_comil_results"
FINAL_MEAN_STD_CSV = FINAL_DIR / "10_angle_aware_mean_std.csv"
FINAL_FOLD_METRICS_CSV = FINAL_DIR / "10_angle_aware_fold_metrics.csv"
FINAL_PRED_CSV = FINAL_DIR / "10_angle_aware_predictions.csv"

OUT_DIR = BASE_PATH / "11_animals_paper_outputs"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

BCS_CLASSES = [2, 3, 4, 5, 6]
CLASS5_NAMES = ["BCS2", "BCS3", "BCS4", "BCS5", "BCS6"]
MGMT_NAMES = ["Lean", "Ideal", "High"]
LABEL5_TO_BCS = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6}


def bcs_to_mgmt_label(bcs):
    bcs = int(bcs)
    if bcs in [2, 3]:
        return 0
    if bcs in [4, 5]:
        return 1
    return 2


def read_csv(path, required=True):
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig")
    if required:
        raise FileNotFoundError(f"Missing required file: {path}")
    return None


def save_table(df, name):
    csv_path = TABLE_DIR / f"{name}.csv"
    xlsx_path = TABLE_DIR / f"{name}.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception:
        pass
    print(f"[Table] {csv_path}")
    return csv_path


def save_fig(fig, name, dpi=300):
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] {png}")
    return png


def get_metric(path, metric, value_col="mean"):
    if not path.exists():
        return np.nan
    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = list(df.columns)
    metric_col = cols[0]
    for c in cols:
        if c.lower() in ["metric", "name", "index", "unnamed: 0"]:
            metric_col = c
            break
    target_col = None
    for c in cols:
        if c.lower() == value_col.lower():
            target_col = c
            break
    if target_col is None:
        return np.nan
    hit = df[df[metric_col].astype(str) == metric]
    if len(hit) == 0:
        return np.nan
    return float(hit.iloc[0][target_col])


def plot_confusion_matrix(cm, labels_y, labels_x, title, name, normalize=False):
    arr = cm.astype(float)
    if normalize:
        row_sums = arr.sum(axis=1, keepdims=True)
        arr = np.divide(arr, row_sums, out=np.zeros_like(arr), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(arr, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Proportion" if normalize else "Count", rotation=270, labelpad=14)
    ax.set_xticks(np.arange(len(labels_x)))
    ax.set_yticks(np.arange(len(labels_y)))
    ax.set_xticklabels(labels_x, rotation=35, ha="right")
    ax.set_yticklabels(labels_y)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{arr[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, txt, ha="center", va="center")
    save_fig(fig, name)


def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return (gt - lt) / (len(x) * len(y))


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    out = np.full(n, np.nan)
    prev = 1.0
    for rank_i in range(n - 1, -1, -1):
        idx = order[rank_i]
        rank = rank_i + 1
        prev = min(prev, pvals[idx] * n / rank)
        out[idx] = min(prev, 1.0)
    return out


def sig_label(p):
    if pd.isna(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def add_bracket(ax, x1, x2, y, text, h=0.4):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], linewidth=1)
    ax.text((x1+x2)/2, y+h, text, ha="center", va="bottom", fontsize=9)


def make_dataset_tables_and_figures():
    frame = read_csv(FRAME_CSV)
    qc_frame = read_csv(QC_FRAME_CSV, required=False)
    qc_bag = read_csv(QC_BAG_CSV, required=False)

    animal_col = "animal_id" if "animal_id" in frame.columns else "split_group_id" if "split_group_id" in frame.columns else "bag_id"
    bag_df = frame.groupby("bag_id").agg(
        bcs_raw=("bcs_raw", "first"),
        fold_id=("fold_id", "first"),
        animal_id=(animal_col, "first"),
        frame_count=("bag_id", "size"),
    ).reset_index()

    by_bcs_bag = bag_df.groupby("bcs_raw")["bag_id"].nunique().reindex(BCS_CLASSES, fill_value=0)
    by_bcs_frame = frame.groupby("bcs_raw").size().reindex(BCS_CLASSES, fill_value=0)

    good_frame_count = np.nan
    good_frame_ratio = np.nan
    if qc_frame is not None and "is_good_angle_frame" in qc_frame.columns:
        good_frame_count = int(qc_frame["is_good_angle_frame"].sum())
        good_frame_ratio = float(qc_frame["is_good_angle_frame"].mean())

    eligible_bags = np.nan
    if qc_bag is not None and "bag_good_angle_median" in qc_bag.columns:
        eligible_bags = int(qc_bag["bag_good_angle_median"].notna().sum())

    rows = [
        {"Item": "Total RGB-depth paired frames", "Value": int(len(frame))},
        {"Item": "Total bags / episodes", "Value": int(bag_df["bag_id"].nunique())},
        {"Item": "Animal / split groups", "Value": int(bag_df["animal_id"].nunique())},
        {"Item": "Good angle frames after QC", "Value": good_frame_count},
        {"Item": "Good-frame ratio after QC", "Value": good_frame_ratio},
        {"Item": "Bags with valid QC dorsal angle", "Value": eligible_bags},
    ]
    for b in BCS_CLASSES:
        rows.append({"Item": f"BCS{b} bags", "Value": int(by_bcs_bag.loc[b])})
    for b in BCS_CLASSES:
        rows.append({"Item": f"BCS{b} frames", "Value": int(by_bcs_frame.loc[b])})
    save_table(pd.DataFrame(rows), "Table1_dataset_summary")

    # Fold coverage table
    if qc_bag is not None and "bag_good_angle_median" in qc_bag.columns:
        bag_df = bag_df.merge(qc_bag[["bag_id", "bag_good_angle_median", "good_frame_ratio", "good_angle_frame_count"]], on="bag_id", how="left")
        bag_df["eligible"] = bag_df["bag_good_angle_median"].notna().astype(int)
    else:
        bag_df["eligible"] = 0

    fold_rows = []
    for f in sorted(bag_df["fold_id"].unique()):
        sub = bag_df[bag_df["fold_id"] == f]
        row = {
            "Fold": int(f),
            "Raw test bags": int(len(sub)),
            "Eligible QC-angle bags": int(sub["eligible"].sum()),
            "Coverage": float(sub["eligible"].mean()),
        }
        for b in BCS_CLASSES:
            row[f"BCS{b} bags"] = int((sub["bcs_raw"] == b).sum())
        fold_rows.append(row)
    save_table(pd.DataFrame(fold_rows), "Table2_fold_distribution_and_qc_coverage")

    # Figures
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(BCS_CLASSES))
    width = 0.38
    ax.bar(x - width/2, [by_bcs_bag.loc[b] for b in BCS_CLASSES], width, label="Bags")
    ax.bar(x + width/2, [by_bcs_frame.loc[b] for b in BCS_CLASSES], width, label="Frames")
    ax.set_xticks(x)
    ax.set_xticklabels([f"BCS{b}" for b in BCS_CLASSES])
    ax.set_ylabel("Count")
    ax.set_title("Dataset distribution by BCS")
    ax.legend()
    save_fig(fig, "Figure3A_dataset_distribution_by_bcs")

    if qc_frame is not None and "is_good_angle_frame" in qc_frame.columns:
        g = qc_frame.groupby("bcs_raw")["is_good_angle_frame"].mean().reindex(BCS_CLASSES)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar([f"BCS{b}" for b in BCS_CLASSES], g.values)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Good-frame ratio")
        ax.set_title("Anatomically valid frame ratio by BCS")
        save_fig(fig, "Figure3B_good_frame_ratio_by_bcs")


def make_main_result_tables_and_figures():
    final_cov = get_metric(FINAL_MEAN_STD_CSV, "test_coverage")
    rows = [
        {
            "Model": "Baseline RGB-depth CO-MIL",
            "Coverage": 1.0000,
            "Acc5": 0.5230,
            "Macro-F1 5": 0.3426,
            "QWK5": 0.5617,
            "MAE": 0.5433,
            "Within-one Acc": 0.9536,
            "3-class Macro-F1": 0.5295,
        },
        {
            "Model": "Full-coverage RGB-depth LowHigh CO-MIL",
            "Coverage": 1.0000,
            "Acc5": 0.5559,
            "Macro-F1 5": 0.4613,
            "QWK5": 0.5069,
            "MAE": 0.6033,
            "Within-one Acc": 0.9069,
            "3-class Macro-F1": 0.7153,
        },
        {
            "Model": "Final quality-gated angle-aware CO-MIL",
            "Coverage": final_cov,
            "Acc5": get_metric(FINAL_MEAN_STD_CSV, "acc5"),
            "Macro-F1 5": get_metric(FINAL_MEAN_STD_CSV, "macro_f1_5"),
            "QWK5": get_metric(FINAL_MEAN_STD_CSV, "qwk5"),
            "MAE": get_metric(FINAL_MEAN_STD_CSV, "mae_bcs"),
            "Within-one Acc": get_metric(FINAL_MEAN_STD_CSV, "within_one_acc"),
            "3-class Macro-F1": get_metric(FINAL_MEAN_STD_CSV, "macro_f1_3_final"),
        },
    ]
    main_df = pd.DataFrame(rows)
    save_table(main_df, "Table3_main_model_results")

    err_rows = [
        {"Model": "Full-coverage RGB-depth LowHigh CO-MIL", "Lean→High": 3, "High→Lean": 2, "BCS2→BCS6": 3, "BCS6→BCS2/3": 2},
        {"Model": "Final quality-gated angle-aware CO-MIL", "Lean→High": 1, "High→Lean": 0, "BCS2→BCS6": 1, "BCS6→BCS2/3": 0},
    ]
    err_df = pd.DataFrame(err_rows)
    save_table(err_df, "Table5_extreme_error_comparison")

    metrics = ["Acc5", "Macro-F1 5", "QWK5", "Within-one Acc", "3-class Macro-F1"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(metrics))
    width = 0.25
    for i, row in main_df.iterrows():
        ax.bar(x + (i-1)*width, [row[m] for m in metrics], width, label=row["Model"])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model performance comparison")
    ax.legend(fontsize=8)
    save_fig(fig, "Figure7_model_performance_comparison")

    metrics = ["Lean→High", "High→Lean", "BCS2→BCS6", "BCS6→BCS2/3"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(metrics))
    width = 0.35
    for i, row in err_df.iterrows():
        ax.bar(x + (i-0.5)*width, [row[m] for m in metrics], width, label=row["Model"])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_ylabel("Number of extreme errors")
    ax.set_title("Extreme error comparison")
    ax.legend(fontsize=8)
    save_fig(fig, "Figure8_extreme_error_comparison")


def make_final_confusion_matrices():
    if not HAS_SKLEARN:
        print("[Skip] sklearn unavailable")
        return
    pred = read_csv(FINAL_PRED_CSV, required=False)
    if pred is None:
        print("[Skip] final predictions not found")
        return

    true_candidates = ["true_label5", "y_true5", "true5"]
    pred_candidates = ["pred_label5_final", "final_pred_label5", "pred_label5", "pred5"]
    true_col = next((c for c in true_candidates if c in pred.columns), None)
    pred_col = next((c for c in pred_candidates if c in pred.columns), None)
    if true_col is None or pred_col is None:
        print("[Skip] cannot locate final true/pred columns")
        print(pred.columns.tolist())
        return

    y_true5 = pred[true_col].astype(int).values
    y_pred5 = pred[pred_col].astype(int).values
    cm5 = confusion_matrix(y_true5, y_pred5, labels=[0,1,2,3,4])
    plot_confusion_matrix(cm5, CLASS5_NAMES, CLASS5_NAMES, "Final 5-class confusion matrix", "Figure5A_final_cm5_counts")
    plot_confusion_matrix(cm5, CLASS5_NAMES, CLASS5_NAMES, "Final 5-class row-normalized confusion matrix", "Figure5A_final_cm5_row_normalized", normalize=True)

    y_true3 = np.array([bcs_to_mgmt_label(LABEL5_TO_BCS[int(x)]) for x in y_true5])
    y_pred3 = np.array([bcs_to_mgmt_label(LABEL5_TO_BCS[int(x)]) for x in y_pred5])
    cm3 = confusion_matrix(y_true3, y_pred3, labels=[0,1,2])
    plot_confusion_matrix(cm3, MGMT_NAMES, MGMT_NAMES, "Final ordinal-derived management confusion matrix", "Figure5B_final_cm3_counts")
    plot_confusion_matrix(cm3, MGMT_NAMES, MGMT_NAMES, "Final ordinal-derived management row-normalized confusion matrix", "Figure5B_final_cm3_row_normalized", normalize=True)


def make_dorsal_angle_analysis():
    qc_bag = read_csv(QC_BAG_CSV)
    angle_col = "bag_good_angle_median"
    df = qc_bag.dropna(subset=[angle_col]).copy()
    df["mgmt"] = df["bcs_raw"].apply(bcs_to_mgmt_label)
    df["mgmt_name"] = df["mgmt"].map({0:"Lean", 1:"Ideal", 2:"High"})

    summary = df.groupby("bcs_raw")[angle_col].describe().reset_index().rename(columns={"bcs_raw":"BCS"})
    save_table(summary, "Table6_QC_dorsal_angle_summary_by_BCS")
    summary_m = df.groupby("mgmt_name")[angle_col].describe().reset_index().rename(columns={"mgmt_name":"Management class"})
    save_table(summary_m, "Table6B_QC_dorsal_angle_summary_by_management")

    stat_rows = []
    pair_rows = []
    if HAS_SCIPY:
        rho, p = stats.spearmanr(df["bcs_raw"].astype(float), df[angle_col].astype(float), nan_policy="omit")
        stat_rows.append({"Test":"Spearman correlation", "Comparison":"BCS vs QC dorsal angle", "Statistic":rho, "p_value":p, "p_adjusted":p, "Effect":"rho", "Effect_value":rho})

        groups = [g[angle_col].dropna().values for _, g in df.groupby("bcs_raw")]
        kw = stats.kruskal(*groups)
        stat_rows.append({"Test":"Kruskal-Wallis", "Comparison":"Across BCS classes", "Statistic":kw.statistic, "p_value":kw.pvalue, "p_adjusted":kw.pvalue, "Effect":"", "Effect_value":np.nan})

        # Pairwise management tests for main text.
        mgmt_order = ["Lean", "Ideal", "High"]
        tmp = []
        raw_p = []
        for i in range(len(mgmt_order)):
            for j in range(i+1, len(mgmt_order)):
                a, b = mgmt_order[i], mgmt_order[j]
                xa = df[df["mgmt_name"] == a][angle_col].dropna().values
                xb = df[df["mgmt_name"] == b][angle_col].dropna().values
                u = stats.mannwhitneyu(xa, xb, alternative="two-sided")
                d = cliffs_delta(xa, xb)
                tmp.append((a,b,len(xa),len(xb),np.median(xa),np.median(xb),u.statistic,u.pvalue,d))
                raw_p.append(u.pvalue)
        adj = bh_fdr(raw_p)
        for k, item in enumerate(tmp):
            a,b,na,nb,meda,medb,u,pval,d = item
            stat_rows.append({"Test":"Mann-Whitney U with BH-FDR", "Comparison":f"{a} vs {b}", "n_a":na, "n_b":nb, "median_a":meda, "median_b":medb, "Statistic":u, "p_value":pval, "p_adjusted":adj[k], "Effect":"Cliff's delta", "Effect_value":d})

        # Pairwise BCS tests for supplementary.
        tmp = []
        raw_p = []
        for i in range(len(BCS_CLASSES)):
            for j in range(i+1, len(BCS_CLASSES)):
                a, b = BCS_CLASSES[i], BCS_CLASSES[j]
                xa = df[df["bcs_raw"] == a][angle_col].dropna().values
                xb = df[df["bcs_raw"] == b][angle_col].dropna().values
                u = stats.mannwhitneyu(xa, xb, alternative="two-sided")
                d = cliffs_delta(xa, xb)
                tmp.append((a,b,len(xa),len(xb),np.median(xa),np.median(xb),u.statistic,u.pvalue,d))
                raw_p.append(u.pvalue)
        adj = bh_fdr(raw_p)
        for k, item in enumerate(tmp):
            a,b,na,nb,meda,medb,u,pval,d = item
            pair_rows.append({"Comparison":f"BCS{a} vs BCS{b}", "n_a":na, "n_b":nb, "median_a":meda, "median_b":medb, "MannWhitney_U":u, "p_value":pval, "p_adjusted_BH_FDR":adj[k], "Cliffs_delta":d, "Significance":sig_label(adj[k])})

    stat_df = pd.DataFrame(stat_rows)
    save_table(stat_df, "Table6C_QC_dorsal_angle_statistical_tests")
    save_table(pd.DataFrame(pair_rows), "Supplementary_pairwise_BCS_angle_tests")

    # Figure 6A by BCS.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    data = [df[df["bcs_raw"] == b][angle_col].dropna().values for b in BCS_CLASSES]
    ax.boxplot(data, labels=[f"BCS{b}" for b in BCS_CLASSES], showfliers=False)
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data, start=1):
        ax.scatter(np.full(len(vals), i) + rng.normal(0, 0.04, size=len(vals)), vals, s=18, alpha=0.65)
    ax.set_ylabel("Quality-controlled dorsal ridge angle (degree)")
    title = "QC dorsal ridge angle by BCS"
    if HAS_SCIPY:
        rho, p = stats.spearmanr(df["bcs_raw"].astype(float), df[angle_col].astype(float), nan_policy="omit")
        title += f"\nSpearman rho={rho:.3f}, p={p:.2e}"
    ax.set_title(title)
    save_fig(fig, "Figure6A_QC_dorsal_angle_by_BCS")

    # Figure 6B by management with significance brackets.
    order = ["Lean", "Ideal", "High"]
    data_m = [df[df["mgmt_name"] == m][angle_col].dropna().values for m in order]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.boxplot(data_m, labels=order, showfliers=False)
    rng = np.random.default_rng(43)
    for i, vals in enumerate(data_m, start=1):
        ax.scatter(np.full(len(vals), i) + rng.normal(0, 0.04, size=len(vals)), vals, s=18, alpha=0.65)
    ax.set_ylabel("Quality-controlled dorsal ridge angle (degree)")
    ax.set_title("QC dorsal ridge angle by management class")
    if len(stat_df) > 0:
        pairs = stat_df[stat_df["Test"].astype(str).str.contains("Mann-Whitney", na=False)]
        ymax = max(np.max(v) for v in data_m if len(v) > 0)
        ymin = min(np.min(v) for v in data_m if len(v) > 0)
        step = max(0.8, 0.08 * (ymax-ymin))
        base = ymax + step
        mapx = {("Lean", "Ideal"):(1,2), ("Lean", "High"):(1,3), ("Ideal", "High"):(2,3)}
        k = 0
        for _, r in pairs.iterrows():
            a, b = str(r["Comparison"]).split(" vs ")
            if (a,b) in mapx:
                x1, x2 = mapx[(a,b)]
                add_bracket(ax, x1, x2, base + k*step, sig_label(r["p_adjusted"]), h=0.25*step)
                k += 1
    save_fig(fig, "Figure6B_QC_dorsal_angle_by_management")


def make_contextual_comparison():
    rows = [
        {"Study / Model":"Winkler et al. LACO ViT", "Unit":"Image", "Input":"DGE image", "Curation":"Manual image curation", "Split":"LACO", "Coverage":1.0000, "Acc5":0.6400, "Macro-F1 5":0.3800, "Weighted-F1 5":0.6200, "Within-one Acc":0.7800, "Notes":"Contextual reference only"},
        {"Study / Model":"Ours: Full-coverage LowHigh CO-MIL", "Unit":"Episode / bag", "Input":"RGB-depth video frames", "Curation":"Automatic bag aggregation", "Split":"Animal-wise", "Coverage":1.0000, "Acc5":0.5559, "Macro-F1 5":0.4613, "Weighted-F1 5":0.5550, "Within-one Acc":0.9069, "Notes":"Full-coverage model"},
        {"Study / Model":"Ours: Quality-gated angle-aware CO-MIL", "Unit":"Eligible episode / bag", "Input":"RGB-depth + QC dorsal angle", "Curation":"Automatic quality gate", "Split":"Animal-wise", "Coverage":get_metric(FINAL_MEAN_STD_CSV, "test_coverage"), "Acc5":get_metric(FINAL_MEAN_STD_CSV, "acc5"), "Macro-F1 5":get_metric(FINAL_MEAN_STD_CSV, "macro_f1_5"), "Weighted-F1 5":get_metric(FINAL_MEAN_STD_CSV, "weighted_f1_5"), "Within-one Acc":get_metric(FINAL_MEAN_STD_CSV, "within_one_acc"), "Notes":"Deployment-oriented model with abstention"},
    ]
    save_table(pd.DataFrame(rows), "Table7_contextual_comparison_with_reference")


def write_readme():
    text = f"""Animals paper tables and figures generated under:\n{OUT_DIR}\n\nKey outputs:\nTables: {TABLE_DIR}\nFigures: {FIG_DIR}\n\nImportant notes:\n1. Dorsal angle statistical testing is performed at bag/episode level, not frame level, to avoid pseudo-replication.\n2. Pairwise tests use Mann-Whitney U with Benjamini-Hochberg FDR correction.\n3. Reference-study comparison is contextual only, because input representation, curation, and prediction unit differ.\n4. Final quality-gated model has coverage < 1.0 and should be interpreted as deployment-oriented reliable prediction with abstention.\n"""
    path = OUT_DIR / "README_outputs.txt"
    path.write_text(text, encoding="utf-8")
    print(f"[README] {path}")


def main():
    warnings.filterwarnings("ignore")
    print("="*100)
    print("Generate Animals paper tables and figures")
    print(f"BASE_PATH: {BASE_PATH}")
    print(f"OUT_DIR:   {OUT_DIR}")
    print("="*100)

    make_dataset_tables_and_figures()
    make_main_result_tables_and_figures()
    make_final_confusion_matrices()
    make_dorsal_angle_analysis()
    make_contextual_comparison()
    write_readme()

    print("\nFinished successfully.")
    print(f"Tables:  {TABLE_DIR}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
