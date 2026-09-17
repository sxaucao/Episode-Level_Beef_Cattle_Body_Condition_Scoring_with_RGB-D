import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                             cohen_kappa_score, mean_absolute_error)

BAG_QC_CSV = Path("./03_quality_controlled_dorsal_angle/03_qc_bag_dorsal_angle.csv")
PRED_DIR = Path("./04_full_ablation_results")
PRED_CSV = PRED_DIR / "qg_rgb_depth_lowhigh_with_angle/predictions.csv"

OUT_DIR = Path("./04B_threshold_performance")
OUT_DIR.mkdir(parents=True, exist_ok=True)


THRESHOLDS = [1, 2, 3, 4, 5]
BASE_THR = 3

TRUE_COLS = ["true", "y_true", "true_bcs", "label", "bcs_raw", "gt", "target"]
PRED_COLS = ["pred", "y_pred", "pred_bcs", "predicted", "prediction", "yhat"]
ID_COLS = ["bag_id", "episode_id", "episode", "bag"]


def pick(df, cands, what):
    lower = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(f"找不到{what}列。现有列: {list(df.columns)}")


def metrics(y, yh):
    y = np.asarray(y, dtype=int)
    yh = np.asarray(yh, dtype=int)
    if len(y) == 0:
        return {}
    return {
        "n": len(y),
        "Acc5": accuracy_score(y, yh),
        "Macro-F1": f1_score(y, yh, average="macro", zero_division=0),
        "QWK": cohen_kappa_score(y, yh, weights="quadratic"),
        "MAE": mean_absolute_error(y, yh),
        "Within-one": float(np.mean(np.abs(y - yh) <= 1)),
    }


def boot_diff(c_sub, c_all, n_boot=2000, seed=42):

    rng = np.random.default_rng(seed)
    c_sub = np.asarray(c_sub, dtype=float)
    c_all = np.asarray(c_all, dtype=float)
    ns, na = len(c_sub), len(c_all)
    if ns < 5 or na < 5:
        return np.nan, np.nan, np.nan, np.nan
    dd = np.empty(n_boot)
    for i in range(n_boot):
        si = rng.integers(0, ns, ns)
        ai = rng.integers(0, na, na)
        dd[i] = c_sub[si].mean() - c_all[ai].mean()
    obs = c_sub.mean() - c_all.mean()
    lo, hi = np.percentile(dd, [2.5, 97.5])
    p = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
    return float(obs), float(lo), float(hi), float(min(p, 1.0))


def main():
    print("=" * 100)
    print("04B 难度梯度 / 阈值充分性分析 rev.2（无需重训）")
    print(f"CWD: {Path.cwd()}")
    print("=" * 100)

    for p in [BAG_QC_CSV, PRED_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"缺少: {p}")

    bag = pd.read_csv(BAG_QC_CSV, encoding="utf-8-sig")
    pred = pd.read_csv(PRED_CSV, encoding="utf-8-sig")

    id_c = pick(pred, ID_COLS, "episode id")
    t_c = pick(pred, TRUE_COLS, "true")
    p_c = pick(pred, PRED_COLS, "pred")
    print(f"\n列映射: id={id_c}  true={t_c}  pred={p_c}")

    bag["_id"] = bag["bag_id"].astype(str)
    pred["_id"] = pred[id_c].astype(str)
    bag["_cnt"] = pd.to_numeric(bag["good_angle_frame_count"], errors="coerce").fillna(0).astype(int)
    bag["_valid"] = ~bag["bag_good_angle_median"].isna()

    m = pred.merge(bag[["_id", "_cnt", "_valid"]], on="_id", how="inner")
    if m.empty:
        raise ValueError("合并后为空，检查 bag_id 是否匹配")

    n_pred = m["_id"].nunique()
    n_total_ep = len(bag)
    print(f"\n有预测的 episode: {n_pred}   （阈值 {BASE_THR} 筛出的合格 episode）")
    print(f"总 episode:       {n_total_ep}")
    print(f"注: 模型未对未通过阈值 {BASE_THR} 的 episode 生成预测，")
    print(f"    因此无法评估更宽松阈值(<{BASE_THR})的性能 —— 本表不是阈值扫描。\n")

    y_all = m[t_c].astype(int).values
    yh_all = m[p_c].astype(int).values
    base_m = metrics(y_all, yh_all)
    c_all = (yh_all == y_all).astype(float)

    rows = []
    print("阈值 |  n  | 覆盖率 |  Acc5   | ΔAcc5   | 95%CI               | Macro-F1 |   MAE   | Within-one")
    print("-" * 108)

    for thr in THRESHOLDS:
        sub = m[(m["_cnt"] >= thr) & (m["_valid"])]
        if sub.empty:
            continue
        mm = metrics(sub[t_c], sub[p_c])
        cov = sub["_id"].nunique() / n_total_ep
        d = mm["Acc5"] - base_m["Acc5"]

        if thr == BASE_THR:
            lo = hi = pv = np.nan
            ci_txt = "      (基准)          "
        else:
            c_sub = (sub[p_c].astype(int).values == sub[t_c].astype(int).values).astype(float)
            _, lo, hi, pv = boot_diff(c_sub, c_all)
            ci_txt = f"[{lo:+.4f},{hi:+.4f}] p={pv:.3f}"

        rows.append({
            "min_good_frames": thr,
            "n_episodes": mm["n"],
            "coverage": round(cov, 4),
            "Acc5": round(mm["Acc5"], 4),
            "delta_Acc5_vs_thr3": round(d, 4),
            "ci_lo": None if thr == BASE_THR else round(lo, 4),
            "ci_hi": None if thr == BASE_THR else round(hi, 4),
            "p_value": None if thr == BASE_THR else round(pv, 4),
            "Macro-F1": round(mm["Macro-F1"], 4),
            "QWK": round(mm["QWK"], 4),
            "MAE": round(mm["MAE"], 4),
            "Within-one": round(mm["Within-one"], 4),
        })
        print(f" >= {thr:<2}| {mm['n']:3d} | {cov:.4f} | {mm['Acc5']:.4f} | {d:+.4f} | "
              f"{ci_txt} |  {mm['Macro-F1']:.4f}  | {mm['MAE']:.4f} |  {mm['Within-one']:.4f}")

    df = pd.DataFrame(rows)
    out = OUT_DIR / "T1_threshold_performance_tradeoff.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[saved] {out}")

    # ---------- 自动判读 ----------
    print("\n" + "=" * 100)
    print("自动判读")
    print("=" * 100)

    acc = df["Acc5"].tolist()
    mae = df["MAE"].tolist()
    wo = df["Within-one"].tolist()

    mono_acc_up = all(acc[i] <= acc[i + 1] + 1e-9 for i in range(len(acc) - 1))
    mono_mae_worse = all(mae[i] <= mae[i + 1] + 1e-9 for i in range(len(mae) - 1))
    mono_wo_worse = all(wo[i] >= wo[i + 1] - 1e-9 for i in range(len(wo) - 1))

    print(f"  Acc5 是否随阈值单调上升:        {mono_acc_up}    (期望 False = 非单调 = 噪声)")
    print(f"  MAE  是否随阈值单调变差:        {mono_mae_worse}    (期望 True  = 越严越差)")
    print(f"  Within-one 是否随阈值单调下降:  {mono_wo_worse}    (期望 True  = 越严越差)")

    sub_df = df[df["min_good_frames"] > BASE_THR]
    sig = sub_df[sub_df["p_value"] < 0.05] if not sub_df.empty else pd.DataFrame()
    if sig.empty:
        print(f"\n  >> 所有更严阈值的 Acc5 提升均【不显著】(p >= 0.05)")
    else:
        print(f"\n  >> 以下更严阈值 Acc5 提升显著 (p < 0.05)，需在 Response 中说明:")
        for _, r in sig.iterrows():
            print(f"       阈值 {int(r['min_good_frames'])}: ΔAcc5={r['delta_Acc5_vs_thr3']:+.4f} "
                  f"但 coverage 仅 {r['coverage']:.4f}")

    cov_loss = int(round((1 - df.iloc[-1]["coverage"] / df.iloc[0]["coverage"]) * 100))
    acc_txt = ", ".join(f"{a:.4f}" for a in acc)
    mae_txt = ", ".join(f"{x:.4f}" for x in mae)
    wo_txt = ", ".join(f"{w:.4f}" for w in wo)


    if mono_mae_worse:
        mae_clause = (f'while MAE increased monotonically ({mae_txt}), indicating that '
                      f'stricter gating degraded error magnitude,')
    else:
        mae_clause = (f'and MAE showed no consistent improvement ({mae_txt}),')

    if mono_wo_worse:
        wo_clause = (f' and within-one accuracy decreased monotonically ({wo_txt}).')
    else:
        wo_clause = f' and within-one accuracy showed no consistent gain ({wo_txt}).'




if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
