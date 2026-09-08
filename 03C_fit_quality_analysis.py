"""
03C_fit_quality_analysis.py  (rev.2 — 口径修正版)
=========================================================================
回应 Reviewer 1 — Comment 8：
  "The angle-fitting procedure does not use a separate outlier-removal or
   goodness-of-fit criterion, which may reduce robustness when depth
   profiles are noisy."

【rev.2 关键修正】
  旧版 episode 角度用"每帧取中位数、再对帧中位数取中位数"（per-frame-median 口径），
  而 03 脚本的 bag_good_angle_median 用的是"把所有 good 帧的全部有效横截面角度
  池化后取中位数"（pooled 口径，见 03 源码 line 1100: bag_good_angles.extend(angles)
  位于 if is_good: 内）。
  两种口径系统性不同，导致旧版在 R^2 阈值 = 0（完全不过滤）时 Δ 仍为 0.441°，
  使整个敏感性分析混入了常数偏移，不可用。

  rev.2 改为：
    - 主口径 = pooled（与 03 的 bag_good_angle_median 严格一致）
    - 阈值 = 0 时 Δ 必须 = 0，脚本内置 [PASS]/[FAIL] 自检
    - per-frame-median 口径作为第二列一并输出，供对照

本脚本【不重训模型】，仅用 03 新增的逐横截面拟合优度明细做两件事：

  [A] R^2 分布诊断
      所有横截面双侧 OLS 的 R^2（决定系数）分布，按 BCS 分组

  [B] 离群剔除敏感性分析（核心）
      模拟"若额外施加 R^2 过滤会怎样"：
        对每个阈值 thr，剔除 r2_min < thr 的横截面，然后
          (1) 重算每帧角度   = 剩余横截面 angle 的中位数
          (2) 重判 Good 帧   = 剩余横截面数 >= MIN_VALID_PROFILES 且原为 Good 帧
          (3) 重算 episode 角度（pooled 口径，对齐 03）
          (4) 重判 eligible  = 新 Good 帧数 >= MIN_GOOD_FRAMES 且角度有效
        输出：剔除比例、episode 角度变化量、覆盖率变化

输入（相对路径，CWD = 项目根目录）：
  ./03_quality_controlled_dorsal_angle/03_profile_fit_quality.csv   <- 03 重跑后新增
  ./03_quality_controlled_dorsal_angle/03_qc_frame_dorsal_angle.csv
  ./03_quality_controlled_dorsal_angle/03_qc_bag_dorsal_angle.csv

输出目录：./03C_fit_quality_analysis/
  G1_r2_distribution.csv            R^2 分布（全体 + 按 BCS）
  G2_outlier_filter_sensitivity.csv 离群剔除敏感性（核心表）
  G3_frame_angle_shift.csv          帧级角度变化明细
  G4_episode_angle_shift.csv        episode 级角度变化明细
=========================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 路径（相对路径，CWD = 项目根目录）
# ============================================================
QC_DIR = Path("./03_quality_controlled_dorsal_angle")
PROFILE_CSV = QC_DIR / "03_profile_fit_quality.csv"
FRAME_QC_CSV = QC_DIR / "03_qc_frame_dorsal_angle.csv"
BAG_QC_CSV = QC_DIR / "03_qc_bag_dorsal_angle.csv"

OUT_DIR = Path("./03C_fit_quality_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 与 03/04 一致
MIN_VALID_PROFILES = 3
MIN_GOOD_FRAMES = 3

# R^2 阈值扫描
R2_THRESHOLDS = [0.00, 0.50, 0.70, 0.80, 0.90, 0.95]

BCS_LIST = [2, 3, 4, 5, 6]
BCS_NAMES = {2: "BCS2", 3: "BCS3", 4: "BCS4", 5: "BCS5", 6: "BCS6"}


def log(msg=""):
    print(msg)


def save(df: pd.DataFrame, name: str):
    p = OUT_DIR / name
    df.to_csv(p, index=False, encoding="utf-8-sig")
    log(f"  [saved] {p}")
    return p


# ============================================================
# [A] R^2 分布
# ============================================================
def analyze_r2_distribution(prof: pd.DataFrame):
    log("\n" + "=" * 92)
    log("[A] 横截面拟合优度 R^2 分布")
    log("=" * 92)

    p = prof.copy()
    p["r2_min"] = pd.to_numeric(p["r2_min"], errors="coerce")
    valid = p.dropna(subset=["r2_min"])
    log(f"  有效横截面数: {len(valid)}")

    rows = []
    for bcs in BCS_LIST:
        s = valid[valid["bcs_raw"].astype(int) == bcs]
        if s.empty:
            continue
        rows.append({
            "BCS": BCS_NAMES[bcs],
            "n_profiles": len(s),
            "mean": round(s["r2_min"].mean(), 4),
            "q05": round(s["r2_min"].quantile(0.05), 4),
            "q10": round(s["r2_min"].quantile(0.10), 4),
            "q25": round(s["r2_min"].quantile(0.25), 4),
            "q50": round(s["r2_min"].quantile(0.50), 4),
            "q75": round(s["r2_min"].quantile(0.75), 4),
            "frac_below_0.80": round(float((s["r2_min"] < 0.80).mean()), 4),
        })

    rows.append({
        "BCS": "ALL",
        "n_profiles": len(valid),
        "mean": round(valid["r2_min"].mean(), 4),
        "q05": round(valid["r2_min"].quantile(0.05), 4),
        "q10": round(valid["r2_min"].quantile(0.10), 4),
        "q25": round(valid["r2_min"].quantile(0.25), 4),
        "q50": round(valid["r2_min"].quantile(0.50), 4),
        "q75": round(valid["r2_min"].quantile(0.75), 4),
        "frac_below_0.80": round(float((valid["r2_min"] < 0.80).mean()), 4),
    })

    dist = pd.DataFrame(rows)
    save(dist, "G1_r2_distribution.csv")

    log("\n  按 BCS（r2_min = 左右两侧 R^2 的较小者，越大越好）:")
    for _, r in dist.iterrows():
        log(f"    {r['BCS']:6s} n={int(r['n_profiles']):6d}  mean={r['mean']:.4f}  "
            f"median(q50)={r['q50']:.4f}  q05={r['q05']:.4f}  q10={r['q10']:.4f}  "
            f"低于0.80占比={r['frac_below_0.80']:.4f}")

    return dist


# ============================================================
# [B] 离群剔除敏感性分析（rev.2：pooled 口径 + thr=0 自检）
# ============================================================
def analyze_outlier_filter(prof: pd.DataFrame, frame_df: pd.DataFrame, bag_df: pd.DataFrame):
    log("\n" + "=" * 92)
    log("[B] 离群剔除敏感性分析（模拟额外 R^2 过滤）")
    log("=" * 92)

    p = prof.copy()
    p["r2_min"] = pd.to_numeric(p["r2_min"], errors="coerce")
    p["angle"] = pd.to_numeric(p["angle"], errors="coerce")
    p = p.dropna(subset=["angle"])
    p["_fk"] = p["bag_id"].astype(str) + "||" + p["frame_key"].astype(str)
    p["bid"] = p["bag_id"].astype(str)

    # ---- 基准帧级 ----
    f = frame_df.copy()
    f["_fk"] = f["bag_id"].astype(str) + "||" + f["frame_key"].astype(str)
    f["is_good"] = f["is_good_angle_frame"].astype(int)
    good_base_fks = set(f.loc[f["is_good"] == 1, "_fk"])

    # ---- 基准 episode 级（03 的 bag_good_angle_median = pooled 口径）----
    b = bag_df.copy()
    b["bag_id"] = b["bag_id"].astype(str)
    n_episodes = len(b)
    base_bag_angle = dict(zip(b["bag_id"],
                              pd.to_numeric(b["bag_good_angle_median"], errors="coerce")))
    base_counts = pd.to_numeric(b["good_angle_frame_count"], errors="coerce").fillna(0).astype(int)
    base_valid = ~pd.to_numeric(b["bag_good_angle_median"], errors="coerce").isna()
    base_eligible = set(b.loc[(base_counts >= MIN_GOOD_FRAMES) & base_valid, "bag_id"])

    log(f"  基准 episode: {n_episodes}  基准合格 episode: {len(base_eligible)}  "
        f"coverage = {len(base_eligible) / max(n_episodes, 1):.4f}")

    rows = []
    shift_rows = []
    sanity_result = None

    log("\n  阈值 | 剔除横截面% | Δepisode(pooled,均值|中位|最大) | Δ(帧中位口径) | 仍合格 | 覆盖率")
    log("  " + "-" * 100)

    for thr in R2_THRESHOLDS:
        if thr <= 0:
            kept = p.copy()
        else:
            kept = p[(p["r2_min"].isna()) | (p["r2_min"] >= thr)].copy()

        n_all, n_kept = len(p), len(kept)
        pct_removed = (n_all - n_kept) / max(n_all, 1)

        # ---- 只保留原本是 Good 的帧 ----
        kept_good = kept[kept["_fk"].isin(good_base_fks)].copy()

        if kept_good.empty:
            rows.append({"r2_threshold": thr, "pct_profiles_removed": round(pct_removed, 4),
                         "n_profiles_kept": n_kept})
            continue

        # ---- 帧级：重算角度 + 剩余横截面数 ----
        g = kept_good.groupby("_fk")["angle"]
        frame_med = g.median()
        frame_n = g.count()

        # ---- 帧级：重判 Good（剩余横截面数 >= MIN_VALID_PROFILES）----
        valid_fks = set(frame_n[frame_n >= MIN_VALID_PROFILES].index)
        kept_valid = kept_good[kept_good["_fk"].isin(valid_fks)].copy()

        # ---- episode 级 口径1：pooled（对齐 03 的 bag_good_angle_median）----
        # 03 的 safe_stats() 用 np.array(values, dtype=np.float32) 后取中位数，
        # 这里同样转 float32 再取中位数，消除 float64/float32 舍入带来的 ~1e-6 差异。
        if not kept_valid.empty:
            ep_pooled = kept_valid.groupby("bid")["angle"].apply(
                lambda x: float(np.median(np.asarray(x, dtype=np.float32)))
            )
        else:
            ep_pooled = pd.Series(dtype=float)

        # ---- episode 级 口径2：per-frame-median（每帧中位数再取中位数）----
        fm = frame_med[frame_med.index.isin(valid_fks)]
        if len(fm) > 0:
            fm_df = fm.rename("angle").reset_index()
            # 注意：必须用 regex=False，否则 "||" 被当作正则（空或空）导致全部切成 ''
            fm_df["bid"] = fm_df["_fk"].astype(str).str.split("||", regex=False).str[0]
            ep_fm = fm_df.groupby("bid")["angle"].median()
        else:
            ep_fm = pd.Series(dtype=float)

        # ---- 重判 eligible：新 Good 帧数 >= MIN_GOOD_FRAMES ----
        fm_counts = fm_df.groupby("bid")["angle"].count() if len(fm) > 0 else pd.Series(dtype=int)
        new_eligible = set(fm_counts[fm_counts >= MIN_GOOD_FRAMES].index)
        new_eligible &= set(ep_pooled.dropna().index)

        # ---- Δ 计算（主口径 = pooled，与 base_bag_angle 同口径）----
        deltas = []
        deltas_fm = []
        for bid in sorted(new_eligible):
            base_ang = base_bag_angle.get(bid, np.nan)
            if np.isnan(base_ang):
                continue
            new_ang = float(ep_pooled.get(bid, np.nan))
            new_ang_fm = float(ep_fm.get(bid, np.nan))
            if np.isnan(new_ang):
                continue
            d = new_ang - base_ang
            deltas.append(abs(d))
            d_fm = (new_ang_fm - base_ang) if not np.isnan(new_ang_fm) else np.nan
            deltas_fm.append(abs(d_fm) if not np.isnan(d_fm) else np.nan)
            shift_rows.append({
                "bag_id": bid,
                "base_episode_angle_pooled": round(float(base_ang), 4),
                "new_episode_angle_pooled": round(new_ang, 4),
                "new_episode_angle_framemedian": None if np.isnan(new_ang_fm) else round(new_ang_fm, 4),
                "abs_delta_pooled": round(abs(d), 4),
                "signed_delta_pooled": round(float(d), 4),
                "abs_delta_framemedian": None if np.isnan(d_fm) else round(abs(d_fm), 4),
            })

        deltas_fm_clean = [x for x in deltas_fm if x is not None and not np.isnan(x)]

        if deltas:
            mean_d = float(np.mean(deltas))
            med_d = float(np.median(deltas))
            max_d = float(np.max(deltas))
        else:
            mean_d = med_d = max_d = np.nan

        mean_d_fm = float(np.mean(deltas_fm_clean)) if deltas_fm_clean else np.nan

        cov = len(new_eligible) / max(n_episodes, 1)

        rows.append({
            "r2_threshold": thr,
            "pct_profiles_removed": round(pct_removed, 4),
            "n_profiles_kept": int(n_kept),
            "mean_abs_delta_pooled_deg": round(mean_d, 4) if not np.isnan(mean_d) else None,
            "median_abs_delta_pooled_deg": round(med_d, 4) if not np.isnan(med_d) else None,
            "max_abs_delta_pooled_deg": round(max_d, 4) if not np.isnan(max_d) else None,
            "mean_abs_delta_framemedian_deg": round(mean_d_fm, 4) if not np.isnan(mean_d_fm) else None,
            "eligible_episodes": len(new_eligible),
            "total_episodes": n_episodes,
            "coverage": round(cov, 4),
            "delta_coverage": round(cov - len(base_eligible) / max(n_episodes, 1), 4),
        })

        log(f"  {thr:4.2f} |    {pct_removed * 100:6.2f}%   | "
            f"{mean_d:.3f} | {med_d:.3f} | {max_d:.3f} | "
            f"{mean_d_fm if not np.isnan(mean_d_fm) else float('nan'):.3f}        | "
            f"{len(new_eligible):4d}  | {cov:.4f}")

        # ---- 自检：thr = 0 时 pooled Δ 必须 = 0 ----
        if thr <= 0:
            sanity_result = {
                "pooled_mean_delta": mean_d,
                "framemedian_mean_delta": mean_d_fm,
                "coverage": cov,
                "eligible": len(new_eligible),
            }

    out = pd.DataFrame(rows)
    save(out, "G2_outlier_filter_sensitivity.csv")

    if shift_rows:
        save(pd.DataFrame(shift_rows).sort_values("abs_delta_pooled", ascending=False),
             "G4_episode_angle_shift.csv")

    # ============================================================
    # 口径自检报告
    # ============================================================
    log("\n" + "=" * 92)
    log("  口径对齐自检（阈值 = 0 时，必须 Δ = 0）")
    log("=" * 92)
    if sanity_result:
        pd_d = sanity_result["pooled_mean_delta"]
        fm_d = sanity_result["framemedian_mean_delta"]
        # 判据分三档：
        #   < 1e-3  -> PASS（float32 舍入噪声，口径已对齐）
        #   < 0.05  -> WARN（可能有少量横截面集合差异，Δ 仍可解读但需留意）
        #   >= 0.05 -> FAIL（口径或集合不一致，G2 表不可用）
        TOL_PASS = 1e-3
        TOL_WARN = 0.05
        if np.isnan(pd_d):
            ok, level = False, "FAIL"
        elif abs(pd_d) < TOL_PASS:
            ok, level = True, "PASS"
        elif abs(pd_d) < TOL_WARN:
            ok, level = True, "WARN"
        else:
            ok, level = False, "FAIL"

        log(f"  [{level}] pooled 口径（对齐 03 的 bag_good_angle_median）"
            f"  Δ均值 = {pd_d:.6f}°")
        log(f"  [INFO] 帧中位数口径（旧版误用，仅供参考）      Δ均值 = {fm_d:.6f}°")
        log(f"  [INFO] 覆盖率 = {sanity_result['coverage']:.4f}  "
            f"合格 episode = {sanity_result['eligible']}")
        log(f"  [INFO] 判据: |Δ| < {TOL_PASS} -> PASS, < {TOL_WARN} -> WARN, 否则 FAIL")
        log(f"  [INFO] 注: 03 的 safe_stats() 使用 float32，与 float64 重算存在 ~1e-6 舍入差异，属正常。")

        if level == "FAIL":
            log("\n  [严重] 阈值=0 时 pooled Δ 过大，说明 profile CSV 与 03 的 angles 集合")
            log("         不完全一致（例如 CSV 含无效横截面）。此时 G2 表不可用，请检查 03 补丁。")
        elif level == "WARN":
            log("\n  [注意] 阈值=0 时 pooled Δ 略大于浮点噪声，可能少量横截面集合有差异。")
            log("         G2 表的 Δ 可解读，但建议在正文注明该基线偏移。")
        else:
            log("\n  [OK] pooled 口径与 03 完全一致（差异为 float32 舍入噪声），")
            log("       G2 表的 Δ 可直接解读为'过滤效应'。")
            log(f"  [OK] 旧版 0.441° 的偏移 = 帧中位数口径与 pooled 口径之差"
                f"（{fm_d:.3f}°），已剔除。")

    return out


# ============================================================
# 自动判读
# ============================================================
def interpret(sens: pd.DataFrame):
    log("\n" + "=" * 92)
    log("  自动判读（基于 pooled 口径）")
    log("=" * 92)

    base_cov = sens.loc[sens["r2_threshold"] <= 0, "coverage"].iloc[0]

    for thr in [0.70, 0.80, 0.90, 0.95]:
        r = sens[sens["r2_threshold"] == thr]
        if r.empty:
            continue
        r = r.iloc[0]
        pct = r["pct_profiles_removed"]
        d = r["mean_abs_delta_pooled_deg"]
        dcov = r["delta_coverage"]
        if d is None or np.isnan(d):
            continue
        verdict = ("过滤无实质收益（角度几乎不变、覆盖率下降）"
                   if (d < 0.5 and dcov <= 0) else "需人工判读")
        log(f"  R^2 >= {thr:.2f}: 剔除 {pct * 100:5.2f}% 横截面, "
            f"Δepisode 均值 {d:.3f}°, 覆盖率 {dcov:+.4f}  -> {verdict}")

    # 生成结论模板（用 0.80 行）
    r = sens[sens["r2_threshold"] == 0.80]
    if not r.empty:
        r = r.iloc[0]
        d80 = r["mean_abs_delta_pooled_deg"]
        max80 = r["max_abs_delta_pooled_deg"]
        if d80 is None or max80 is None or np.isnan(d80):
            log("\n  [WARN] 0.80 行的 Δ 为空，无法生成结论模板。请先修复上方自检失败。")
            return
        log("\n  结论模板（可直接写进 Response，数字已自动代入）:")
        log(f'    "Applying an additional goodness-of-fit filter (retaining only')
        log(f'     transverse sections with R^2 >= 0.80) removed {r["pct_profiles_removed"] * 100:.1f}% of')
        log(f'     transverse sections but changed the episode-level angle by only')
        log(f'     {d80:.3f} deg on average (max {max80:.3f} deg),')
        log(f'     while reducing coverage from {base_cov:.4f} to {r["coverage"]:.4f}.')
        log(f'     This indicates that the existing QC stage already removes')
        log(f'     geometrically unreliable profiles, and a separate outlier-removal')
        log(f'     step would discard usable data without materially changing')
        log(f'     the estimates."')


# ============================================================
# main
# ============================================================
def main():
    log("=" * 92)
    log("03C 拟合优度分析 rev.2（R1-8，不重训；pooled 口径对齐 03）")
    log(f"CWD: {Path.cwd()}")
    log("=" * 92)

    for fp in [PROFILE_CSV, FRAME_QC_CSV, BAG_QC_CSV]:
        if not fp.exists():
            raise FileNotFoundError(
                f"缺少: {fp}\nCWD: {Path.cwd()}\n"
                f"请先重跑 03（需带 R^2 补丁版本）。"
            )

    prof = pd.read_csv(PROFILE_CSV, encoding="utf-8-sig")
    frame_df = pd.read_csv(FRAME_QC_CSV, encoding="utf-8-sig")
    bag_df = pd.read_csv(BAG_QC_CSV, encoding="utf-8-sig")

    log(f"\n  profile 明细: {len(prof)} 行  <- {PROFILE_CSV.name}")
    log(f"  frame   QC : {len(frame_df)} 行")
    log(f"  bag     QC : {len(bag_df)} 行")

    analyze_r2_distribution(prof)
    sens = analyze_outlier_filter(prof, frame_df, bag_df)
    interpret(sens)

    log("\n" + "=" * 92)
    log(f"完成。输出目录: {OUT_DIR.resolve()}")
    log("=" * 92)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
