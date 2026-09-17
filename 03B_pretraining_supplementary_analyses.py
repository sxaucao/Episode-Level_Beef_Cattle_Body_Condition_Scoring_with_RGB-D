import sys
from pathlib import Path

import numpy as np
import pandas as pd

FRAME_INDEX_CSV = Path("./02_make_longitudinal_protocol_splits/02A_animalwise_3fold.csv")

QC_DIR = Path("./03_quality_controlled_dorsal_angle")
FRAME_QC_CSV = QC_DIR / "03_qc_frame_dorsal_angle.csv"
BAG_QC_CSV = QC_DIR / "03_qc_bag_dorsal_angle.csv"
BAD_REASON_CSV = QC_DIR / "03_bad_reason_counts.csv"

OUT_DIR = Path("./03B_pretraining_supplementary_analyses")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_VALID_PROFILES = 3       # 03: MIN_VALID_PROFILES_FOR_GOOD
MIN_GOOD_FRAMES_DEFAULT = 3  # 04: MIN_GOOD_FRAMES_PER_BAG

BCS_LIST = [2, 3, 4, 5, 6]
BCS_NAMES = {2: "BCS2", 3: "BCS3", 4: "BCS4", 5: "BCS5", 6: "BCS6"}



def log(msg=""):
    print(msg)


def save(df: pd.DataFrame, name: str):
    p = OUT_DIR / name
    df.to_csv(p, index=False, encoding="utf-8-sig")
    log(f"  [saved] {p}")
    return p


def compute_eligible_ids(bag_df: pd.DataFrame, min_frames: int = None) -> set:

    m = MIN_GOOD_FRAMES_DEFAULT if min_frames is None else min_frames
    b = bag_df.copy()
    b["good_angle_frame_count"] = pd.to_numeric(
        b["good_angle_frame_count"], errors="coerce").fillna(0).astype(int)
    elig = b[(b["good_angle_frame_count"] >= m) & (~b["bag_good_angle_median"].isna())]
    return set(elig["bag_id"].astype(str).tolist())


def analyze_qc_exclusion(frame_df, bag_df):
    log("\n" + "=" * 88)
    log("[1] QC 阶段排除明细（3815 配对帧 -> 建模帧）")
    log("=" * 88)

    n_paired = len(frame_df)

    bad = frame_df["bad_reasons"].fillna("").astype(str).str.strip()
    has_bad = bad != ""
    n_angles = pd.to_numeric(frame_df["n_angles"], errors="coerce").fillna(0).astype(int)
    insufficient = (~has_bad) & (n_angles < MIN_VALID_PROFILES)
    is_good = frame_df["is_good_angle_frame"].astype(int) == 1

    n_qc_fail = int(has_bad.sum())
    n_insuf = int(insufficient.sum())
    n_good = int(is_good.sum())

    if n_qc_fail + n_insuf + n_good != n_paired:
        log(f"  [WARN] 计数不闭合: {n_qc_fail} + {n_insuf} + {n_good} != {n_paired}")

    eligible_ids = compute_eligible_ids(bag_df)
    n_eligible = len(eligible_ids)
    n_total_ep = len(bag_df)

    ids = frame_df["bag_id"].astype(str)
    n_final = int(((is_good == 1) & (ids.isin(eligible_ids))).sum())

    rows = [
        {"Step": "0",
         "Description": "Paired RGB-D frames entering QC",
         "Frames": n_paired, "Retained": n_paired, "Excluded": 0,
         "Note": "02A index rows (= Table 1 retained 3815)"},
        {"Step": "A",
         "Description": "Excluded: geometry/QC check failed (bad_reasons non-empty)",
         "Frames": n_qc_fail, "Retained": n_paired - n_qc_fail, "Excluded": n_qc_fail,
         "Note": "trunk / posture / contour / centerline failures"},
        {"Step": "B",
         "Description": f"Excluded: < {MIN_VALID_PROFILES} valid transverse-section angles (no other failure)",
         "Frames": n_insuf, "Retained": n_paired - n_qc_fail - n_insuf, "Excluded": n_insuf,
         "Note": "clean but angle-sparse frames"},
        {"Step": "C",
         "Description": "Good-angle frames (after QC)",
         "Frames": n_good, "Retained": n_good, "Excluded": 0,
         "Note": "is_good_angle_frame = 1"},
        {"Step": "D",
         "Description": f"Excluded: good frames in episodes with < {MIN_GOOD_FRAMES_DEFAULT} good frames or invalid median",
         "Frames": n_good - n_final, "Retained": n_final, "Excluded": n_good - n_final,
         "Note": f"{n_total_ep - n_eligible} of {n_total_ep} episodes dropped"},
        {"Step": "E",
         "Description": "Final modelling frames (inside eligible episodes)",
         "Frames": n_final, "Retained": n_final, "Excluded": 0,
         "Note": f"{n_eligible}/{n_total_ep} episodes, coverage {n_eligible / max(n_total_ep, 1):.4f}"},
    ]
    save(pd.DataFrame(rows), "S1_qc_exclusion_breakdown.csv")

    log(f"\n  配对帧 (QC 输入):      {n_paired}")
    log(f"  QC 几何失败 (A):       {n_qc_fail}")
    log(f"  横截面不足 (B):        {n_insuf}")
    log(f"  Good-angle 帧 (C):     {n_good}   (QC 保留率 {n_good / max(n_paired, 1):.4f})")
    log(f"  episode 级剔除 (D):    {n_good - n_final}")
    log(f"  最终建模帧 (E):        {n_final}")
    log(f"  合格 episode:          {n_eligible}/{n_total_ep}  "
        f"coverage = {n_eligible / max(n_total_ep, 1):.4f}")


    if BAD_REASON_CSV.exists():
        try:
            r = pd.read_csv(BAD_REASON_CSV, encoding="utf-8-sig")
            if not r.empty:
                r["count"] = pd.to_numeric(r["count"], errors="coerce").fillna(0).astype(int)
                r["pct_of_qc_input"] = (r["count"] / max(n_paired, 1)).round(4)
                r = r.sort_values("count", ascending=False)
                log("\n  QC 排除原因细项（一帧可命中多项，之和 >= A 类帧数）:")
                for _, x in r.iterrows():
                    log(f"    {str(x['reason']):34s} {int(x['count']):6d} "
                        f"({x['pct_of_qc_input'] * 100:5.1f}%)")
                save(r, "S1b_exclusion_reason_detail.csv")
        except Exception as e:
            log(f"  [WARN] 读取 {BAD_REASON_CSV.name} 失败: {e}")

    return eligible_ids, n_good, n_final, n_eligible


def analyze_threshold_coverage(bag_df):
    log("\n" + "=" * 88)
    log("[2] 阈值-覆盖率（描述性）")
    log("=" * 88)
    log("  !! 本表只反映覆盖率损失，不能证明阈值 3 最优。")
    log("  !! 性能证据见 04B_threshold_performance.py。\n")

    n_total = len(bag_df)
    counts = pd.to_numeric(bag_df["good_angle_frame_count"], errors="coerce").fillna(0).astype(int)
    valid = ~bag_df["bag_good_angle_median"].isna()
    bcs = bag_df["bcs_raw"].astype(int)

    rows = []
    for thr in [1, 2, 3, 4, 5, 6, 8]:
        m = (counts >= thr) & valid
        n_e = int(m.sum())
        row = {"min_good_frames": thr,
               "eligible_episodes": n_e,
               "total_episodes": n_total,
               "coverage": round(n_e / max(n_total, 1), 4),
               "dropped_episodes": n_total - n_e}
        for b in BCS_LIST:
            tot = int((bcs == b).sum())
            kept = int((bcs[m] == b).sum()) if n_e else 0
            row[f"{BCS_NAMES[b]}_kept"] = kept
            row[f"{BCS_NAMES[b]}_total"] = tot
            row[f"{BCS_NAMES[b]}_retention"] = round(kept / tot, 4) if tot else np.nan
        rows.append(row)
        log(f"  >= {thr}: coverage = {n_e / max(n_total, 1):.4f} ({n_e}/{n_total})  "
            + "  ".join(f"{BCS_NAMES[b]}={row[f'{BCS_NAMES[b]}_kept']}/"
                        f"{row[f'{BCS_NAMES[b]}_total']}" for b in BCS_LIST))
    save(pd.DataFrame(rows), "S2_threshold_coverage.csv")


def analyze_intra_episode(frame_df, bag_df, eligible_ids):
    log("\n" + "=" * 88)
    log("[3] Episode 内角度一致性（仅合格 episode）")
    log("=" * 88)

    g = frame_df[
        (frame_df["is_good_angle_frame"].astype(int) == 1)
        & (~frame_df["frame_angle_median"].isna())
        & (frame_df["bag_id"].astype(str).isin(eligible_ids))
        ].copy()

    if g.empty:
        log("  [WARN] 无可用帧，跳过。")
        return pd.DataFrame()

    grp = g.groupby("bag_id")["frame_angle_median"]
    epi = pd.DataFrame({
        "n_good_frames": grp.count(),
        "angle_mean": grp.mean(),
        "angle_median": grp.median(),
        "angle_std": grp.std(),
        "angle_min": grp.min(),
        "angle_max": grp.max(),
    })
    q = grp.quantile([0.25, 0.75]).unstack()
    if q.shape[1] >= 2:
        epi["angle_q25"] = q.iloc[:, 0]
        epi["angle_q75"] = q.iloc[:, 1]
    else:
        epi["angle_q25"] = np.nan
        epi["angle_q75"] = np.nan
    epi["angle_iqr"] = epi["angle_q75"] - epi["angle_q25"]
    epi["angle_range"] = epi["angle_max"] - epi["angle_min"]
    epi["angle_cv"] = epi["angle_std"] / epi["angle_mean"].abs().replace(0, np.nan)
    epi["angle_mad"] = grp.apply(lambda x: float((x - x.median()).abs().median()))

    bcs_map = bag_df.set_index("bag_id")["bcs_raw"].astype(int).to_dict()
    epi["bcs_raw"] = epi.index.map(bcs_map)
    epi = epi.reset_index()
    save(epi, "S3a_intra_episode_per_episode.csv")

    def summ(tag, s):
        if s.empty:
            return None
        return {"BCS": tag,
                "n_episodes": len(s),
                "mean_n_good_frames": round(s["n_good_frames"].mean(), 2),
                "std_mean": round(s["angle_std"].mean(), 4),
                "std_median": round(s["angle_std"].median(), 4),
                "iqr_mean": round(s["angle_iqr"].mean(), 4),
                "iqr_median": round(s["angle_iqr"].median(), 4),
                "mad_mean": round(s["angle_mad"].mean(), 4),
                "range_mean": round(s["angle_range"].mean(), 4),
                "cv_mean": round(s["angle_cv"].mean(), 4)}

    rs = [x for b in BCS_LIST if (x := summ(BCS_NAMES[b], epi[epi["bcs_raw"] == b]))]
    rs.append(summ("ALL", epi))
    summary = pd.DataFrame(rs)
    save(summary, "S3_intra_episode_consistency.csv")

    log(f"  合格 episode 数: {len(epi)}（应 = {len(eligible_ids)}）")
    log("  按 BCS（离散度越小越一致）:")
    for _, r in summary.iterrows():
        log(f"    {r['BCS']:6s} n={int(r['n_episodes']):4d}  "
            f"std={r['std_mean']:6.3f}  IQR={r['iqr_mean']:6.3f}  "
            f"MAD={r['mad_mean']:6.3f}  range={r['range_mean']:6.3f}")
    return summary


def analyze_animal_groups(frame_index_df):
    log("\n" + "=" * 88)
    log("[4] Animal-level groups 构成")
    log("=" * 88)

    if "animal_id" not in frame_index_df.columns:
        log("  [WARN] 缺少 animal_id 列，跳过。")
        return pd.DataFrame()

    animals = frame_index_df["animal_id"].nunique()
    episodes = frame_index_df["episode_id"].nunique()
    frames = len(frame_index_df)

    log(f"  Animals: {animals}   Episodes: {episodes}   Frames: {frames}")

    dist = frame_index_df.groupby("animal_id")["episode_id"].nunique().value_counts().sort_index()
    comp = [{"episodes_per_animal": int(k), "n_animals": int(v)} for k, v in dist.items()]
    log("\n  每个 animal 拥有的 episode 数分布:")
    for c in comp:
        log(f"    {c['episodes_per_animal']} ep: {c['n_animals']:3d} animal(s)")
    tot = sum(c["episodes_per_animal"] * c["n_animals"] for c in comp)
    log(f"  校验 sum = {tot}（应 = {episodes}）")
    if tot != episodes:
        log(f"  [WARN] 分布和不等于 episode 总数，请检查 animal_id 归并逻辑。")

    save(pd.DataFrame(comp), "S4_animal_group_composition.csv")
    save(pd.DataFrame([{
        "n_animals": int(animals),
        "n_episodes": int(episodes),
        "n_frames": int(frames),
        "mean_episodes_per_animal": round(episodes / max(animals, 1), 3),
    }]), "S4a_animal_group_summary.csv")
    return pd.DataFrame(comp)


def analyze_pass_rate(frame_df, bag_df, eligible_ids):
    log("\n" + "=" * 88)
    log("[5] QC 通过率 与 episode 覆盖率")
    log("=" * 88)

    f = frame_df.copy()
    f["bcs_raw"] = f["bcs_raw"].astype(int)
    f["is_good"] = f["is_good_angle_frame"].astype(int)

    # --- 帧级通过率（按 BCS）---
    rows = []
    for b in BCS_LIST:
        s = f[f["bcs_raw"] == b]
        if s.empty:
            continue
        rows.append({"bcs": BCS_NAMES[b],
                     "total_frames": len(s),
                     "good_frames": int(s["is_good"].sum()),
                     "frame_pass_rate": round(s["is_good"].mean(), 4)})
    fr = pd.DataFrame(rows)
    save(fr, "S5a_frame_qc_pass_rate_by_bcs.csv")

    # --- episode 级覆盖率（按 BCS）<< 关键 ---
    b = bag_df.copy()
    b["bcs_raw"] = b["bcs_raw"].astype(int)
    b["fold_id"] = b["fold_id"].astype(int)
    b["eligible"] = b["bag_id"].astype(str).isin(eligible_ids).astype(int)

    er = []
    for x in BCS_LIST:
        s = b[b["bcs_raw"] == x]
        if s.empty:
            continue
        er.append({"bcs": BCS_NAMES[x],
                   "total_episodes": len(s),
                   "eligible_episodes": int(s["eligible"].sum()),
                   "episode_coverage": round(s["eligible"].mean(), 4)})
    ec = pd.DataFrame(er)
    save(ec, "S5c_episode_coverage_by_bcs.csv")

    log("  【关键对比】帧级通过率 -> episode 级覆盖率")
    log("  （聚合后偏差被吸收，回应 R1-3 选择偏差质疑）")
    for _, r in ec.iterrows():
        m = fr[fr["bcs"] == r["bcs"]]
        fp = m.iloc[0]["frame_pass_rate"] if not m.empty else float("nan")
        gain = r["episode_coverage"] - fp
        log(f"    {r['bcs']:6s} {fp:.4f} -> {r['episode_coverage']:.4f}  "
            f"({int(r['eligible_episodes'])}/{int(r['total_episodes'])})   "
            f"提升 {gain:+.4f}")

    # --- episode 覆盖率（按 fold）---
    fold_rows = []
    for fd in sorted(b["fold_id"].unique()):
        s = b[b["fold_id"] == fd]
        fold_rows.append({"fold_id": int(fd),
                          "episodes": len(s),
                          "eligible": int(s["eligible"].sum()),
                          "coverage": round(s["eligible"].mean(), 4)})
    save(pd.DataFrame(fold_rows), "S5b_episode_coverage_by_fold.csv")

    log("\n  episode 覆盖率（按 fold）:")
    for r in fold_rows:
        log(f"    fold {r['fold_id']}: {r['eligible']}/{r['episodes']} = {r['coverage']:.4f}")
    return fr, ec


# ============================================================
# main
# ============================================================
def main():
    log("=" * 88)
    log("03B 训练前补充分析 rev.2（不训练 / 不做消融）")
    log(f"CWD: {Path.cwd()}")
    log("=" * 88)

    for p in [FRAME_QC_CSV, BAG_QC_CSV]:
        if not p.exists():
            raise FileNotFoundError(
                f"缺少输入文件: {p}\n"
                f"当前工作目录: {Path.cwd()}\n"
                f"请先运行 03_quality_controlled_dorsal_angle.py，"
                f"并在项目根目录下运行本脚本。"
            )

    frame_df = pd.read_csv(FRAME_QC_CSV, encoding="utf-8-sig")
    bag_df = pd.read_csv(BAG_QC_CSV, encoding="utf-8-sig")
    log(f"\n  frame QC: {len(frame_df)} 行")
    log(f"  bag   QC: {len(bag_df)} 行")

    fi = None
    if FRAME_INDEX_CSV.exists():
        fi = pd.read_csv(FRAME_INDEX_CSV, encoding="utf-8-sig")
        log(f"  frame idx: {len(fi)} 行")
    else:
        log(f"  [INFO] 未找到 {FRAME_INDEX_CSV.name}，跳过 animal groups 分析。")

    eligible_ids, n_good, n_final, n_eligible = analyze_qc_exclusion(frame_df, bag_df)
    analyze_threshold_coverage(bag_df)
    analyze_intra_episode(frame_df, bag_df, eligible_ids)
    if fi is not None:
        analyze_animal_groups(fi)
    analyze_pass_rate(frame_df, bag_df, eligible_ids)

    log("\n" + "=" * 88)
    log(f"全部完成。输出目录: {OUT_DIR.resolve()}")
    log("=" * 88)
    log("输出清单:")
    log("  S1_qc_exclusion_breakdown.csv     QC 排除明细 -> 投稿 Table S1")
    log("  S1b_exclusion_reason_detail.csv   排除原因细项")
    log("  S2_threshold_coverage.csv         阈值-覆盖率（描述性）")
    log("  S3_intra_episode_consistency.csv  episode 内一致性（仅合格）")
    log("  S4_animal_group_composition.csv   52 animal groups 构成")
    log("  S5a_frame_qc_pass_rate_by_bcs.csv 帧级通过率")
    log("  S5c_episode_coverage_by_bcs.csv   episode 覆盖率 << R1-3 关键")
    log("  S5b_episode_coverage_by_fold.csv  fold 覆盖率")
    log("=" * 88)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
