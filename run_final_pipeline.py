import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path



BASE_PATH = Path(r"./")

QC_SCRIPT = BASE_PATH / "03_quality_controlled_dorsal_angle.py"
SUPP_SCRIPT = BASE_PATH / "03B_pretraining_supplementary_analyses.py"
FITQ_SCRIPT = BASE_PATH / "03C_fit_quality_analysis.py"
TRAIN_SCRIPT = BASE_PATH / "04_final_clean_quality_gated_angle_aware_lowhigh_comil.py"

QC_DIR = BASE_PATH / "./03_quality_controlled_dorsal_angle"
FRAME_QC_CSV = QC_DIR / "03_qc_frame_dorsal_angle.csv"
BAG_QC_CSV = QC_DIR / "03_qc_bag_dorsal_angle.csv"
GOOD_FRAME_CSV = QC_DIR / "03_good_frame_index.csv"
PROFILE_FITQ_CSV = QC_DIR / "03_profile_fit_quality.csv"

SUPP_OUT_DIR = BASE_PATH / "03B_pretraining_supplementary_analyses"
FITQ_OUT_DIR = BASE_PATH / "03C_fit_quality_analysis"
TRAIN_OUT_DIR = BASE_PATH / "04_final_clean_quality_gated_angle_aware_lowhigh_comil_results"

LOG_DIR = BASE_PATH / "pipeline_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_command(cmd, cwd: Path, log_path: Path):
    print("=" * 100)
    print("Running:", " ".join(str(x) for x in cmd))
    print("CWD:", cwd)
    print("Log:", log_path)
    print("=" * 100)

    start = time.time()

    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write("Command: " + " ".join(str(x) for x in cmd) + "\n")
        log_f.write("CWD: " + str(cwd) + "\n")
        log_f.write("Start: " + datetime.now().isoformat() + "\n")
        log_f.write("=" * 100 + "\n")
        log_f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)

        ret = proc.wait()
        elapsed = time.time() - start
        log_f.write("\n" + "=" * 100 + "\n")
        log_f.write(f"Return code: {ret}\n")
        log_f.write(f"Elapsed seconds: {elapsed:.1f}\n")
        log_f.write("End: " + datetime.now().isoformat() + "\n")

    if ret != 0:
        raise RuntimeError(f"Command failed with return code {ret}. Check log: {log_path}")

    print("\nFinished successfully.")
    print(f"Elapsed: {elapsed:.1f}s")


def check_base_files():
    if not BASE_PATH.exists():
        raise FileNotFoundError(f"BASE_PATH not found: {BASE_PATH}")

    index_csv = Path("./02_make_longitudinal_protocol_splits/02A_animalwise_3fold.csv")
    if not index_csv.exists():
        raise FileNotFoundError(
            f"Cannot find {index_csv}. Run the animal-wise fold/index script first."
        )


def qc_outputs_exist() -> bool:
    return FRAME_QC_CSV.exists() and BAG_QC_CSV.exists() and GOOD_FRAME_CSV.exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-qc",
        action="store_true",
        help="Regenerate 03 QC CSV files even if they already exist.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only generate/check 03 QC outputs; do not train final model.",
    )
    parser.add_argument(
        "--skip-qc",
        action="store_true",
        help="Do not run 03 QC. Require existing QC CSV files.",
    )
    parser.add_argument(
        "--skip-supplementary",
        action="store_true",
        help="Skip 03B pre-training supplementary analyses "
             "(exclusion breakdown / threshold sensitivity / intra-episode consistency).",
    )
    parser.add_argument(
        "--skip-fitq",
        action="store_true",
        help="Skip 03C fit-quality analysis (R1-8: R^2 distribution and "
             "outlier-removal sensitivity). Requires the R^2-patched 03 script.",
    )
    args = parser.parse_args()

    print("=" * 100)
    print("Final CLEAN quality-gated angle-aware LowHigh CO-MIL pipeline")
    print("BASE_PATH:", BASE_PATH)
    print("=" * 100)

    check_base_files()

    if args.skip_qc:
        print("[Info] --skip-qc was set. Will not run 03 QC script.")
        if not qc_outputs_exist():
            raise FileNotFoundError(
                "QC outputs are missing, but --skip-qc was set. Missing expected files:\n"
                f"  {FRAME_QC_CSV}\n"
                f"  {BAG_QC_CSV}\n"
                f"  {GOOD_FRAME_CSV}"
            )
    else:
        need_qc = args.force_qc or (not qc_outputs_exist())

        if need_qc:
            if not QC_SCRIPT.exists():
                raise FileNotFoundError(
                    f"QC script not found: {QC_SCRIPT}\n"
                    "Put 03_quality_controlled_dorsal_angle.py in BASE_PATH first."
                )

            qc_log = LOG_DIR / f"03_qc_{now_stamp()}.log"
            run_command([sys.executable, str(QC_SCRIPT)], cwd=BASE_PATH, log_path=qc_log)

            if not qc_outputs_exist():
                raise RuntimeError(
                    "03 QC finished, but expected output files are still missing:\n"
                    f"  {FRAME_QC_CSV}\n"
                    f"  {BAG_QC_CSV}\n"
                    f"  {GOOD_FRAME_CSV}"
                )
        else:
            print("[Info] 03 QC outputs already exist. Skipping QC generation.")
            print("       Use --force-qc to regenerate them.")

    print("\nQC outputs ready:")
    print(" ", FRAME_QC_CSV)
    print(" ", BAG_QC_CSV)
    print(" ", GOOD_FRAME_CSV)

    # ------------------------------------------------------------
    # 03B: 训练前补充分析（不训练模型、不做消融）
    # 必须在 04 训练之前跑完，因为训练后数值可能变化
    # ------------------------------------------------------------
    if args.skip_supplementary:
        print("\n[Info] --skip-supplementary was set. Skipping 03B analyses.")
    elif not SUPP_SCRIPT.exists():
        print(f"\n[WARN] 03B script not found: {SUPP_SCRIPT}")
        print("       Skipping pre-training supplementary analyses.")
    else:
        supp_log = LOG_DIR / f"03B_supplementary_{now_stamp()}.log"
        run_command([sys.executable, str(SUPP_SCRIPT)], cwd=BASE_PATH, log_path=supp_log)
        print("\nSupplementary analyses (03B) finished. Outputs:")
        print(" ", SUPP_OUT_DIR / "S1_qc_exclusion_breakdown.csv")
        print(" ", SUPP_OUT_DIR / "S2_threshold_coverage.csv")
        print(" ", SUPP_OUT_DIR / "S3_intra_episode_consistency.csv")

    # ------------------------------------------------------------
    # 03C: 拟合优度分析（R1-8，不重训）
    # 依赖 03 打过 R^2 补丁后新增的 03_profile_fit_quality.csv
    # 若该文件不存在（03 未重跑），警告并跳过，不中断流程
    # ------------------------------------------------------------
    if args.skip_fitq:
        print("\n[Info] --skip-fitq was set. Skipping 03C fit-quality analysis.")
    elif not FITQ_SCRIPT.exists():
        print(f"\n[WARN] 03C script not found: {FITQ_SCRIPT}")
        print("       Skipping fit-quality analysis.")
    elif not PROFILE_FITQ_CSV.exists():
        print(f"\n[WARN] {PROFILE_FITQ_CSV} not found.")
        print("       请先用【打过 R^2 补丁的】03 脚本重跑一次（--force-qc），")
        print("       然后再运行 03C_fit_quality_analysis.py。本次跳过。")
    else:
        fitq_log = LOG_DIR / f"03C_fitq_{now_stamp()}.log"
        run_command([sys.executable, str(FITQ_SCRIPT)], cwd=BASE_PATH, log_path=fitq_log)
        print("\nFit-quality analysis (03C) finished. Outputs:")
        print(" ", FITQ_OUT_DIR / "G2_outlier_filter_sensitivity.csv")

    if args.skip_train:
        print("\n[Info] --skip-train was set. Pipeline stopped after QC check/generation.")
        return

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(
            f"Training script not found: {TRAIN_SCRIPT}\n"
            "Put 04_final_clean_quality_gated_angle_aware_lowhigh_comil.py in BASE_PATH first."
        )

    train_log = LOG_DIR / f"04_final_clean_train_{now_stamp()}.log"
    run_command([sys.executable, str(TRAIN_SCRIPT)], cwd=BASE_PATH, log_path=train_log)

    print("\n" + "=" * 100)
    print("Pipeline finished.")
    print("Final result directory:")
    print(TRAIN_OUT_DIR)
    print("Useful files to inspect:")
    print(TRAIN_OUT_DIR / "04_final_aggregate_confusion_report.txt")
    print(TRAIN_OUT_DIR / "04_angle_aware_mean_std.csv")
    print(TRAIN_OUT_DIR / "04_angle_aware_predictions.csv")
    print("Logs:")
    print(LOG_DIR)
    print("=" * 100)


if __name__ == "__main__":
    main()
