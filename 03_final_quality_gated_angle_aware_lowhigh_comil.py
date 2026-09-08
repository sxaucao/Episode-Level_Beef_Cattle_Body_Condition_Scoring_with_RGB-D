from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report, cohen_kappa_score,
    confusion_matrix, f1_score,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = nn = F = None
    Dataset = object


NUM_CLASSES = 5
N_SPLITS = 3
CLASS5_NAMES = ["BCS2", "BCS3", "BCS4", "BCS5", "BCS6"]
CLASS3_NAMES = ["Lean", "Ideal", "High"]
LABEL5_TO_RAW = dict(enumerate(range(2, 7)))
LABEL5_TO_LABEL3 = np.array([0, 0, 1, 1, 2], dtype=np.int64)


@dataclass
class Config:
    # 路径相对于 base_path；Windows 用户可通过 --base-path 指定 D:\\...。
    base_path: str = "."
    frame_qc_csv: str = "03_quality_controlled_dorsal_angle/03_qc_frame_dorsal_angle.csv"
    bag_qc_csv: str = "03_quality_controlled_dorsal_angle/03_qc_bag_dorsal_angle.csv"
    out_dir: str = "04_paper_aligned_comil_results"

    # 文章 2.2.1、2.3.4：随机动物级验证划分，无类别分层或稀有类别优先。
    seed: int = 42                    # 文章未指定；沿用原脚本。
    val_ratio: float = 0.22
    epochs: int = 70
    img_size: int = 160
    frames_per_bag: int = 12
    batch_size: int = 8
    num_workers: int = 4
    prefetch_factor: int = 2          # 文章未指定；沿用原脚本。
    lr: float = 2e-4
    weight_decay: float = 1e-4
    cosine_eta_min: float = 0.0       # 文章未指定；CosineAnnealingLR 默认值。
    amp: bool = True                  # 仅 CUDA 启用；CPU 自动关闭。
    grad_clip_norm: float = 5.0
    deterministic: bool = True       # 可重复性补充设置。
    device: str = "auto"             # auto / cpu / cuda / cuda:0 等。

    # 文章 2.3.4：先裁剪，再对有效非零深度求百分位数，最后反转为 closeness。
    depth_center_crop_ratio: float = 0.70
    depth_percentile_low: float = 1.0
    depth_percentile_high: float = 99.0
    # RGB: x/127.5-1；Depth: [0,1] closeness，缺失值为0；均无翻转/旋转增强。

    # 文章 2.2.4：只以数量和有效角度决定资格，不额外增加 Good-frame 比例门槛。
    min_good_frames_per_bag: int = 3
    frame_angles_column: str = ""     # 可选：每帧有效横截面角度 JSON 数组列。
    frame_order_column: str = ""      # 可选：时间戳/帧序号列；否则自然排序 frame_key。
    check_files: bool = True

    # 文章 2.3.1–2.3.3：CNN 32/64/128/192，模态128，视觉256，几何32，共享192。
    conv_channels: tuple = (32, 64, 128, 192)
    modality_dim: int = 128
    frame_fusion_dim: int = 256
    frame_dropout: float = 0.20
    attention_hidden_dim: int = 128   # 文章未给出；沿用原 attention MLP 隐藏维数。
    angle_dim: int = 3
    angle_hidden_dim: int = 32        # 文章未指定第一层宽度；沿用原脚本32。
    angle_embedding_dim: int = 32
    angle_dropout: float = 0.10
    shared_dim: int = 192
    angle_only_hidden_dim: int = 64

    # 文章 2.3.4：加权CE + 0.45*BCE_low + 0.45*BCE_high。
    lambda_low: float = 0.45
    lambda_high: float = 0.45
    selection_qwk_weight: float = 0.35
    selection_extreme_weight: float = 0.04
    # Macro-F1缺失类别口径文章未写清；默认保留原脚本 sklearn 的 observed 口径。
    # fixed 表示始终计入全部5/3个类别；请在同一实验中保持口径一致。
    macro_f1_labels: str = "observed"

    def validate(self):
        for field in fields(self):
            default, value = field.default, getattr(self, field.name)
            if isinstance(default, bool):
                if not isinstance(value, bool):
                    raise ValueError(f"{field.name} 必须为布尔值true/false")
            elif isinstance(default, int):
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(f"{field.name} 必须为整数")
            elif isinstance(default, float):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{field.name} 必须为有限数值")
            elif isinstance(default, str) and not isinstance(value, str):
                raise ValueError(f"{field.name} 必须为字符串")
        for name in ("epochs", "img_size", "frames_per_bag", "batch_size",
                     "prefetch_factor", "min_good_frames_per_bag", "modality_dim",
                     "frame_fusion_dim", "attention_hidden_dim", "angle_hidden_dim",
                     "angle_embedding_dim", "shared_dim", "angle_only_hidden_dim"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正整数")
        if self.img_size < 16 or self.num_workers < 0 or self.seed < 0:
            raise ValueError("img_size >= 16，num_workers >= 0，seed >= 0")
        if not 0 < self.val_ratio < 1:
            raise ValueError("val_ratio 必须介于0与1之间")
        if not 0 < self.depth_center_crop_ratio <= 1:
            raise ValueError("depth_center_crop_ratio 必须在(0,1]内")
        if not 0 <= self.depth_percentile_low < self.depth_percentile_high <= 100:
            raise ValueError("深度百分位数必须满足 0 <= low < high <= 100")
        if self.lr <= 0 or self.weight_decay < 0 or self.grad_clip_norm <= 0:
            raise ValueError("lr、grad_clip_norm 必须为正，weight_decay 不能为负")
        if not 0 <= self.cosine_eta_min <= self.lr:
            raise ValueError("cosine_eta_min 必须在[0,lr]内")
        if self.angle_dim != 3:
            raise ValueError("本文的角度向量固定为3维")
        if (not isinstance(self.conv_channels, (list, tuple)) or len(self.conv_channels) != 4
                or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in self.conv_channels)):
            raise ValueError("conv_channels 必须包含四个正整数")
        if any(not 0 <= d < 1 for d in (self.frame_dropout, self.angle_dropout)):
            raise ValueError("dropout 必须在[0,1)内")
        if any(v < 0 for v in (self.lambda_low, self.lambda_high,
                              self.selection_qwk_weight, self.selection_extreme_weight)):
            raise ValueError("损失和模型选择权重不能为负")
        if self.macro_f1_labels not in {"observed", "fixed"}:
            raise ValueError("macro_f1_labels 只能是 observed 或 fixed")
        return self


@dataclass(frozen=True)
class Variant:
    name: str
    title: str
    rgb: bool
    depth: bool
    auxiliary: bool
    gate: bool
    angle: bool


VARIANTS = {
    v.name: v for v in (
        Variant("rgb_only", "RGB-only LowHigh CO-MIL", True, False, True, False, False),
        Variant("depth_only", "Depth-only LowHigh CO-MIL", False, True, True, False, False),
        Variant("rgb_depth", "RGB-depth CO-MIL", True, True, False, False, False),
        Variant("rgb_depth_lowhigh", "RGB-depth LowHigh CO-MIL", True, True, True, False, False),
        Variant("gated_no_angle", "Quality-gated RGB-depth LowHigh, no angle", True, True, True, True, False),
        Variant("final", "Final quality-gated angle-aware CO-MIL", True, True, True, True, True),
        Variant("angle_only", "Angle-only MLP", False, False, True, True, True),
    )
}

UPSTREAM_QC_CONTRACT = {
    "implemented_in_this_file": False,
    "requires_all_five_geometric_quality_checks": True,
    "max_candidate_profiles_per_frame": 9,
    "min_valid_profiles_per_good_angle_frame": 3,
    "apex_relative_position": [0.35, 0.65],
    "side_fit_relative_distance_from_apex": [0.05, 0.40],
    "min_valid_points_per_side": 4,
    "profile_fit": "ordinary_least_squares_linear",
    "default_r2_rejection_threshold": None,
    "episode_angle": "median_of_pooled_valid_profile_angles_from_good_angle_frames",
}

IMPLEMENTATION_CHOICES = {
    "source": "中文修改(2).docx, sections 2.2–2.4",
    "seed": "42 inherited from uploaded script; paper does not specify a seed",
    "validation": "shuffle sorted raw non-test animal IDs with seed+fold, then round(n*0.22)",
    "split_before_gate": "share the same raw animal partition across variants and thresholds",
    "cnn_details": "one Conv3x3(pad1,bias=False)-BN-ReLU-MaxPool2 per stage; GAP-Linear-ReLU to128",
    "shared_mapping": "Linear(288,192) only for final model, as z_s=W*z_f+b",
    "attention": "Linear-Tanh-Linear, hidden128 inherited; paper does not specify scorer details",
    "angle_only": "two hidden Linear-ReLU layers of width64; no visual or attention branch",
    "depth_constant_or_empty": "all-zero closeness when no valid pixels or percentile span is zero",
    "class_weights": "inverse training counts; absent classes weight0; normalize present weights to mean1",
    "macro_f1": "observed preserves original sklearn behavior; configurable fixed full-label alternative",
    "qwk": "fixed complete ordinal labels preserve BCS distances even when classes are absent",
    "undefined_val_qwk": "record NaN; use -1 only when computing model-selection score",
    "std": "sample standard deviation, ddof=1, across folds",
    "ties": "keep the earliest epoch with the maximum validation score",
    "geometry": "upstream QC CSV contract; missing geometric rules are not invented",
}


def require_torch():
    if torch is None:
        raise RuntimeError("训练/预测需要 torch>=2.3。请在你的 PyTorch 环境运行；--check-data 不需要 torch。")


def resolve_path(base, value):
    # 同时支持在 Windows 上使用本地绝对路径，以及 CSV 中的相对反斜杠路径。
    path = Path(str(value).replace("\\", "/")).expanduser()
    return path if path.is_absolute() else Path(base) / path


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def checked_int(value, where):
    number = safe_float(value)
    if not np.isfinite(number) or number != int(number):
        raise ValueError(f"{where}: 需要整数，实际为 {value!r}")
    return int(number)


def canonical_animal_id(value):
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none"}:
        raise ValueError("animal_id 为空；不能进行 Animal-wise 划分")
    # 只处理文章明确说明的 Cow_5 / Cow_5_1 / Cow_5_2 形式。
    match = re.fullmatch(r"cow_(\d+)(?:_\d+)*", value, flags=re.IGNORECASE)
    return f"Cow_{int(match.group(1))}" if match else value


def natural_key(value):
    return tuple((0, int(p)) if p.isdigit() else (1, p.lower())
                 for p in re.split(r"(\d+)", str(value)))


def label5_to_label3(label5):
    if int(label5) not in LABEL5_TO_RAW:
        raise ValueError(f"label5 超出0–4范围: {label5}")
    return int(LABEL5_TO_LABEL3[int(label5)])


def read_qc_table(path, required):
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}；请指定 --base-path 或先生成03阶段QC结果。")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} 缺少列: {missing}")
    if df.empty:
        raise ValueError(f"{path.name} 为空")
    df["bag_id"] = df["bag_id"].str.strip()
    if (df["bag_id"] == "").any():
        raise ValueError(f"{path.name} 存在空 bag_id")
    return df


def build_bags(cfg):
    """读取全部 Episode；使用真实帧计数核验 QC 元数据，再决定预测资格。"""
    base = Path(cfg.base_path).expanduser().resolve()
    frame_path = resolve_path(base, cfg.frame_qc_csv)
    bag_path = resolve_path(base, cfg.bag_qc_csv)
    frame_df = read_qc_table(frame_path, ["bag_id", "is_good_angle_frame", "rgb_path", "depth_npy_path"])
    bag_df = read_qc_table(bag_path, ["bag_id", "good_angle_frame_count", "good_frame_ratio", "bag_good_angle_median"])
    if bag_df["bag_id"].duplicated().any():
        raise ValueError("bag QC CSV 每个 bag_id 必须恰好一行")
    unknown = set(frame_df["bag_id"]) - set(bag_df["bag_id"])
    if unknown:
        raise ValueError(f"帧CSV含有未列入Episode CSV的bag_id: {sorted(unknown)[:5]}")
    for col in (cfg.frame_angles_column, cfg.frame_order_column):
        if col and col not in frame_df.columns:
            raise ValueError(f"帧CSV不存在指定列: {col}")
    groups = {bid: grp.to_dict("records") for bid, grp in frame_df.groupby("bag_id", sort=False)}
    bags = []
    for meta in bag_df.to_dict("records"):
        bid = meta["bag_id"]
        records = groups.get(bid, [])
        if not records:
            raise ValueError(f"Episode {bid} 没有任何帧记录；不能静默丢弃或改变Coverage分母")

        def consistent_value(column, parser):
            values = [r[column] for r in [meta] + records if str(r.get(column, "")).strip()]
            if not values:
                raise ValueError(f"Episode {bid} 在两个CSV中均缺少 {column}")
            parsed = {parser(v) for v in values}
            if len(parsed) != 1:
                raise ValueError(f"Episode {bid} 的 {column} 不一致: {sorted(parsed)}")
            return next(iter(parsed))

        animal = consistent_value("animal_id", canonical_animal_id)
        fold = consistent_value("fold_id", lambda v: checked_int(v, f"{bid}/fold_id"))
        raw = consistent_value("bcs_raw", lambda v: checked_int(v, f"{bid}/bcs_raw"))
        if raw not in range(2, 7):
            raise ValueError(f"Episode {bid}: bcs_raw 应为2–6，实际为{raw}")
        label5, label3 = raw - 2, label5_to_label3(raw - 2)
        for r in [meta] + records:
            for col, expected in (("label5", label5), ("label3", label3)):
                if str(r.get(col, "")).strip() and checked_int(r[col], f"{bid}/{col}") != expected:
                    raise ValueError(f"Episode {bid}: {col} 与BCS映射不一致")

        frames, good_frames, pooled_angles, seen = [], [], [], set()
        for i, r in enumerate(records):
            flag = str(r["is_good_angle_frame"]).strip().lower()
            if flag not in {"0", "1", "0.0", "1.0", "true", "false"}:
                raise ValueError(f"{bid}: is_good_angle_frame 应为0/1或布尔值")
            good = flag in {"1", "1.0", "true"}
            if not r["rgb_path"].strip() or not r["depth_npy_path"].strip():
                raise ValueError(f"{bid}: RGB/Depth路径为空")
            pair = (r["rgb_path"], r["depth_npy_path"])
            if pair in seen:
                raise ValueError(f"{bid}: 重复RGB-D帧对，可能虚增Good-angle-frame数量")
            seen.add(pair)
            frame = {
                "rgb_path": str(resolve_path(base, r["rgb_path"])),
                "depth_npy_path": str(resolve_path(base, r["depth_npy_path"])),
                "frame_key": r.get("frame_key", "") or r["rgb_path"],
                "order": r[cfg.frame_order_column] if cfg.frame_order_column else (r.get("frame_key", "") or r["rgb_path"]),
                "is_good": good,
            }
            frames.append(frame)
            if good:
                good_frames.append(frame)
                if cfg.frame_angles_column:
                    try:
                        angles = np.asarray(json.loads(r[cfg.frame_angles_column]), dtype=float)
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"{bid}, frame {i}: 有效横截面角度应为JSON数组") from exc
                    if angles.ndim != 1 or not 3 <= len(angles) <= 9:
                        raise ValueError(f"{bid}, frame {i}: Good-angle frame应有3–9个有效角度")
                    if not np.isfinite(angles).all() or ((angles < 0) | (angles > 180)).any():
                        raise ValueError(f"{bid}, frame {i}: 横截面角度无效")
                    pooled_angles.extend(angles.tolist())

        if cfg.frame_order_column:
            # 完整数值列按数值排序（支持秒时间戳小数），其他列按自然顺序排序。
            numeric_order = all(np.isfinite(safe_float(f["order"])) for f in frames)
        else:
            numeric_order = False
        order_key = (lambda f: safe_float(f["order"])) if numeric_order else (lambda f: natural_key(f["order"]))
        frames.sort(key=order_key)
        good_frames.sort(key=order_key)
        count = len(good_frames)
        ratio = count / len(frames)
        declared_count = checked_int(meta["good_angle_frame_count"], f"{bid}/good_angle_frame_count")
        declared_ratio = safe_float(meta["good_frame_ratio"])
        if count != declared_count:
            raise ValueError(f"{bid}: CSV声明{declared_count}张Good-angle frames，实际标记{count}张；请修正03阶段索引")
        if not np.isfinite(declared_ratio) or not np.isclose(ratio, declared_ratio, atol=1e-3, rtol=0):
            raise ValueError(f"{bid}: good_frame_ratio={declared_ratio}，但实际{count}/{len(frames)}={ratio:.6f}；帧CSV需包含全部可用帧")
        angle = safe_float(meta["bag_good_angle_median"])
        if cfg.frame_angles_column:
            angle = float(np.median(pooled_angles)) if pooled_angles else float("nan")
        if np.isfinite(angle) and not 0 <= angle <= 180:
            raise ValueError(f"{bid}: Episode角度超出0–180度")
        reasons = []
        if count < cfg.min_good_frames_per_bag:
            reasons.append(f"good_angle_frames<{cfg.min_good_frames_per_bag}")
        if not np.isfinite(angle):
            reasons.append("invalid_episode_angle")
        bags.append({
            "bag_id": bid, "episode_id": meta.get("episode_id", bid), "animal_id": animal,
            "fold_id": fold, "bcs_raw": raw, "label5": label5, "label3": label3,
            "is_low": int(label5 <= 1), "is_high": int(label5 == 4),
            "all_frames": frames, "good_frames": good_frames,
            "frame_count": len(frames), "good_angle_frame_count": count,
            "good_frame_ratio": ratio, "bag_good_angle_median": angle,
            "angle_feat_raw": np.array([angle, ratio, math.log1p(count)], dtype=np.float32),
            "eligible": not reasons, "rejection_reason": ";".join(reasons),
        })
    bags.sort(key=lambda b: natural_key(b["bag_id"]))
    verify_no_fold_leakage(bags)
    return bags


def verify_no_fold_leakage(bags):
    animal_to_folds = defaultdict(set)
    for bag in bags:
        animal_to_folds[bag["animal_id"]].add(bag["fold_id"])
    leaked = {k: sorted(v) for k, v in animal_to_folds.items() if len(v) > 1}
    if leaked:
        raise ValueError(f"检测到动物跨fold泄漏（包含门控前全部Episode）: {leaked}")
    folds = sorted({b["fold_id"] for b in bags})
    if folds not in ([0, 1, 2], [1, 2, 3]):
        raise ValueError(f"需要三折，fold_id应为0/1/2或1/2/3，实际为{folds}")


def split_train_val_by_animal(candidate_bags, val_ratio=0.22, seed=42):
    animals = sorted({b["animal_id"] for b in candidate_bags}, key=natural_key)
    if len(animals) < 2:
        raise ValueError("非测试动物不足2个，无法分离训练与验证集")
    random.Random(seed).shuffle(animals)
    n_val = min(len(animals) - 1, max(1, int(round(len(animals) * val_ratio))))
    val_ids = set(animals[:n_val])
    # 不读取标签，不做分层，不强制向验证集分配稀有类。
    return ([b for b in candidate_bags if b["animal_id"] not in val_ids],
            [b for b in candidate_bags if b["animal_id"] in val_ids])


def make_splits(bags, cfg):
    splits = {}
    for fold in sorted({b["fold_id"] for b in bags}):
        candidates = [b for b in bags if b["fold_id"] != fold]
        train, val = split_train_val_by_animal(candidates, cfg.val_ratio, cfg.seed + fold)
        test = [b for b in bags if b["fold_id"] == fold]
        sets = [set(b["animal_id"] for b in part) for part in (train, val, test)]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise RuntimeError("训练/验证/测试存在动物交叉")
        splits[fold] = {"train": train, "val": val, "test": test}
    return splits


def select_variant_bags(bags, variant):
    return [dict(b, frames=b["good_frames"] if variant.gate else b["all_frames"])
            for b in bags if not variant.gate or b["eligible"]]


def fit_angle_scaler(train_bags):
    if not train_bags:
        raise ValueError("不能在空训练集拟合角度标准化器")
    x = np.stack([b["angle_feat_raw"] for b in train_bags]).astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError("角度模型的训练特征存在NaN/Inf；请检查质量门控")
    mean, std = x.mean(axis=0), x.std(axis=0, ddof=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_angle_scaler(bags, mean, std, use_angle=True):
    out = []
    for bag in bags:
        values = (bag["angle_feat_raw"] - mean) / std if use_angle else np.zeros(3, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{bag['bag_id']}: 标准化后角度特征无效")
        out.append(dict(bag, angle_feat_scaled=values.astype(np.float32)))
    return out


def sample_frame_indices(n, k, train, rng=None):
    if n <= 0 or k <= 0:
        raise ValueError("Episode及采样数量均不能为空")
    rng = rng or random
    if train:
        # 帧数不足时按文章使用有放回随机采样，随机性不用于验证/测试。
        return rng.sample(range(n), k) if n >= k else rng.choices(range(n), k=k)
    if n >= k:
        return np.linspace(0, n - 1, k).astype(int).tolist()
    return list(range(n)) + [n - 1] * (k - n)


def preprocess_depth_array(array, cfg):
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2 or min(arr.shape) == 0:
        raise ValueError(f"深度npy应为非空二维数组，实际shape={arr.shape}")
    h, w = arr.shape
    nh, nw = max(1, int(h * cfg.depth_center_crop_ratio)), max(1, int(w * cfg.depth_center_crop_ratio))
    top, left = (h - nh) // 2, (w - nw) // 2
    arr = arr[top:top + nh, left:left + nw]  # 先裁剪，再统计百分位数。
    valid = np.isfinite(arr) & (arr > 0)
    closeness = np.zeros_like(arr, dtype=np.float32)
    if valid.any():
        lo, hi = np.percentile(arr[valid], [cfg.depth_percentile_low, cfg.depth_percentile_high])
        if hi > lo:
            closeness[valid] = 1.0 - (np.clip(arr[valid], lo, hi) - lo) / (hi - lo)
    # float32的F模式避免原代码uint8量化；无效深度仍为0，不能反转成近处的1。
    img = Image.fromarray(closeness)
    resized = img.resize((cfg.img_size, cfg.img_size), resample=Image.Resampling.BILINEAR)
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0).copy()


def preprocess_rgb_image(image, cfg):
    image = image.convert("RGB").resize((cfg.img_size, cfg.img_size), Image.Resampling.BILINEAR)
    return (np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0).copy()


class RGBDepthAngleBagDataset(Dataset):
    def __init__(self, bags, cfg, variant, train=False):
        self.bags, self.cfg, self.variant, self.train = bags, cfg, variant, train

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        require_torch()
        bag = self.bags[idx]
        # Angle-only没有视觉读取；单模态模型也不读取被消融的另一模态。
        rgb, depth = [], []
        if self.variant.rgb or self.variant.depth:
            indices = sample_frame_indices(len(bag["frames"]), self.cfg.frames_per_bag, self.train)
            cache = {}
            for i in indices:
                if i not in cache:
                    frame = bag["frames"][i]
                    rgb_arr = depth_arr = None
                    if self.variant.rgb:
                        with Image.open(frame["rgb_path"]) as img:
                            rgb_arr = preprocess_rgb_image(img, self.cfg)
                    if self.variant.depth:
                        raw_depth = np.load(frame["depth_npy_path"], allow_pickle=False)
                        depth_arr = preprocess_depth_array(raw_depth, self.cfg)[None, ...]
                    cache[i] = (rgb_arr, depth_arr)
                rgb_arr, depth_arr = cache[i]
                if self.variant.rgb:
                    rgb.append(torch.from_numpy(rgb_arr))
                if self.variant.depth:
                    depth.append(torch.from_numpy(depth_arr))
        return {
            "rgb": torch.stack(rgb) if rgb else torch.empty(0),
            "depth": torch.stack(depth) if depth else torch.empty(0),
            "angle": torch.from_numpy(bag["angle_feat_scaled"]),
            "label5": bag["label5"], "bag_id": bag["bag_id"],
        }


ModuleBase = nn.Module if nn is not None else object


class FrameEncoder(ModuleBase):
    def __init__(self, input_channels, cfg):
        require_torch()
        super().__init__()
        layers, previous = [], input_channels
        for channels in cfg.conv_channels:
            layers += [nn.Conv2d(previous, channels, 3, padding=1, bias=False),
                       nn.BatchNorm2d(channels), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            previous = channels
        self.features = nn.Sequential(*layers)
        self.projection = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Linear(previous, cfg.modality_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.projection(self.features(x))


class AttentionMILPooling(ModuleBase):
    def __init__(self, in_dim=256, hidden_dim=128):
        require_torch()
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, features):
        weights = torch.softmax(self.attn(features).squeeze(-1), dim=1)
        return torch.sum(features * weights.unsqueeze(-1), dim=1), weights


class AngleAwareRGBDepthLowHighCOMIL(ModuleBase):
    def __init__(self, cfg=None, variant=None):
        require_torch()
        super().__init__()
        cfg, variant = cfg or Config(), variant or VARIANTS["final"]
        self.cfg, self.variant = cfg, variant
        self.rgb_encoder = FrameEncoder(3, cfg) if variant.rgb else None
        self.depth_encoder = FrameEncoder(1, cfg) if variant.depth else None
        visual = variant.rgb or variant.depth
        if visual:
            input_dim = cfg.modality_dim * (int(variant.rgb) + int(variant.depth))
            self.frame_fusion = nn.Sequential(nn.Linear(input_dim, cfg.frame_fusion_dim),
                                             nn.ReLU(inplace=True), nn.Dropout(cfg.frame_dropout))
            self.pooling = AttentionMILPooling(cfg.frame_fusion_dim, cfg.attention_hidden_dim)
            fusion_dim = cfg.frame_fusion_dim
            if variant.angle:
                self.angle_encoder = nn.Sequential(
                    nn.Linear(cfg.angle_dim, cfg.angle_hidden_dim), nn.ReLU(inplace=True),
                    nn.Dropout(cfg.angle_dropout), nn.Linear(cfg.angle_hidden_dim, cfg.angle_embedding_dim),
                    nn.ReLU(inplace=True))
                fusion_dim += cfg.angle_embedding_dim
            self.shared_mapping = nn.Linear(fusion_dim, cfg.shared_dim)
            head_dim = cfg.shared_dim
        else:
            if not variant.angle:
                raise ValueError("至少需要一种输入模态")
            self.angle_encoder = nn.Sequential(
                nn.Linear(cfg.angle_dim, cfg.angle_only_hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(cfg.angle_only_hidden_dim, cfg.angle_only_hidden_dim), nn.ReLU(inplace=True))
            head_dim = cfg.angle_only_hidden_dim
        self.bcs_head = nn.Linear(head_dim, NUM_CLASSES)
        self.low_head = nn.Linear(head_dim, 1) if variant.auxiliary else None
        self.high_head = nn.Linear(head_dim, 1) if variant.auxiliary else None

    def forward(self, rgb_bag, depth_bag, angle_feat, include_aux=False):
        weights = None
        if self.variant.rgb or self.variant.depth:
            reference = rgb_bag if self.variant.rgb else depth_bag
            b, k = reference.shape[:2]
            features = []
            if self.variant.rgb:
                features.append(self.rgb_encoder(rgb_bag.flatten(0, 1)))
            if self.variant.depth:
                features.append(self.depth_encoder(depth_bag.flatten(0, 1)))
            frame_features = self.frame_fusion(torch.cat(features, dim=1)).reshape(b, k, -1)
            shared, weights = self.pooling(frame_features)
            if self.variant.angle:
                shared = torch.cat([shared, self.angle_encoder(angle_feat)], dim=1)
            shared = self.shared_mapping(shared)
        else:
            shared = self.angle_encoder(angle_feat)
        output = {"logits": self.bcs_head(shared), "attention": weights}
        # 推理时不调用辅助头，不存在Low/High阈值或覆盖最终BCS的决策。
        if include_aux and self.variant.auxiliary:
            output["low_logit"] = self.low_head(shared).squeeze(-1)
            output["high_logit"] = self.high_head(shared).squeeze(-1)
        return output


def inverse_class_weights(bags):
    counts = np.bincount([b["label5"] for b in bags], minlength=NUM_CLASSES)
    if counts.sum() == 0:
        raise ValueError("训练集为空")
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    present = counts > 0
    weights[present] = 1.0 / counts[present]
    weights[present] /= weights[present].mean()
    return counts, weights


def total_training_loss(output, labels, class_weights, cfg, variant):
    # CE中权重只由当前fold训练Episode计算。无CORAL、focal或SupCon。
    bcs = F.cross_entropy(output["logits"], labels, weight=class_weights)
    zero = bcs.new_zeros(())
    low = F.binary_cross_entropy_with_logits(output["low_logit"], (labels <= 1).float()) if variant.auxiliary else zero
    high = F.binary_cross_entropy_with_logits(output["high_logit"], (labels == 4).float()) if variant.auxiliary else zero
    total = bcs + cfg.lambda_low * low + cfg.lambda_high * high
    return total, {"loss_bcs": bcs, "loss_low": low, "loss_high": high}


def calculate_metrics(y_true, y_pred, cfg):
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("评价集合为空或真值/预测长度不同")
    if ((y_true < 0) | (y_true > 4) | (y_pred < 0) | (y_pred > 4)).any():
        raise ValueError("五级预测必须使用0–4标签")
    true3, pred3 = LABEL5_TO_LABEL3[y_true], LABEL5_TO_LABEL3[y_pred]
    fixed = cfg.macro_f1_labels == "fixed"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        qwk5 = cohen_kappa_score(y_true, y_pred, labels=list(range(5)), weights="quadratic")
        qwk3 = cohen_kappa_score(true3, pred3, labels=list(range(3)), weights="quadratic")
    metrics = {
        "acc5": float(accuracy_score(y_true, y_pred)),
        "macro_f1_5": float(f1_score(y_true, y_pred, labels=list(range(5)) if fixed else None, average="macro", zero_division=0)),
        "weighted_f1_5": float(f1_score(y_true, y_pred, labels=list(range(5)), average="weighted", zero_division=0)),
        "qwk5": float(qwk5), "mae_bcs": float(np.abs(y_true - y_pred).mean()),
        "within_one_acc": float((np.abs(y_true - y_pred) <= 1).mean()),
        "acc3_final": float(accuracy_score(true3, pred3)),
        "macro_f1_3_final": float(f1_score(true3, pred3, labels=list(range(3)) if fixed else None, average="macro", zero_division=0)),
        "weighted_f1_3_final": float(f1_score(true3, pred3, labels=list(range(3)), average="weighted", zero_division=0)),
        "qwk3_final": float(qwk3),
        "lean_to_high_errors": int(((true3 == 0) & (pred3 == 2)).sum()),
        "high_to_lean_errors": int(((true3 == 2) & (pred3 == 0)).sum()),
        "bcs2_to_bcs6_errors": int(((y_true == 0) & (y_pred == 4)).sum()),
        "bcs3_to_bcs6_errors": int(((y_true == 1) & (y_pred == 4)).sum()),
        "bcs6_to_bcs2_or_3_errors": int(((y_true == 4) & (y_pred <= 1)).sum()),
    }
    for label, name in ((0, "lean"), (2, "high")):
        support, predicted = int((true3 == label).sum()), int((pred3 == label).sum())
        tp = int(((true3 == label) & (pred3 == label)).sum())
        metrics[f"{name}_support"] = support
        metrics[f"{name}_recall_final"] = tp / support if support else float("nan")
        metrics[f"{name}_precision_final"] = tp / predicted if predicted else 0.0
    # 保留含义正确的旧mgmt列名；不继续使用ordinal或branch_aux这些旧命名。
    metrics.update(acc3_mgmt=metrics["acc3_final"], macro_f1_3_mgmt=metrics["macro_f1_3_final"],
                   qwk3_mgmt=metrics["qwk3_final"])
    return metrics


def selection_score(metrics, cfg):
    # 文章式(9)(10)：重叠极端错误有意重复计数，不应去重。
    penalty = (2 * metrics["bcs2_to_bcs6_errors"] + metrics["bcs3_to_bcs6_errors"]
               + 2 * metrics["bcs6_to_bcs2_or_3_errors"])
    qwk = metrics["qwk5"] if np.isfinite(metrics["qwk5"]) else -1.0
    return metrics["macro_f1_3_final"] + cfg.selection_qwk_weight * qwk - cfg.selection_extreme_weight * penalty


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
        torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def choose_device(cfg):
    require_torch()
    name = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("当前脚本支持cpu或cuda设备")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了CUDA，但当前PyTorch环境没有可用CUDA设备")
    return device


def make_loader(bags, cfg, variant, device, train, seed):
    dataset = RGBDepthAngleBagDataset(bags, cfg, variant, train)
    kwargs = {"batch_size": cfg.batch_size, "num_workers": cfg.num_workers,
              "pin_memory": device.type == "cuda", "worker_init_fn": seed_worker,
              "persistent_workers": cfg.num_workers > 0,
              "generator": torch.Generator().manual_seed(seed + 10000)}
    if cfg.num_workers > 0:
        kwargs["prefetch_factor"] = cfg.prefetch_factor
    if train:
        counts, _ = inverse_class_weights(bags)
        sample_weights = [1.0 / counts[b["label5"]] for b in bags]
        kwargs["sampler"] = WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.double), len(bags), replacement=True,
            generator=torch.Generator().manual_seed(seed))
    return DataLoader(dataset, shuffle=False, **kwargs)


def forward_batch(model, batch, device, include_aux=False):
    return model(batch["rgb"].to(device, non_blocking=True),
                 batch["depth"].to(device, non_blocking=True),
                 batch["angle"].to(device, non_blocking=True), include_aux=include_aux)


def evaluate(model, loader, cfg, device):
    model.eval()
    predictions, true, predicted = [], [], []
    with torch.no_grad():
        for batch in loader:
            output = forward_batch(model, batch, device, include_aux=False)
            if not torch.isfinite(output["logits"]).all():
                raise FloatingPointError("验证/测试主分类logits出现NaN/Inf")
            probabilities = torch.softmax(output["logits"].float(), dim=1).cpu().numpy()
            pred5 = probabilities.argmax(axis=1)
            attention = output["attention"]
            entropy = ((-(attention * attention.clamp_min(1e-8).log()).sum(dim=1)).cpu().tolist()
                       if attention is not None else [float("nan")] * len(pred5))
            for bid, yt, yp, probs, ent in zip(batch["bag_id"], batch["label5"].tolist(), pred5, probabilities, entropy):
                yp = int(yp)
                true.append(yt)
                predicted.append(yp)
                row = {"bag_id": bid, "true_label5": yt, "pred_label5": yp,
                       "true_bcs": yt + 2, "pred_bcs": yp + 2,
                       "true_label3": label5_to_label3(yt), "pred_label3_final": label5_to_label3(yp),
                       "true_label3_name": CLASS3_NAMES[label5_to_label3(yt)],
                       "pred_label3_final_name": CLASS3_NAMES[label5_to_label3(yp)],
                       "attention_entropy": ent}
                row.update({f"prob_bcs{c + 2}": float(probs[c]) for c in range(5)})
                predictions.append(row)
    return calculate_metrics(true, predicted, cfg), predictions


def train_one_fold(fold, raw_split, cfg, variant, device, out_dir):
    set_seed(cfg.seed + fold, cfg.deterministic)
    parts = {role: select_variant_bags(bags, variant) for role, bags in raw_split.items()}
    if any(not values for values in parts.values()):
        empty = [role for role, values in parts.items() if not values]
        raise ValueError(f"{variant.name}/fold{fold} 门控后{empty}为空。保持既定划分，不能静默重新抽取动物。")
    mean, std = fit_angle_scaler(parts["train"]) if variant.angle else (np.zeros(3, np.float32), np.ones(3, np.float32))
    parts = {role: apply_angle_scaler(values, mean, std, variant.angle) for role, values in parts.items()}
    train_loader = make_loader(parts["train"], cfg, variant, device, True, cfg.seed + fold)
    val_loader = make_loader(parts["val"], cfg, variant, device, False, cfg.seed + fold + 100)
    test_loader = make_loader(parts["test"], cfg, variant, device, False, cfg.seed + fold + 200)
    model = AngleAwareRGBDepthLowHighCOMIL(cfg, variant).to(device)
    counts, weights = inverse_class_weights(parts["train"])
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    if (counts == 0).any():
        print(f"  注意：fold{fold}训练集缺少BCS {np.flatnonzero(counts == 0) + 2}；保留原动物随机划分。", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.cosine_eta_min)
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score, best_state, best_epoch, best_metrics = -float("inf"), None, None, None
    history = []
    print(f"{variant.name} / fold{fold}: train={len(parts['train'])}, val={len(parts['val'])}, "
          f"test={len(parts['test'])}/{len(raw_split['test'])}, device={device}, AMP={use_amp}", flush=True)
    for epoch in range(1, cfg.epochs + 1):
        started, lr_used = time.perf_counter(), optimizer.param_groups[0]["lr"]
        totals = dict(loss_total=0.0, loss_bcs=0.0, loss_low=0.0, loss_high=0.0)
        model.train()
        for batch in train_loader:
            labels = batch["label5"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            with autocast:
                output = forward_batch(model, batch, device, include_aux=True)
                loss, components = total_training_loss(output, labels, class_weights, cfg, variant)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{variant.name}/fold{fold}/epoch{epoch} loss出现NaN/Inf")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # 在unscale之后裁剪，AMP下也保持最大范数5.0的含义。
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            totals["loss_total"] += float(loss.detach())
            for key, value in components.items():
                totals[key] += float(value.detach())
        scheduler.step()
        metrics, _ = evaluate(model, val_loader, cfg, device)
        score = selection_score(metrics, cfg)
        if not np.isfinite(score):
            raise FloatingPointError("验证模型选择得分无效")
        if score > best_score:
            best_score, best_epoch, best_metrics = score, epoch, dict(metrics)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        log = {"epoch": epoch, "lr": lr_used, "val_score": score,
               "val_macro_f1_3": metrics["macro_f1_3_final"], "val_qwk5": metrics["qwk5"],
               "val_extreme_penalty": 2 * metrics["bcs2_to_bcs6_errors"] + metrics["bcs3_to_bcs6_errors"] + 2 * metrics["bcs6_to_bcs2_or_3_errors"],
               "seconds": time.perf_counter() - started}
        log.update({key: value / len(train_loader) for key, value in totals.items()})
        history.append(log)
        write_csv(out_dir / f"fold{fold}_training_history.csv", history)
        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            print(f"  epoch {epoch:03d}/{cfg.epochs}: loss={log['loss_total']:.4f}, "
                  f"val_F1_3={metrics['macro_f1_3_final']:.4f}, val_QWK5={metrics['qwk5']:.4f}, "
                  f"score={score:.4f}, best_epoch={best_epoch}", flush=True)
    # 完整训练全部epoch后才回溯选择，不早停；测试集未参与任何模型选择。
    if best_state is None:
        raise RuntimeError("未产生有效模型快照")
    model.load_state_dict(best_state)
    checkpoint = {
        "format_version": 1, "model_state": best_state, "config": asdict(cfg),
        "variant": variant.name, "best_epoch": best_epoch, "best_val_score": best_score,
        "angle_mean": mean.tolist(), "angle_std": std.tolist(),
        "class_counts": counts.tolist(), "class_weights": weights.tolist(),
        "split_bag_ids": {role: [b["bag_id"] for b in values] for role, values in parts.items()},
        "split_animal_ids_raw": {role: sorted({b["animal_id"] for b in values}) for role, values in raw_split.items()},
        "class_mapping": {"bcs": [2, 3, 4, 5, 6], "management": [0, 0, 1, 1, 2]},
    }
    torch.save(checkpoint, out_dir / f"fold{fold}_best.pth")
    metrics, predictions = evaluate(model, test_loader, cfg, device)
    test_lookup = {b["bag_id"]: b for b in parts["test"]}
    for row in predictions:
        info = test_lookup[row["bag_id"]]
        row.update(fold=fold, variant=variant.name, animal_id=info["animal_id"])
        for key in ("bag_good_angle_median", "good_frame_ratio", "good_angle_frame_count"):
            row[key] = info[key]
    fold_metrics = {"fold": fold, "best_epoch": best_epoch, "best_val_score": best_score,
                    "best_val_qwk": best_metrics["qwk5"],
                    "train_bags": len(parts["train"]), "val_bags": len(parts["val"]),
                    "raw_test_bags": len(raw_split["test"]), "eligible_test_bags": len(parts["test"]),
                    "test_coverage": len(parts["test"]) / len(raw_split["test"]), **metrics}
    write_json(out_dir / f"fold{fold}_training_parameters.json", {
        "angle_mean": mean, "angle_std": std, "class_counts": counts, "class_weights": weights,
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "amp_effective": use_amp, "seed": cfg.seed + fold,
    })
    write_csv(out_dir / f"fold{fold}_predictions.csv", predictions)
    save_confusions(predictions, out_dir, f"fold{fold}")
    return fold_metrics, predictions


def summarize(rows):
    result = []
    keys = [key for key in rows[0] if key != "fold"]
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        result.append({"metric": key, "mean": float(values.mean()) if len(values) else float("nan"),
                       "std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                       "n_valid_folds": len(values)})
    return result


def save_confusions(predictions, out_dir, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    true5 = [r["true_label5"] for r in predictions]
    pred5 = [r["pred_label5"] for r in predictions]
    report = []
    for name, true, pred, names in (
        ("cm5", true5, pred5, CLASS5_NAMES),
        ("cm3", [label5_to_label3(y) for y in true5], [label5_to_label3(y) for y in pred5], CLASS3_NAMES),
    ):
        labels = list(range(len(names)))
        cm = confusion_matrix(true, pred, labels=labels)
        rows = [{"true_class": names[i], **{names[j]: int(cm[i, j]) for j in labels}} for i in labels]
        write_csv(out_dir / f"{prefix}_{name}.csv", rows)
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        ax.imshow(cm, cmap="Blues")
        for i in labels:
            for j in labels:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(xticks=labels, yticks=labels, xticklabels=names, yticklabels=names,
               xlabel="Predicted", ylabel="True", title=f"{prefix}: {'BCS' if name == 'cm5' else 'Management from BCS'}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_{name}.png", dpi=200)
        plt.close(fig)
        report.append(name + "\n" + classification_report(true, pred, labels=labels, target_names=names, zero_division=0))
    (out_dir / f"{prefix}_classification_report.txt").write_text("\n\n".join(report), encoding="utf-8")


def save_data_audit(bags, splits, cfg, out_dir):
    rows = []
    for b in bags:
        row = {k: b[k] for k in ("bag_id", "animal_id", "fold_id", "bcs_raw", "label5", "label3",
                                  "frame_count", "good_angle_frame_count", "good_frame_ratio", "bag_good_angle_median")}
        row.update(eligible_for_prediction=int(b["eligible"]), rejection_reason=b["rejection_reason"])
        rows.append(row)
    write_csv(out_dir / "quality_gate_eligibility_report.csv", rows)
    split_rows, fold_counts = [], []
    for fold, split in splits.items():
        test = split["test"]
        eligible = sum(b["eligible"] for b in test)
        fold_counts.append({"fold": fold, "raw_test_bags": len(test), "eligible_test_bags": eligible,
                            "coverage": eligible / len(test), "animal_groups": len({b["animal_id"] for b in test})})
        for role, values in split.items():
            for b in values:
                split_rows.append({"test_fold": fold, "split": role, "bag_id": b["bag_id"],
                                   "animal_id": b["animal_id"], "bcs_raw": b["bcs_raw"],
                                   "eligible_for_prediction": int(b["eligible"])})
    write_csv(out_dir / "split_manifest.csv", split_rows)
    write_csv(out_dir / "coverage_by_fold.csv", fold_counts)
    coverage = [row["coverage"] for row in fold_counts]
    summary = {
        "raw_episodes": len(bags), "animal_groups": len({b["animal_id"] for b in bags}),
        "all_frames": sum(b["frame_count"] for b in bags),
        "good_angle_frames": sum(b["good_angle_frame_count"] for b in bags),
        "eligible_episodes": sum(b["eligible"] for b in bags),
        "coverage_mean_across_folds": float(np.mean(coverage)),
        "coverage_std_across_folds": float(np.std(coverage, ddof=1)),
        "coverage_pooled": sum(b["eligible"] for b in bags) / len(bags),
        "episode_bcs_counts": dict(Counter(b["bcs_raw"] for b in bags)),
        "upstream_geometry_revalidated": False,
        "episode_angle_recomputed_from_profile_arrays": bool(cfg.frame_angles_column),
        "note": "Coverage按三折均值和总体比例分别报告；不把文章中的151/111或准确率写入计算。",
    }
    write_json(out_dir / "data_audit.json", summary)
    print(f"数据检查：{summary['raw_episodes']} episodes，{summary['animal_groups']} animals，"
          f"{summary['all_frames']} frames；合格{summary['eligible_episodes']}；"
          f"平均Coverage={summary['coverage_mean_across_folds']:.4f}", flush=True)
    return summary


def check_image_paths(bags, variants):
    needed = set()
    for variant in variants:
        for bag in select_variant_bags(bags, variant):
            for frame in bag["frames"]:
                if variant.rgb:
                    needed.add(frame["rgb_path"])
                if variant.depth:
                    needed.add(frame["depth_npy_path"])
    missing = sorted(path for path in needed if not Path(path).is_file())
    if missing:
        raise FileNotFoundError(f"找不到{len(missing)}个所需图像/npy文件，例如 {missing[:5]}。请修正base_path或CSV路径。")


def run_variant(bags, splits, cfg, variant, device, run_dir):
    out_dir = run_dir / variant.name
    out_dir.mkdir(parents=True, exist_ok=False)
    fold_rows, all_predictions = [], []
    for fold, split in splits.items():
        fold_row, predictions = train_one_fold(fold, split, cfg, variant, device, out_dir)
        fold_rows.append(fold_row)
        all_predictions.extend(predictions)
        write_csv(out_dir / "fold_metrics_partial.csv", fold_rows)
    expected = {b["bag_id"] for b in bags if not variant.gate or b["eligible"]}
    actual = [r["bag_id"] for r in all_predictions]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError("OOF预测必须对每个应预测Episode恰好产生一次结果")
    write_csv(out_dir / "fold_metrics.csv", fold_rows)
    write_csv(out_dir / "mean_std.csv", summarize(fold_rows))
    write_csv(out_dir / "oof_predictions.csv", all_predictions)
    by_id = {r["bag_id"]: r for r in all_predictions}
    status = []
    for b in bags:
        row = {"bag_id": b["bag_id"], "animal_id": b["animal_id"], "fold": b["fold_id"],
               "status": "Predicted" if b["bag_id"] in by_id else "No reliable prediction",
               "rejection_reason": b["rejection_reason"] if b["bag_id"] not in by_id else ""}
        if b["bag_id"] in by_id:
            row.update(by_id[b["bag_id"]])
        # 拒绝的Episode不写入BCS/概率；在CSV中保留空值。
        status.append(row)
    write_csv(out_dir / "all_episode_prediction_status.csv", status)
    save_confusions(all_predictions, out_dir, "oof")
    pooled = calculate_metrics([r["true_label5"] for r in all_predictions], [r["pred_label5"] for r in all_predictions], cfg)
    write_json(out_dir / "oof_pooled_metrics.json", {"aggregation": "pooled, not fold_mean", **pooled})
    print(f"{variant.name} 三折完成：Acc5={np.mean([r['acc5'] for r in fold_rows]):.4f}", flush=True)
    return fold_rows, all_predictions


def save_matched_subset(results, bags, cfg, out_dir):
    if not {"rgb_depth_lowhigh", "gated_no_angle"}.issubset(results):
        return
    eligible_ids = {b["bag_id"] for b in bags if b["eligible"]}
    output, predictions = [], []
    for name in ("rgb_depth_lowhigh", "gated_no_angle"):
        _, all_rows = results[name]
        rows = [r for r in all_rows if r["bag_id"] in eligible_ids]
        if {r["bag_id"] for r in rows} != eligible_ids:
            raise RuntimeError("Matched-subset模型的Episode集合不一致")
        by_fold = []
        for fold in sorted({b["fold_id"] for b in bags}):
            subset = [r for r in rows if r["fold"] == fold]
            metrics = calculate_metrics([r["true_label5"] for r in subset], [r["pred_label5"] for r in subset], cfg)
            by_fold.append({"fold": fold, **metrics})
        for row in summarize(by_fold):
            output.append({"variant": name, **row})
        write_csv(out_dir / f"matched_{name}_fold_metrics.csv", by_fold)
        predictions.extend(rows)
    write_csv(out_dir / "matched_subset_mean_std.csv", output)
    write_csv(out_dir / "matched_subset_predictions.csv", predictions)


def save_run_metadata(cfg, run_dir):
    write_json(run_dir / "effective_config.json", asdict(cfg))
    base = Path(cfg.base_path).expanduser().resolve()
    source_hashes = {}
    for label, path in (("code", Path(__file__)), ("frame_qc_csv", resolve_path(base, cfg.frame_qc_csv)),
                        ("bag_qc_csv", resolve_path(base, cfg.bag_qc_csv))):
        source_hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(run_dir / "run_provenance.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "torch": str(torch.__version__) if torch is not None else None,
        "numpy": np.__version__, "pandas": pd.__version__, "source_sha256": source_hashes,
        "implementation_choices": IMPLEMENTATION_CHOICES, "upstream_qc_contract": UPSTREAM_QC_CONTRACT,
    })


def self_test():
    """无真实数据的逻辑与可选PyTorch前后向测试；不声称复现文章实验结果。"""
    cfg = Config().validate()
    passed = []
    assert sample_frame_indices(3, 6, False) == [0, 1, 2, 2, 2, 2]
    assert sample_frame_indices(13, 4, False) == [0, 4, 8, 12]
    assert sample_frame_indices(3, 12, True, random.Random(5)) == sample_frame_indices(3, 12, True, random.Random(5))
    passed.append("deterministic evaluation padding and seeded training sampling")
    animals = [{"animal_id": f"Cow_{i}", "label5": i % 5} for i in range(20)]
    train, val = split_train_val_by_animal(animals)
    assert len(val) == 4 and not ({b["animal_id"] for b in train} & {b["animal_id"] for b in val})
    changed = [dict(b, label5=4 - b["label5"]) for b in animals]
    _, val2 = split_train_val_by_animal(changed)
    assert [b["animal_id"] for b in val] == [b["animal_id"] for b in val2]
    assert canonical_animal_id("Cow_5_2") == canonical_animal_id("Cow_5")
    passed.append("animal isolation and label-independent validation split")
    small = replace(cfg, img_size=4, depth_center_crop_ratio=1.0)
    arr = np.tile([10.0, 20.0, 30.0, 0.0], (4, 1)).astype(np.float32)
    processed = preprocess_depth_array(arr, small)
    assert processed[0, 0] > processed[0, 1] > processed[0, 2]
    assert np.all(processed[:, 3] == 0)
    dirty = np.array([[np.nan, np.inf], [-1.0, 0.0]], np.float32)
    assert not preprocess_depth_array(dirty, small).any()
    assert not preprocess_depth_array(np.ones((4, 4)), small).any()
    crop_cfg = replace(small, depth_center_crop_ratio=0.5)
    border = np.full((8, 8), 1e9, np.float32)
    border[2:6, 2:6] = arr
    assert np.allclose(preprocess_depth_array(border, crop_cfg), processed)
    passed.append("crop-before-percentile depth, closeness direction, invalid pixels")
    scaler_bags = [{"angle_feat_raw": np.array([x, 0.5, 2.0], np.float32)} for x in [160, 170]]
    mean, std = fit_angle_scaler(scaler_bags)
    assert np.allclose(mean, [165, 0.5, 2]) and np.allclose(std, [5, 1, 1])
    assert np.allclose(apply_angle_scaler(scaler_bags, mean, std)[0]["angle_feat_scaled"], [-1, 0, 0])
    passed.append("train-only scaling and zero-variance features")
    labels, predictions = [0, 1, 2, 3, 4], [4, 4, 2, 3, 0]
    metrics = calculate_metrics(labels, predictions, cfg)
    expected = metrics["macro_f1_3_final"] + 0.35 * metrics["qwk5"] - 0.04 * 5
    assert np.isclose(selection_score(metrics, cfg), expected)
    assert metrics["lean_to_high_errors"] == 2 and metrics["bcs2_to_bcs6_errors"] == 1
    passed.append("BCS mapping, overlapping extreme-error penalty and selection formula")
    # 覆盖率必须区别于池化比例：文章的42/51、36/50、33/50。
    assert np.isclose(np.mean([42/51, 36/50, 33/50]), 0.7345098039215686)
    assert not np.isclose(np.mean([42/51, 36/50, 33/50]), 111/151)
    passed.append("fold-average versus pooled coverage")
    if torch is None:
        print(json.dumps({"passed": passed, "torch_tests": "SKIPPED: PyTorch not installed"}, indent=2))
        return
    torch.set_num_threads(1)
    tiny = replace(cfg, img_size=32, frames_per_bag=2, num_workers=0, amp=False)
    set_seed(42)
    for variant in VARIANTS.values():
        model = AngleAwareRGBDepthLowHighCOMIL(tiny, variant)
        rgb = torch.randn(2, 2, 3, 32, 32) if variant.rgb else torch.empty(2, 0)
        depth = torch.randn(2, 2, 1, 32, 32) if variant.depth else torch.empty(2, 0)
        angle = torch.randn(2, 3)
        out = model(rgb, depth, angle, include_aux=True)
        assert out["logits"].shape == (2, 5)
        if out["attention"] is not None:
            assert torch.allclose(out["attention"].sum(1), torch.ones(2))
        loss, _ = total_training_loss(out, torch.tensor([0, 4]), torch.ones(5), tiny, variant)
        loss.backward()
        assert torch.isfinite(loss) and model.bcs_head.weight.grad is not None
        model.eval()
        with torch.no_grad():
            before = model(rgb, depth, angle)["logits"].clone()
            if variant.auxiliary:
                model.low_head.weight.fill_(1e6)
                model.high_head.weight.fill_(-1e6)
            after = model(rgb, depth, angle)["logits"]
        assert torch.equal(before, after)
    passed.append("all seven variants forward/backward; attention normalization; auxiliary heads cannot affect inference")
    print(json.dumps({"passed": passed, "torch_tests": "PASSED", "torch_version": str(torch.__version__)}, indent=2))


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    preliminary, _ = config_parser.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="可选JSON配置；命令行参数优先")
    parser.add_argument("--variants", nargs="+", default=["final"], choices=list(VARIANTS) + ["all"])
    parser.add_argument("--check-data", action="store_true", help="仅检查输入、门控与动物划分，不训练")
    parser.add_argument("--print-config", action="store_true", help="打印默认/合并后配置，不读数据")
    parser.add_argument("--self-test", action="store_true", help="使用合成数据验证逻辑，不使用真实数据")
    parser.add_argument("--sensitivity", nargs="+", type=int, help="如3 4 5 6；固定原动物划分，每个阈值重新训练最终模型")
    defaults = asdict(Config())
    if preliminary.config:
        loaded = json.loads(preliminary.config.read_text(encoding="utf-8-sig"))
        if not isinstance(loaded, dict):
            parser.error("配置文件必须为JSON对象")
        unknown = set(loaded) - set(defaults)
        if unknown:
            parser.error(f"配置包含未知参数: {sorted(unknown)}")
        defaults.update(loaded)
    for field in fields(Config):
        option = "--" + field.name.replace("_", "-")
        value = getattr(Config(), field.name)
        if isinstance(value, bool):
            parser.add_argument(option, action=argparse.BooleanOptionalAction, default=defaults[field.name])
        elif isinstance(value, tuple):
            parser.add_argument(option, type=int, nargs=4, default=defaults[field.name])
        else:
            parser.add_argument(option, type=type(value), default=defaults[field.name])
    args = parser.parse_args(argv)
    cfg = Config(**{field.name: getattr(args, field.name) for field in fields(Config)}).validate()
    if args.sensitivity and (args.variants != ["final"] or args.check_data):
        parser.error("--sensitivity 仅用于final完整训练，不与--check-data组合")
    if args.sensitivity and (min(args.sensitivity) < 1 or len(set(args.sensitivity)) != len(args.sensitivity)):
        parser.error("敏感性阈值必须为不同的正整数")
    return args, cfg


def main(argv=None):
    args, cfg = parse_args(argv)
    if args.print_config:
        print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
        return
    if args.self_test:
        self_test()
        return
    variants = list(VARIANTS.values()) if "all" in args.variants else [VARIANTS[n] for n in dict.fromkeys(args.variants)]
    bags = build_bags(cfg)
    splits = make_splits(bags, cfg)
    # 在产生任何训练结果前检查每个实验的非空划分，避免跑到后面才发现无验证集。
    thresholds = args.sensitivity or [cfg.min_good_frames_per_bag]
    for threshold in thresholds:
        for fold, parts in splits.items():
            for role, values in parts.items():
                for variant in variants:
                    if variant.gate and not any(b["good_angle_frame_count"] >= threshold and np.isfinite(b["bag_good_angle_median"]) for b in values):
                        raise ValueError(f"threshold={threshold}, fold={fold}, {role}在门控后为空；无法按当前划分完成实验")
    if cfg.check_files:
        path_check_bags = [dict(b, eligible=b["good_angle_frame_count"] >= min(thresholds)
                               and np.isfinite(b["bag_good_angle_median"])) for b in bags]
        check_image_paths(path_check_bags, variants)
    device = choose_device(cfg) if not args.check_data else None
    root = resolve_path(Path(cfg.base_path).expanduser().resolve(), cfg.out_dir)
    run_dir = root / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    save_run_metadata(cfg, run_dir)
    save_data_audit(bags, splits, cfg, run_dir)
    if args.check_data:
        print(f"检查完成：{run_dir}")
        return
    if args.sensitivity:
        sensitivity_rows = []
        # 固定动物划分；不同阈值分别改变训练/验证/测试资格并重新训练。
        for threshold in args.sensitivity:
            current = replace(cfg, min_good_frames_per_bag=threshold)
            threshold_bags = []
            for b in bags:
                reasons = []
                if b["good_angle_frame_count"] < threshold:
                    reasons.append(f"good_angle_frames<{threshold}")
                if not np.isfinite(b["bag_good_angle_median"]):
                    reasons.append("invalid_episode_angle")
                threshold_bags.append(dict(b, eligible=not reasons, rejection_reason=";".join(reasons)))
            by_id = {b["bag_id"]: b for b in threshold_bags}
            current_splits = {fold: {role: [by_id[b["bag_id"]] for b in values] for role, values in parts.items()}
                              for fold, parts in splits.items()}
            directory = run_dir / f"tau_{threshold}"
            directory.mkdir()
            save_run_metadata(current, directory)
            audit = save_data_audit(threshold_bags, current_splits, current, directory)
            fold_rows, _ = run_variant(threshold_bags, current_splits, current, VARIANTS["final"], device, directory)
            row = {"tau_frame": threshold, "eligible_episodes": audit["eligible_episodes"],
                   "coverage_mean": audit["coverage_mean_across_folds"], "coverage_pooled": audit["coverage_pooled"]}
            for metric in ("acc5", "macro_f1_5", "qwk5", "within_one_acc", "macro_f1_3_final"):
                values = [r[metric] for r in fold_rows]
                row.update({metric + "_mean": float(np.mean(values)), metric + "_std": float(np.std(values, ddof=1))})
            sensitivity_rows.append(row)
            write_csv(run_dir / "sensitivity_retrained_mean_std.csv", sensitivity_rows)
    else:
        results, comparison = {}, []
        for variant in variants:
            results[variant.name] = run_variant(bags, splits, cfg, variant, device, run_dir)
            for row in summarize(results[variant.name][0]):
                comparison.append({"variant": variant.name, **row})
            write_csv(run_dir / "ablation_mean_std.csv", comparison)
        save_matched_subset(results, bags, cfg, run_dir)
    print(f"全部完成，结果目录：{run_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError, FloatingPointError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
