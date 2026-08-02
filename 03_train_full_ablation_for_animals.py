from pathlib import Path
from collections import defaultdict, Counter
import argparse
import json
import math
import random
import time
import warnings

import numpy as np
import pandas as pd

from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        cohen_kappa_score,
        confusion_matrix,
        classification_report,
        precision_recall_fscore_support,
        average_precision_score,
        roc_auc_score,
    )
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False



BASE_PATH = Path(r"../dataset")

FRAME_CSV = BASE_PATH / "03A_animalwise_3fold.csv"

QC_FRAME_CSV = (
    BASE_PATH
    / "08e_quality_controlled_dorsal_angle"
    / "08e_qc_frame_dorsal_angle.csv"
)

QC_BAG_CSV = (
    BASE_PATH
    / "08e_quality_controlled_dorsal_angle"
    / "08e_qc_bag_dorsal_angle.csv"
)

OUT_DIR = BASE_PATH / "12_full_ablation_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BCS_CLASSES = [2, 3, 4, 5, 6]
LABEL5_TO_BCS = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6}
BCS_TO_LABEL5 = {v: k for k, v in LABEL5_TO_BCS.items()}
MGMT_NAMES = ["Lean", "Ideal", "High"]

DEFAULT_VARIANTS = [
    "rgb_only_lowhigh",
    "depth_only_lowhigh",
    "rgb_depth_comil",
    "rgb_depth_lowhigh",
    "qg_rgb_depth_lowhigh_no_angle",
    "qg_rgb_depth_lowhigh_with_angle",
    "angle_only_mlp",
]



def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bcs_to_label3(bcs):
    bcs = int(bcs)
    if bcs in [2, 3]:
        return 0
    if bcs in [4, 5]:
        return 1
    return 2


def label5_to_label3(label5):
    return bcs_to_label3(LABEL5_TO_BCS[int(label5)])


def safe_read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def ensure_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}\nAvailable: {list(df.columns)}")


def find_col(df, candidates, required=True, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Cannot find any of columns {candidates}. Available: {list(df.columns)}")
    return default


def make_out_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def fmt(x, nd=4):
    if x is None or pd.isna(x):
        return "nan"
    return f"{float(x):.{nd}f}"


def load_frame_index(quality_gated=False, min_good_frames=3, require_angle=False):

    if quality_gated:
        frame_df = safe_read_csv(QC_FRAME_CSV)
        ensure_columns(
            frame_df,
            ["bag_id", "bcs_raw", "label5", "label3", "fold_id",
             "rgb_path", "depth_npy_path", "is_good_angle_frame"],
            "QC frame CSV",
        )

        frame_df = frame_df[frame_df["is_good_angle_frame"].astype(int) == 1].copy()

        qc_bag = safe_read_csv(QC_BAG_CSV)
        ensure_columns(
            qc_bag,
            ["bag_id", "bag_good_angle_median", "good_frame_ratio", "good_angle_frame_count"],
            "QC bag CSV",
        )

        eligible = qc_bag[
            (qc_bag["bag_good_angle_median"].notna())
            & (qc_bag["good_angle_frame_count"].fillna(0).astype(float) >= min_good_frames)
        ].copy()

        frame_df = frame_df[frame_df["bag_id"].astype(str).isin(eligible["bag_id"].astype(str))].copy()

        # Merge bag-level QC features.
        merge_cols = [
            "bag_id",
            "bag_good_angle_median",
            "bag_good_angle_mean",
            "bag_good_angle_std",
            "good_frame_ratio",
            "good_angle_frame_count",
            "sampled_frame_count",
        ]
        merge_cols = [c for c in merge_cols if c in eligible.columns]
        frame_df = frame_df.merge(eligible[merge_cols], on="bag_id", how="left")

    else:
        frame_df = safe_read_csv(FRAME_CSV)
        ensure_columns(
            frame_df,
            ["bag_id", "bcs_raw", "label5", "label3", "fold_id", "rgb_path", "depth_npy_path"],
            "Frame CSV",
        )

        # Add dummy QC features for unified code.
        frame_df["bag_good_angle_median"] = np.nan
        frame_df["bag_good_angle_mean"] = np.nan
        frame_df["bag_good_angle_std"] = np.nan
        frame_df["good_frame_ratio"] = 1.0
        frame_df["good_angle_frame_count"] = frame_df.groupby("bag_id")["bag_id"].transform("size")
        frame_df["sampled_frame_count"] = frame_df.groupby("bag_id")["bag_id"].transform("size")

    if len(frame_df) == 0:
        raise RuntimeError("No frames remained after filtering. Check quality-gated settings.")

    agg_dict = {
        "bcs_raw": "first",
        "label5": "first",
        "label3": "first",
        "fold_id": "first",
        "rgb_path": list,
        "depth_npy_path": list,
        "good_frame_ratio": "first",
        "good_angle_frame_count": "first",
        "sampled_frame_count": "first",
        "bag_good_angle_median": "first",
        "bag_good_angle_mean": "first",
        "bag_good_angle_std": "first",
    }

    if "animal_id" in frame_df.columns:
        agg_dict["animal_id"] = "first"
    else:
        frame_df["animal_id"] = frame_df["bag_id"]
        agg_dict["animal_id"] = "first"

    bag_df = (
        frame_df.groupby("bag_id")
        .agg(agg_dict)
        .reset_index()
    )

    if require_angle:
        bag_df = bag_df[bag_df["bag_good_angle_median"].notna()].copy()

    return frame_df, bag_df


def compute_angle_features(bag_df):

    x = np.zeros((len(bag_df), 3), dtype=np.float32)

    x[:, 0] = bag_df["bag_good_angle_median"].fillna(bag_df["bag_good_angle_median"].median()).astype(float).values
    x[:, 1] = bag_df["good_frame_ratio"].fillna(0.0).astype(float).values
    x[:, 2] = np.log1p(bag_df["good_angle_frame_count"].fillna(0).astype(float).values)

    return x


def fit_angle_scaler(train_bags):
    feats = compute_angle_features(train_bags)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_angle_scaler(bag_df, mean, std):
    feats = compute_angle_features(bag_df)
    return ((feats - mean) / std).astype(np.float32)



def load_rgb(path, img_size):
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Simple normalization, no heavy preprocessing.
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.float32)


def center_crop_array(arr, ratio=0.70):
    if ratio >= 0.999:
        return arr
    h, w = arr.shape[:2]
    nh = max(1, int(h * ratio))
    nw = max(1, int(w * ratio))
    y1 = max(0, (h - nh) // 2)
    x1 = max(0, (w - nw) // 2)
    return arr[y1:y1 + nh, x1:x1 + nw]


def load_depth(path, img_size, crop_ratio=0.70):
    arr = np.load(path).astype(np.float32)
    arr = center_crop_array(arr, crop_ratio)
    valid = arr > 0

    if valid.sum() > 50:
        vals = arr[valid]
        lo = np.percentile(vals, 1)
        hi = np.percentile(vals, 99)
        if hi <= lo:
            hi = lo + 1.0
        arr = np.clip(arr, lo, hi)
        # smaller depth = closer/higher, invert to closeness
        arr = 1.0 - (arr - lo) / (hi - lo)
        arr[~valid] = 0.0
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr[None, :, :]
    return arr.astype(np.float32)



class BagDataset(Dataset):
    def __init__(
        self,
        bag_df,
        modality="rgb_depth",
        use_angle=False,
        angle_mean=None,
        angle_std=None,
        img_size=160,
        frames_per_bag=12,
        train=True,
        depth_crop_ratio=0.70,
    ):
        self.bag_df = bag_df.reset_index(drop=True).copy()
        self.modality = modality
        self.use_angle = use_angle
        self.angle_mean = angle_mean
        self.angle_std = angle_std
        self.img_size = img_size
        self.frames_per_bag = frames_per_bag
        self.train = train
        self.depth_crop_ratio = depth_crop_ratio

        if use_angle:
            if angle_mean is None or angle_std is None:
                raise ValueError("angle_mean and angle_std are required when use_angle=True")
            self.angle_feats = apply_angle_scaler(self.bag_df, angle_mean, angle_std)
        else:
            self.angle_feats = np.zeros((len(self.bag_df), 3), dtype=np.float32)

    def __len__(self):
        return len(self.bag_df)

    def choose_indices(self, n):
        k = self.frames_per_bag
        if n <= 0:
            return []
        if self.train:
            if n >= k:
                return np.random.choice(n, size=k, replace=False).tolist()
            return np.random.choice(n, size=k, replace=True).tolist()
        else:
            if n >= k:
                return np.linspace(0, n - 1, k).astype(int).tolist()
            idx = list(range(n))
            while len(idx) < k:
                idx.append(idx[-1])
            return idx[:k]

    def __getitem__(self, idx):
        row = self.bag_df.iloc[idx]

        rgb_paths = row["rgb_path"]
        depth_paths = row["depth_npy_path"]

        if isinstance(rgb_paths, str):
            # Should not happen after groupby(list), but keep robust.
            rgb_paths = [rgb_paths]
        if isinstance(depth_paths, str):
            depth_paths = [depth_paths]

        n = min(len(rgb_paths), len(depth_paths))
        indices = self.choose_indices(n)

        rgb_list = []
        depth_list = []

        if self.modality in ["rgb", "rgb_depth"]:
            for i in indices:
                rgb_list.append(load_rgb(rgb_paths[i], self.img_size))
            rgb = np.stack(rgb_list, axis=0)
        else:
            rgb = np.zeros((len(indices), 3, self.img_size, self.img_size), dtype=np.float32)

        if self.modality in ["depth", "rgb_depth"]:
            for i in indices:
                depth_list.append(load_depth(depth_paths[i], self.img_size, self.depth_crop_ratio))
            depth = np.stack(depth_list, axis=0)
        else:
            depth = np.zeros((len(indices), 1, self.img_size, self.img_size), dtype=np.float32)

        sample = {
            "bag_id": str(row["bag_id"]),
            "rgb": torch.from_numpy(rgb),
            "depth": torch.from_numpy(depth),
            "angle": torch.from_numpy(self.angle_feats[idx]),
            "label5": torch.tensor(int(row["label5"]), dtype=torch.long),
            "label3": torch.tensor(int(row["label3"]), dtype=torch.long),
            "bcs_raw": torch.tensor(int(row["bcs_raw"]), dtype=torch.long),
            "fold_id": torch.tensor(int(row["fold_id"]), dtype=torch.long),
            "good_frame_ratio": torch.tensor(float(row.get("good_frame_ratio", np.nan)), dtype=torch.float32),
            "good_angle_frame_count": torch.tensor(float(row.get("good_angle_frame_count", np.nan)), dtype=torch.float32),
        }
        return sample


def collate_bags(batch):
    out = {}
    out["bag_id"] = [b["bag_id"] for b in batch]
    for k in ["rgb", "depth", "angle", "label5", "label3", "bcs_raw", "fold_id", "good_frame_ratio", "good_angle_frame_count"]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out



class SmallCNN(nn.Module):
    def __init__(self, in_ch, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 192, 3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
        )

    def forward(self, x):
        return self.fc(self.net(x))


class AttentionMIL(nn.Module):
    def __init__(self, in_dim, attn_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, frame_feats):
        # frame_feats: [B, T, D]
        scores = self.attn(frame_feats).squeeze(-1)  # [B, T]
        weights = torch.softmax(scores, dim=1)
        bag_feat = torch.sum(frame_feats * weights.unsqueeze(-1), dim=1)
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1)
        return bag_feat, weights, entropy


class ImageMILModel(nn.Module):
    def __init__(
        self,
        modality="rgb_depth",
        use_lowhigh=True,
        use_angle=False,
        feat_dim=128,
        fusion_dim=256,
        angle_dim=32,
    ):
        super().__init__()
        self.modality = modality
        self.use_lowhigh = use_lowhigh
        self.use_angle = use_angle

        if modality in ["rgb", "rgb_depth"]:
            self.rgb_encoder = SmallCNN(3, feat_dim)
        else:
            self.rgb_encoder = None

        if modality in ["depth", "rgb_depth"]:
            self.depth_encoder = SmallCNN(1, feat_dim)
        else:
            self.depth_encoder = None

        if modality == "rgb_depth":
            frame_in = feat_dim * 2
        else:
            frame_in = feat_dim

        self.frame_fusion = nn.Sequential(
            nn.Linear(frame_in, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.mil = AttentionMIL(fusion_dim, attn_dim=128)

        if use_angle:
            self.angle_encoder = nn.Sequential(
                nn.Linear(3, angle_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(angle_dim, angle_dim),
                nn.ReLU(inplace=True),
            )
            head_in = fusion_dim + angle_dim
        else:
            self.angle_encoder = None
            head_in = fusion_dim

        self.head_shared = nn.Sequential(
            nn.Linear(head_in, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

        self.ordinal_head = nn.Linear(192, 5)
        self.low_head = nn.Linear(192, 1)
        self.high_head = nn.Linear(192, 1)

    def forward(self, rgb, depth, angle):
        # rgb: [B,T,3,H,W], depth: [B,T,1,H,W]
        B, T = rgb.shape[:2]
        feats = []

        if self.modality in ["rgb", "rgb_depth"]:
            x = rgb.reshape(B * T, rgb.shape[2], rgb.shape[3], rgb.shape[4])
            f = self.rgb_encoder(x).reshape(B, T, -1)
            feats.append(f)

        if self.modality in ["depth", "rgb_depth"]:
            x = depth.reshape(B * T, depth.shape[2], depth.shape[3], depth.shape[4])
            f = self.depth_encoder(x).reshape(B, T, -1)
            feats.append(f)

        frame_feat = torch.cat(feats, dim=-1)
        frame_feat = self.frame_fusion(frame_feat)
        bag_feat, attn_w, attn_entropy = self.mil(frame_feat)

        if self.use_angle:
            angle_feat = self.angle_encoder(angle)
            bag_feat = torch.cat([bag_feat, angle_feat], dim=-1)

        shared = self.head_shared(bag_feat)
        logits5 = self.ordinal_head(shared)
        low_logit = self.low_head(shared).squeeze(-1)
        high_logit = self.high_head(shared).squeeze(-1)

        return {
            "logits5": logits5,
            "low_logit": low_logit,
            "high_logit": high_logit,
            "attn_entropy": attn_entropy,
        }


class AngleOnlyMLP(nn.Module):
    def __init__(self, use_lowhigh=True):
        super().__init__()
        self.use_lowhigh = use_lowhigh
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
        )
        self.ordinal_head = nn.Linear(64, 5)
        self.low_head = nn.Linear(64, 1)
        self.high_head = nn.Linear(64, 1)

    def forward(self, rgb, depth, angle):
        shared = self.net(angle)
        return {
            "logits5": self.ordinal_head(shared),
            "low_logit": self.low_head(shared).squeeze(-1),
            "high_logit": self.high_head(shared).squeeze(-1),
            "attn_entropy": torch.zeros(angle.shape[0], device=angle.device),
        }



def class_weights_from_bags(train_bags, device):
    labels = train_bags["label5"].astype(int).values
    counts = np.bincount(labels, minlength=5).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_sampler(train_bags):
    labels = train_bags["label5"].astype(int).values
    counts = Counter(labels.tolist())
    weights = [1.0 / counts[int(y)] for y in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def compute_loss(outputs, label5, label3, ce_weight=None, use_lowhigh=True, low_w=0.45, high_w=0.45):
    ce = F.cross_entropy(outputs["logits5"], label5, weight=ce_weight)

    if not use_lowhigh:
        return ce, {"ce": ce.item(), "low": 0.0, "high": 0.0}

    low_target = (label3 == 0).float()
    high_target = (label3 == 2).float()

    low_loss = F.binary_cross_entropy_with_logits(outputs["low_logit"], low_target)
    high_loss = F.binary_cross_entropy_with_logits(outputs["high_logit"], high_target)

    loss = ce + low_w * low_loss + high_w * high_loss

    return loss, {
        "ce": ce.item(),
        "low": low_loss.item(),
        "high": high_loss.item(),
    }


def qwk_score(y_true, y_pred):
    if HAS_SKLEARN:
        try:
            return cohen_kappa_score(y_true, y_pred, weights="quadratic")
        except Exception:
            return np.nan
    return np.nan


def safe_auc(y_true_binary, score, kind="pr"):
    if not HAS_SKLEARN:
        return np.nan
    y_true_binary = np.asarray(y_true_binary).astype(int)
    score = np.asarray(score).astype(float)
    if len(np.unique(y_true_binary)) < 2:
        return np.nan
    try:
        if kind == "pr":
            return average_precision_score(y_true_binary, score)
        return roc_auc_score(y_true_binary, score)
    except Exception:
        return np.nan


def compute_metrics_from_predictions(df):
    y_true5 = df["true_label5"].astype(int).values
    y_pred5 = df["pred_label5"].astype(int).values

    y_true_bcs = np.array([LABEL5_TO_BCS[int(x)] for x in y_true5])
    y_pred_bcs = np.array([LABEL5_TO_BCS[int(x)] for x in y_pred5])

    y_true3 = np.array([label5_to_label3(x) for x in y_true5])
    y_pred3 = np.array([label5_to_label3(x) for x in y_pred5])

    m = {}

    if HAS_SKLEARN:
        m["acc5"] = accuracy_score(y_true5, y_pred5)
        m["balanced_acc5"] = balanced_accuracy_score(y_true5, y_pred5)
        m["macro_f1_5"] = f1_score(y_true5, y_pred5, average="macro", zero_division=0)
        m["weighted_f1_5"] = f1_score(y_true5, y_pred5, average="weighted", zero_division=0)
        m["acc3_final"] = accuracy_score(y_true3, y_pred3)
        m["macro_f1_3_final"] = f1_score(y_true3, y_pred3, average="macro", zero_division=0)
        m["weighted_f1_3_final"] = f1_score(y_true3, y_pred3, average="weighted", zero_division=0)
    else:
        m["acc5"] = float(np.mean(y_true5 == y_pred5))
        m["balanced_acc5"] = np.nan
        m["macro_f1_5"] = np.nan
        m["weighted_f1_5"] = np.nan
        m["acc3_final"] = float(np.mean(y_true3 == y_pred3))
        m["macro_f1_3_final"] = np.nan
        m["weighted_f1_3_final"] = np.nan

    m["qwk5"] = qwk_score(y_true5, y_pred5)
    m["qwk3_final"] = qwk_score(y_true3, y_pred3)
    m["mae_bcs"] = float(np.mean(np.abs(y_true_bcs - y_pred_bcs)))
    m["within_one_acc"] = float(np.mean(np.abs(y_true_bcs - y_pred_bcs) <= 1))

    # Management class details.
    if HAS_SKLEARN:
        pr, rc, f1, sup = precision_recall_fscore_support(
            y_true3, y_pred3, labels=[0, 1, 2], zero_division=0
        )
        m["lean_precision_final"] = float(pr[0])
        m["lean_recall_final"] = float(rc[0])
        m["high_precision_final"] = float(pr[2])
        m["high_recall_final"] = float(rc[2])
    else:
        m["lean_precision_final"] = np.nan
        m["lean_recall_final"] = np.nan
        m["high_precision_final"] = np.nan
        m["high_recall_final"] = np.nan

    # Aux branch binary metrics.
    if "low_score" in df.columns:
        low_true = (y_true3 == 0).astype(int)
        low_pred = (df["low_score"].values >= 0.5).astype(int)
        high_true = (y_true3 == 2).astype(int)
        high_pred = (df["high_score"].values >= 0.5).astype(int)

        for name, true, pred, score in [
            ("low", low_true, low_pred, df["low_score"].values),
            ("high_branch", high_true, high_pred, df["high_score"].values),
        ]:
            tp = int(((true == 1) & (pred == 1)).sum())
            fp = int(((true == 0) & (pred == 1)).sum())
            fn = int(((true == 1) & (pred == 0)).sum())
            tn = int(((true == 0) & (pred == 0)).sum())

            m[f"{name}_precision"] = tp / max(tp + fp, 1)
            m[f"{name}_recall"] = tp / max(tp + fn, 1)
            m[f"{name}_specificity"] = tn / max(tn + fp, 1)
            m[f"{name}_pr_auc"] = safe_auc(true, score, kind="pr")
            m[f"{name}_roc_auc"] = safe_auc(true, score, kind="roc")

    # Extreme errors.
    m["lean_to_high_errors"] = int(((y_true3 == 0) & (y_pred3 == 2)).sum())
    m["high_to_lean_errors"] = int(((y_true3 == 2) & (y_pred3 == 0)).sum())
    m["bcs2_to_bcs6_errors"] = int(((y_true_bcs == 2) & (y_pred_bcs == 6)).sum())
    m["bcs6_to_bcs2_or_3_errors"] = int(((y_true_bcs == 6) & np.isin(y_pred_bcs, [2, 3])).sum())

    if "attn_entropy" in df.columns:
        m["attn_entropy_mean"] = float(df["attn_entropy"].mean())
        m["attn_entropy_std"] = float(df["attn_entropy"].std())

    return m


def mean_std_table(metrics_list):
    keys = sorted(set().union(*[m.keys() for m in metrics_list]))
    rows = []
    for k in keys:
        vals_raw = [m.get(k, np.nan) for m in metrics_list]

        vals_num = pd.to_numeric(pd.Series(vals_raw), errors="coerce").values.astype(np.float64)

        if np.all(np.isnan(vals_num)):
            continue

        rows.append({
            "metric": k,
            "mean": np.nanmean(vals_num),
            "std": np.nanstd(vals_num, ddof=1) if np.sum(~np.isnan(vals_num)) > 1 else 0.0,
        })
    return pd.DataFrame(rows)



def train_one_epoch(model, loader, optimizer, scaler, device, ce_weight, use_lowhigh, amp=True):
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        angle = batch["angle"].to(device, non_blocking=True)
        label5 = batch["label5"].to(device, non_blocking=True)
        label3 = batch["label3"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            outputs = model(rgb, depth, angle)
            loss, _ = compute_loss(outputs, label5, label3, ce_weight, use_lowhigh=use_lowhigh)

        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        bs = label5.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict(model, loader, device, amp=True):
    model.eval()
    rows = []

    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        angle = batch["angle"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            outputs = model(rgb, depth, angle)
            prob5 = torch.softmax(outputs["logits5"], dim=1)
            pred5 = prob5.argmax(dim=1)
            low_score = torch.sigmoid(outputs["low_logit"])
            high_score = torch.sigmoid(outputs["high_logit"])

        for i, bag_id in enumerate(batch["bag_id"]):
            row = {
                "bag_id": bag_id,
                "true_label5": int(batch["label5"][i].item()),
                "pred_label5": int(pred5[i].detach().cpu().item()),
                "true_bcs": int(batch["bcs_raw"][i].item()),
                "pred_bcs": int(LABEL5_TO_BCS[int(pred5[i].detach().cpu().item())]),
                "true_label3": int(batch["label3"][i].item()),
                "pred_label3_final": int(label5_to_label3(int(pred5[i].detach().cpu().item()))),
                "low_score": float(low_score[i].detach().cpu().item()),
                "high_score": float(high_score[i].detach().cpu().item()),
                "attn_entropy": float(outputs["attn_entropy"][i].detach().cpu().item()),
                "good_frame_ratio": float(batch["good_frame_ratio"][i].item()),
                "good_angle_frame_count": float(batch["good_angle_frame_count"][i].item()),
            }
            for c in range(5):
                row[f"prob_bcs{LABEL5_TO_BCS[c]}"] = float(prob5[i, c].detach().cpu().item())
            rows.append(row)

    return pd.DataFrame(rows)


def build_model_for_variant(variant, device):
    if variant == "rgb_only_lowhigh":
        return ImageMILModel(modality="rgb", use_lowhigh=True, use_angle=False).to(device), "rgb", True, False, False

    if variant == "depth_only_lowhigh":
        return ImageMILModel(modality="depth", use_lowhigh=True, use_angle=False).to(device), "depth", True, False, False

    if variant == "rgb_depth_comil":
        return ImageMILModel(modality="rgb_depth", use_lowhigh=False, use_angle=False).to(device), "rgb_depth", False, False, False

    if variant == "rgb_depth_lowhigh":
        return ImageMILModel(modality="rgb_depth", use_lowhigh=True, use_angle=False).to(device), "rgb_depth", True, False, False

    if variant == "qg_rgb_depth_lowhigh_no_angle":
        return ImageMILModel(modality="rgb_depth", use_lowhigh=True, use_angle=False).to(device), "rgb_depth", True, True, False

    if variant == "qg_rgb_depth_lowhigh_with_angle":
        return ImageMILModel(modality="rgb_depth", use_lowhigh=True, use_angle=True).to(device), "rgb_depth", True, True, True

    if variant == "angle_only_mlp":
        return AngleOnlyMLP(use_lowhigh=True).to(device), "rgb_depth", True, True, True

    raise ValueError(f"Unknown variant: {variant}")


def split_train_val_test(bag_df, test_fold, val_fraction=0.22, seed=42):
    test_df = bag_df[bag_df["fold_id"].astype(int) == int(test_fold)].copy()
    train_all = bag_df[bag_df["fold_id"].astype(int) != int(test_fold)].copy()

    # Animal-wise validation split from train_all.
    animals = train_all["animal_id"].astype(str).unique().tolist()
    rng = np.random.default_rng(seed + int(test_fold) * 100)
    rng.shuffle(animals)

    n_val = max(1, int(round(len(animals) * val_fraction)))
    val_animals = set(animals[:n_val])

    val_df = train_all[train_all["animal_id"].astype(str).isin(val_animals)].copy()
    train_df = train_all[~train_all["animal_id"].astype(str).isin(val_animals)].copy()

    # If validation lacks a class badly, still proceed; dataset is small.
    return train_df, val_df, test_df


def run_variant(variant, args, device):
    print("\n" + "=" * 120)
    print(f"Running variant: {variant}")
    print("=" * 120)

    # Create model once to know settings, then recreate per fold.
    dummy_model, modality, use_lowhigh, quality_gated, use_angle = build_model_for_variant(variant, device)
    del dummy_model
    torch.cuda.empty_cache()

    _, bag_df = load_frame_index(
        quality_gated=quality_gated,
        min_good_frames=args.min_good_frames,
        require_angle=use_angle or quality_gated,
    )

    raw_total_bags = safe_read_csv(FRAME_CSV).groupby("bag_id").size().shape[0]
    eligible_total_bags = bag_df["bag_id"].nunique()
    coverage = eligible_total_bags / max(raw_total_bags, 1)

    print(f"Raw total bags:      {raw_total_bags}")
    print(f"Eligible total bags: {eligible_total_bags}")
    print(f"Coverage:            {coverage:.4f}")
    print(f"BCS distribution:    {Counter(bag_df['bcs_raw'].astype(int).tolist())}")

    variant_dir = make_out_dir(OUT_DIR / variant)

    all_fold_metrics = []
    all_predictions = []

    for test_fold in sorted(bag_df["fold_id"].astype(int).unique()):
        print("\n" + "-" * 100)
        print(f"Variant {variant} | Test fold {test_fold}")
        print("-" * 100)

        set_seed(int(args.seed) + int(test_fold))

        train_df, val_df, test_df = split_train_val_test(
            bag_df,
            test_fold=test_fold,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )

        print(f"Train bags: {len(train_df)} | Val bags: {len(val_df)} | Test bags: {len(test_df)}")
        print(f"Train BCS: {Counter(train_df['bcs_raw'].astype(int).tolist())}")
        print(f"Val BCS:   {Counter(val_df['bcs_raw'].astype(int).tolist())}")
        print(f"Test BCS:  {Counter(test_df['bcs_raw'].astype(int).tolist())}")

        if use_angle:
            angle_mean, angle_std = fit_angle_scaler(train_df)
            print(f"Angle mean: {angle_mean.tolist()}")
            print(f"Angle std:  {angle_std.tolist()}")
        else:
            angle_mean = np.zeros(3, dtype=np.float32)
            angle_std = np.ones(3, dtype=np.float32)

        train_ds = BagDataset(
            train_df,
            modality=modality,
            use_angle=use_angle,
            angle_mean=angle_mean,
            angle_std=angle_std,
            img_size=args.img_size,
            frames_per_bag=args.frames_per_bag,
            train=True,
            depth_crop_ratio=args.depth_crop_ratio,
        )
        val_ds = BagDataset(
            val_df,
            modality=modality,
            use_angle=use_angle,
            angle_mean=angle_mean,
            angle_std=angle_std,
            img_size=args.img_size,
            frames_per_bag=args.frames_per_bag,
            train=False,
            depth_crop_ratio=args.depth_crop_ratio,
        )
        test_ds = BagDataset(
            test_df,
            modality=modality,
            use_angle=use_angle,
            angle_mean=angle_mean,
            angle_std=angle_std,
            img_size=args.img_size,
            frames_per_bag=args.frames_per_bag,
            train=False,
            depth_crop_ratio=args.depth_crop_ratio,
        )

        sampler = make_sampler(train_df)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_bags,
            persistent_workers=args.num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_bags,
            persistent_workers=args.num_workers > 0,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_bags,
            persistent_workers=args.num_workers > 0,
        )

        model, _, _, _, _ = build_model_for_variant(variant, device)
        ce_weight = class_weights_from_bags(train_df, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

        best_score = -1e9
        best_state = None
        best_epoch = -1

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                ce_weight=ce_weight,
                use_lowhigh=use_lowhigh,
                amp=args.amp,
            )
            scheduler.step()

            val_pred = predict(model, val_loader, device, amp=args.amp)
            val_metrics = compute_metrics_from_predictions(val_pred)

            # Balanced model selection:
            # prioritize management macro-F1 and ordinal consistency, penalize extreme errors.
            score = (
                val_metrics.get("macro_f1_3_final", 0.0)
                + 0.35 * val_metrics.get("qwk5", 0.0)
                - 0.04 * (
                    val_metrics.get("lean_to_high_errors", 0)
                    + val_metrics.get("high_to_lean_errors", 0)
                    + val_metrics.get("bcs2_to_bcs6_errors", 0)
                    + val_metrics.get("bcs6_to_bcs2_or_3_errors", 0)
                )
            )

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
                print(
                    f"Fold {test_fold} | Epoch {epoch:03d}/{args.epochs} | "
                    f"Loss {loss:.4f} | "
                    f"Val F1_3 {val_metrics.get('macro_f1_3_final', np.nan):.4f} | "
                    f"Val QWK5 {val_metrics.get('qwk5', np.nan):.4f} | "
                    f"Val Within1 {val_metrics.get('within_one_acc', np.nan):.4f} | "
                    f"Best {best_score:.4f}@{best_epoch} | "
                    f"Time {time.time() - t0:.1f}s"
                )

        if best_state is not None:
            model.load_state_dict(best_state)

        test_pred = predict(model, test_loader, device, amp=args.amp)
        test_pred["fold_id"] = int(test_fold)
        test_pred["variant"] = variant
        test_pred["coverage"] = coverage
        test_pred["eligible_total_bags"] = eligible_total_bags
        test_pred["raw_total_bags"] = raw_total_bags

        test_metrics = compute_metrics_from_predictions(test_pred)
        test_metrics["fold_id"] = int(test_fold)
        test_metrics["variant"] = variant
        test_metrics["raw_test_bags"] = int(safe_read_csv(FRAME_CSV).groupby("bag_id").first().reset_index().query("fold_id == @test_fold").shape[0])
        test_metrics["eligible_test_bags"] = int(len(test_df))
        test_metrics["test_coverage"] = float(len(test_df) / max(test_metrics["raw_test_bags"], 1))

        print("\nFold test result")
        print("-" * 100)
        for k in [
            "test_coverage", "acc5", "balanced_acc5", "macro_f1_5", "weighted_f1_5",
            "qwk5", "mae_bcs", "within_one_acc",
            "acc3_final", "macro_f1_3_final", "qwk3_final",
            "lean_recall_final", "high_recall_final",
            "low_pr_auc", "high_branch_pr_auc",
            "lean_to_high_errors", "high_to_lean_errors",
            "bcs2_to_bcs6_errors", "bcs6_to_bcs2_or_3_errors",
        ]:
            if k in test_metrics:
                print(f"{k:30s}: {test_metrics[k]}")

        all_fold_metrics.append(test_metrics)
        all_predictions.append(test_pred)

    fold_metrics_df = pd.DataFrame(all_fold_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    mean_std_df = mean_std_table(all_fold_metrics)

    fold_metrics_path = variant_dir / "fold_metrics.csv"
    pred_path = variant_dir / "predictions.csv"
    mean_std_path = variant_dir / "mean_std.csv"

    fold_metrics_df.to_csv(fold_metrics_path, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    mean_std_df.to_csv(mean_std_path, index=False, encoding="utf-8-sig")

    make_aggregate_report(predictions_df, variant_dir, variant)

    print(f"\nSaved variant results:")
    print(fold_metrics_path)
    print(mean_std_path)
    print(pred_path)

    return fold_metrics_df, mean_std_df, predictions_df



def make_aggregate_report(pred_df, out_dir, variant):
    y_true5 = pred_df["true_label5"].astype(int).values
    y_pred5 = pred_df["pred_label5"].astype(int).values
    y_true3 = np.array([label5_to_label3(x) for x in y_true5])
    y_pred3 = np.array([label5_to_label3(x) for x in y_pred5])

    cm5 = confusion_matrix(y_true5, y_pred5, labels=[0, 1, 2, 3, 4]) if HAS_SKLEARN else np.zeros((5,5), dtype=int)
    cm3 = confusion_matrix(y_true3, y_pred3, labels=[0, 1, 2]) if HAS_SKLEARN else np.zeros((3,3), dtype=int)

    report_path = out_dir / "aggregate_confusion_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Aggregate report for {variant}\n")
        f.write("=" * 100 + "\n\n")
        f.write("5-class confusion matrix\n")
        f.write("Rows=true, columns=predicted\n")
        f.write(str(cm5) + "\n\n")
        f.write("3-class final management confusion matrix\n")
        f.write("Rows=true, columns=predicted\n")
        f.write(str(cm3) + "\n\n")
        f.write("Metrics\n")
        f.write(json.dumps(compute_metrics_from_predictions(pred_df), indent=2, ensure_ascii=False) + "\n\n")

        if HAS_SKLEARN:
            f.write("5-class classification report\n")
            f.write(classification_report(y_true5, y_pred5, labels=[0,1,2,3,4], target_names=[f"BCS{b}" for b in BCS_CLASSES], zero_division=0))
            f.write("\n\n3-class classification report\n")
            f.write(classification_report(y_true3, y_pred3, labels=[0,1,2], target_names=MGMT_NAMES, zero_division=0))

    plot_cm(cm5, [f"BCS{b}" for b in BCS_CLASSES], [f"BCS{b}" for b in BCS_CLASSES], f"{variant}: 5-class confusion matrix", out_dir / "aggregate_cm5.png")
    plot_cm(cm3, MGMT_NAMES, MGMT_NAMES, f"{variant}: final management confusion matrix", out_dir / "aggregate_cm3.png")


def plot_cm(cm, row_labels, col_labels, title, path):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count", rotation=270, labelpad=12)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_all_variants(results):
    summary_rows = []
    extreme_rows = []

    for variant, (fold_metrics_df, mean_std_df, pred_df) in results.items():
        ms = mean_std_df.set_index("metric")

        def get(metric):
            if metric in ms.index:
                return float(ms.loc[metric, "mean"])
            return np.nan

        def get_std(metric):
            if metric in ms.index:
                return float(ms.loc[metric, "std"])
            return np.nan

        row = {
            "Variant": variant,
            "Coverage": get("test_coverage"),
            "Acc5": get("acc5"),
            "Macro-F1 5": get("macro_f1_5"),
            "Weighted-F1 5": get("weighted_f1_5"),
            "QWK5": get("qwk5"),
            "MAE": get("mae_bcs"),
            "Within-one Acc": get("within_one_acc"),
            "Acc3 final": get("acc3_final"),
            "Macro-F1 3 final": get("macro_f1_3_final"),
            "QWK3 final": get("qwk3_final"),
            "Lean recall final": get("lean_recall_final"),
            "High recall final": get("high_recall_final"),
            "Low PR-AUC": get("low_pr_auc"),
            "High PR-AUC": get("high_branch_pr_auc"),
        }
        summary_rows.append(row)

        extreme_rows.append({
            "Variant": variant,
            "Coverage": get("test_coverage"),
            "Lean→High": int(pred_df.assign(
                true3=pred_df["true_label5"].apply(label5_to_label3),
                pred3=pred_df["pred_label5"].apply(label5_to_label3)
            ).query("true3 == 0 and pred3 == 2").shape[0]),
            "High→Lean": int(pred_df.assign(
                true3=pred_df["true_label5"].apply(label5_to_label3),
                pred3=pred_df["pred_label5"].apply(label5_to_label3)
            ).query("true3 == 2 and pred3 == 0").shape[0]),
            "BCS2→BCS6": int(pred_df.query("true_bcs == 2 and pred_bcs == 6").shape[0]),
            "BCS6→BCS2/3": int(pred_df.query("true_bcs == 6 and pred_bcs in [2, 3]").shape[0]),
        })

    summary_df = pd.DataFrame(summary_rows)
    extreme_df = pd.DataFrame(extreme_rows)

    summary_path = OUT_DIR / "12_ablation_summary_table.csv"
    extreme_path = OUT_DIR / "12_ablation_extreme_error_table.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    extreme_df.to_csv(extreme_path, index=False, encoding="utf-8-sig")

    plot_summary_bars(summary_df)
    plot_extreme_bars(extreme_df)

    print("\nAblation summary saved:")
    print(summary_path)
    print(extreme_path)

    return summary_df, extreme_df


def plot_summary_bars(summary_df):
    metrics = ["Acc5", "Macro-F1 5", "QWK5", "Within-one Acc", "Macro-F1 3 final"]
    variants = summary_df["Variant"].tolist()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(metrics))
    width = 0.11

    for i, (_, row) in enumerate(summary_df.iterrows()):
        values = [row[m] for m in metrics]
        ax.bar(x + (i - (len(summary_df)-1)/2) * width, values, width, label=row["Variant"])

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Ablation study: main metrics")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(OUT_DIR / "12_ablation_main_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_extreme_bars(extreme_df):
    metrics = ["Lean→High", "High→Lean", "BCS2→BCS6", "BCS6→BCS2/3"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(metrics))
    width = 0.11

    for i, (_, row) in enumerate(extreme_df.iterrows()):
        values = [row[m] for m in metrics]
        ax.bar(x + (i - (len(extreme_df)-1)/2) * width, values, width, label=row["Variant"])

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_ylabel("Number of extreme errors")
    ax.set_title("Ablation study: extreme errors")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(OUT_DIR / "12_ablation_extreme_errors.png", dpi=300, bbox_inches="tight")
    plt.close(fig)



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        help=f"Variants to run. Default: {DEFAULT_VARIANTS}",
    )
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--frames-per-bag", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--depth-crop-ratio", type=float, default=0.70)
    parser.add_argument("--min-good-frames", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Skip variants with predictions.csv already present.")

    args = parser.parse_args()
    args.amp = not args.no_amp
    return args


def load_existing_variant(variant):
    variant_dir = OUT_DIR / variant
    fold_path = variant_dir / "fold_metrics.csv"
    mean_path = variant_dir / "mean_std.csv"
    pred_path = variant_dir / "predictions.csv"
    if fold_path.exists() and mean_path.exists() and pred_path.exists():
        return (
            pd.read_csv(fold_path, encoding="utf-8-sig"),
            pd.read_csv(mean_path, encoding="utf-8-sig"),
            pd.read_csv(pred_path, encoding="utf-8-sig"),
        )
    return None


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()

    print("=" * 120)
    print("Comprehensive ablation for Animals manuscript")
    print(f"BASE_PATH: {BASE_PATH}")
    print(f"OUT_DIR:   {OUT_DIR}")
    print(f"Variants:  {args.variants}")
    print("=" * 120)

    if not FRAME_CSV.exists():
        raise FileNotFoundError(f"Cannot find {FRAME_CSV}")

    # Check QC files only if needed.
    if any(v.startswith("qg_") or v == "angle_only_mlp" for v in args.variants):
        if not QC_FRAME_CSV.exists() or not QC_BAG_CSV.exists():
            raise FileNotFoundError(
                "Quality-gated or angle variants require 08e QC files.\n"
                f"Missing one of:\n{QC_FRAME_CSV}\n{QC_BAG_CSV}"
            )

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    results = {}

    for variant in args.variants:
        if args.skip_existing:
            existing = load_existing_variant(variant)
            if existing is not None:
                print(f"\n[Skip existing] {variant}")
                results[variant] = existing
                continue

        results[variant] = run_variant(variant, args, device)

    summarize_all_variants(results)

    print("\nFinished successfully.")
    print(f"Results directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
