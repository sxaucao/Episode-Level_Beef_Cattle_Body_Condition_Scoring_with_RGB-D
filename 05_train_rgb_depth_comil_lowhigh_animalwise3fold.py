import os
os.environ["OMP_NUM_THREADS"] = "1"

import csv
import copy
import random
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns



BASE_PATH = Path(r"./dataset")
CSV_PATH = Path("./02_make_longitudinal_protocol_splits/02A_animalwise_3fold.csv")

OUT_DIR = Path("./05_rgb_depth_comil_lowhigh_animalwise3fold")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
NUM_CLASSES = 5
N_SPLITS = 3

IMG_SIZE = 224
FRAMES_PER_BAG = 12

BATCH_SIZE = 12
EPOCHS = 70
LR = 3e-5
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
AMP = True

VAL_RATIO = 0.20

DEPTH_CENTER_CROP_RATIO = 0.70

# Loss weights
LAMBDA_LOW = 0.80
LAMBDA_HIGH = 0.50
LAMBDA_SUPCON = 0.20

# Focal parameters
LOW_ALPHA = 0.80
HIGH_ALPHA = 0.75
FOCAL_GAMMA = 2.0

# Thresholds for management 3-class prediction
LOW_THR = 0.45
HIGH_THR = 0.50

LABEL5_TO_RAW = {
    0: 2,
    1: 3,
    2: 4,
    3: 5,
    4: 6,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def label5_to_label3(label5):
    raw = LABEL5_TO_RAW[int(label5)]
    if raw in [2, 3]:
        return 0
    if raw in [4, 5]:
        return 1
    if raw == 6:
        return 2
    raise ValueError(raw)


def label3_name(label3):
    return {
        0: "Lean",
        1: "Ideal",
        2: "High-condition",
    }[int(label3)]


def build_bags(rows):
    bag_dict = {}

    for r in rows:
        bag_id = r["bag_id"]

        if bag_id not in bag_dict:
            label5 = int(r["label5"])
            bag_dict[bag_id] = {
                "bag_id": bag_id,
                "episode_id": r.get("episode_id", ""),
                "animal_id": r.get("animal_id", ""),
                "split_protocol": r.get("split_protocol", ""),
                "split_unit_id": r.get("split_unit_id", ""),
                "bcs_raw": int(r["bcs_raw"]),
                "label5": label5,
                "label3": int(r["label3"]),
                "is_low": 1 if label5 in [0, 1] else 0,   # BCS2/3
                "is_high": int(r["is_high"]),             # BCS6
                "fold_id": int(r["fold_id"]),
                "frames": [],
            }

        if int(r["label5"]) != bag_dict[bag_id]["label5"]:
            raise RuntimeError(f"Bag {bag_id} has multiple label5 values.")

        if int(r["fold_id"]) != bag_dict[bag_id]["fold_id"]:
            raise RuntimeError(f"Bag {bag_id} appears in multiple folds.")

        bag_dict[bag_id]["frames"].append({
            "rgb_path": r["rgb_path"],
            "depth_npy_path": r["depth_npy_path"],
            "frame_key": r.get("frame_key", ""),
        })

    bags = list(bag_dict.values())

    for b in bags:
        b["frame_count"] = len(b["frames"])

    bags = sorted(bags, key=lambda x: x["bag_id"])
    return bags


def verify_animalwise_no_leakage(bags):
    animal_to_folds = defaultdict(set)

    for b in bags:
        animal_id = b.get("animal_id", "")
        animal_to_folds[animal_id].add(int(b["fold_id"]))

    leaked = {
        aid: folds for aid, folds in animal_to_folds.items()
        if len(folds) > 1
    }

    if leaked:
        print("[ERROR] Animal leakage detected!")
        for aid, folds in list(leaked.items())[:20]:
            print(aid, folds)
        raise RuntimeError("Same animal appears in multiple folds.")

    print("No animal-level leakage detected.")


def split_train_val_by_animal(candidate_bags, val_ratio=0.2, seed=42):
    rng = random.Random(seed)

    animal_to_bags = defaultdict(list)

    for b in candidate_bags:
        animal_to_bags[b["animal_id"]].append(b)

    animal_units = []

    for aid, bs in animal_to_bags.items():
        bcs_counter = Counter(int(b["bcs_raw"]) for b in bs)
        frame_count = sum(int(b["frame_count"]) for b in bs)

        rarity = 0
        if bcs_counter[2] > 0 or bcs_counter[3] > 0:
            rarity += 120
        if bcs_counter[6] > 0:
            rarity += 60
        if bcs_counter[4] > 0:
            rarity += 10
        if bcs_counter[5] > 0:
            rarity += 10

        animal_units.append({
            "animal_id": aid,
            "bags": bs,
            "bcs_counter": bcs_counter,
            "frame_count": frame_count,
            "rarity": rarity,
        })

    n_animals = len(animal_units)
    n_val = max(1, int(round(n_animals * val_ratio)))

    animal_units = sorted(
        animal_units,
        key=lambda x: (-x["rarity"], -x["frame_count"], x["animal_id"])
    )

    val_ids = set()

    for target_group in ["low", "high", "mid"]:
        if len(val_ids) >= n_val:
            break

        if target_group == "low":
            candidates = [
                u for u in animal_units
                if u["animal_id"] not in val_ids and (u["bcs_counter"][2] > 0 or u["bcs_counter"][3] > 0)
            ]
        elif target_group == "high":
            candidates = [
                u for u in animal_units
                if u["animal_id"] not in val_ids and u["bcs_counter"][6] > 0
            ]
        else:
            candidates = [
                u for u in animal_units
                if u["animal_id"] not in val_ids and (u["bcs_counter"][4] > 0 or u["bcs_counter"][5] > 0)
            ]

        if candidates:
            val_ids.add(candidates[0]["animal_id"])

    remaining = [u for u in animal_units if u["animal_id"] not in val_ids]
    rng.shuffle(remaining)

    for u in remaining:
        if len(val_ids) >= n_val:
            break
        val_ids.add(u["animal_id"])

    train_bags = []
    val_bags = []

    for aid, bs in animal_to_bags.items():
        if aid in val_ids:
            val_bags.extend(bs)
        else:
            train_bags.extend(bs)

    return train_bags, val_bags


class RGBDepthBagDataset(Dataset):
    def __init__(self, bags, frames_per_bag=12, train=True):
        self.bags = bags
        self.frames_per_bag = frames_per_bag
        self.train = train

    def __len__(self):
        return len(self.bags)

    def sample_frames(self, frames):
        n = len(frames)
        k = self.frames_per_bag

        if self.train:
            if n >= k:
                return random.sample(frames, k)
            return random.choices(frames, k=k)

        if n >= k:
            idxs = np.linspace(0, n - 1, k).astype(int).tolist()
            return [frames[i] for i in idxs]

        return frames + random.choices(frames, k=k - n)

    def load_depth_pil(self, path):
        arr = np.load(path).astype(np.float32)

        valid = arr > 0

        if valid.sum() > 10:
            valid_values = arr[valid]

            # 最有限的速度优化：抽样计算 percentile，避免整图排序
            if valid_values.size > 50000:
                step = max(1, valid_values.size // 50000)
                valid_values = valid_values[::step]

            lo = np.percentile(valid_values, 1)
            hi = np.percentile(valid_values, 99)

            if hi <= lo:
                hi = lo + 1.0

            arr = np.clip(arr, lo, hi)
            arr = (arr - lo) / (hi - lo)
            arr[~valid] = 0.0
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)

        # Depth center crop 0.70
        ratio = DEPTH_CENTER_CROP_RATIO

        if ratio < 0.999:
            h, w = arr.shape[:2]
            new_h = int(h * ratio)
            new_w = int(w * ratio)
            y1 = max((h - new_h) // 2, 0)
            x1 = max((w - new_w) // 2, 0)
            arr = arr[y1:y1 + new_h, x1:x1 + new_w]

        depth_img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        return depth_img

    def joint_transform(self, rgb_img, depth_img):
        rgb_img = TF.resize(
            rgb_img,
            [IMG_SIZE, IMG_SIZE],
            interpolation=InterpolationMode.BILINEAR,
        )

        depth_img = TF.resize(
            depth_img,
            [IMG_SIZE, IMG_SIZE],
            interpolation=InterpolationMode.BILINEAR,
        )

        if self.train:
            if random.random() < 0.5:
                rgb_img = TF.hflip(rgb_img)
                depth_img = TF.hflip(depth_img)

            angle = random.uniform(-8, 8)

            rgb_img = TF.rotate(
                rgb_img,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )

            depth_img = TF.rotate(
                depth_img,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )

        rgb_tensor = TF.to_tensor(rgb_img)
        rgb_tensor = TF.normalize(
            rgb_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        depth_tensor = TF.to_tensor(depth_img)
        depth_tensor = TF.normalize(
            depth_tensor,
            mean=[0.5],
            std=[0.25],
        )

        return rgb_tensor, depth_tensor

    def __getitem__(self, idx):
        bag = self.bags[idx]
        sampled = self.sample_frames(bag["frames"])

        rgb_frames = []
        depth_frames = []

        for f in sampled:
            rgb_img = Image.open(f["rgb_path"]).convert("RGB")
            depth_img = self.load_depth_pil(f["depth_npy_path"])

            rgb_tensor, depth_tensor = self.joint_transform(rgb_img, depth_img)

            rgb_frames.append(rgb_tensor)
            depth_frames.append(depth_tensor)

        rgb_frames = torch.stack(rgb_frames, dim=0)
        depth_frames = torch.stack(depth_frames, dim=0)

        return (
            rgb_frames,
            depth_frames,
            int(bag["label5"]),
            int(bag["label3"]),
            int(bag["is_low"]),
            int(bag["is_high"]),
            bag["bag_id"],
        )


class FrameEncoderRGB(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class FrameEncoderDepth(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        old_conv = resnet.conv1
        new_conv = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        resnet.conv1 = new_conv

        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class AttentionMILPooling(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, frame_feats):
        scores = self.attn(frame_feats).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        bag_feat = torch.sum(frame_feats * weights.unsqueeze(-1), dim=1)
        return bag_feat, weights


class CoralHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.fc = nn.Linear(in_features, 1, bias=False)

        init_bias = torch.linspace(1.5, -1.5, steps=num_classes - 1)
        self.bias = nn.Parameter(init_bias)

    def forward(self, x):
        score = self.fc(x)
        return score + self.bias


class RGBDepthCOMILLowHigh(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.rgb_encoder = FrameEncoderRGB()
        self.depth_encoder = FrameEncoderDepth()

        self.frame_fusion = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        self.pooling = AttentionMILPooling(in_dim=512, hidden_dim=128)

        self.ordinal_head = CoralHead(512, num_classes)
        self.low_head = nn.Linear(512, 1)
        self.high_head = nn.Linear(512, 1)

        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def forward(self, rgb_bag, depth_bag):
        b, k, c, h, w = rgb_bag.shape

        rgb = rgb_bag.reshape(b * k, 3, h, w)
        depth = depth_bag.reshape(b * k, 1, h, w)

        rgb_feat = self.rgb_encoder(rgb)
        depth_feat = self.depth_encoder(depth)

        frame_feat = torch.cat([rgb_feat, depth_feat], dim=1)
        frame_feat = self.frame_fusion(frame_feat)
        frame_feat = frame_feat.reshape(b, k, -1)

        bag_feat, weights = self.pooling(frame_feat)

        ordinal_logits = self.ordinal_head(bag_feat)
        low_logit = self.low_head(bag_feat).squeeze(1)
        high_logit = self.high_head(bag_feat).squeeze(1)
        contrast_feat = self.proj(bag_feat)

        return ordinal_logits, low_logit, high_logit, contrast_feat, weights


def coral_targets(labels, num_classes):
    targets = torch.zeros((labels.size(0), num_classes - 1), device=labels.device)

    for i in range(num_classes - 1):
        targets[:, i] = (labels > i).float()

    return targets


def coral_loss(logits, labels):
    targets = coral_targets(labels, NUM_CLASSES)
    return F.binary_cross_entropy_with_logits(logits, targets)


def coral_predict(logits):
    probs = torch.sigmoid(logits)
    return torch.sum(probs > 0.5, dim=1)


def binary_focal_loss_with_logits(logits, targets, alpha=0.75, gamma=2.0):
    targets = targets.float()

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)

    pt = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    loss = alpha_t * ((1 - pt) ** gamma) * bce
    return loss.mean()


def supervised_contrastive_loss(features, labels, temperature=0.15):
    device = features.device
    features = F.normalize(features, dim=1)
    labels = labels.contiguous().view(-1, 1)

    batch_size = features.size(0)

    if batch_size <= 1:
        return torch.tensor(0.0, device=device)

    mask = torch.eq(labels, labels.T).float().to(device)
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * logits_mask

    logits = torch.matmul(features, features.T) / temperature
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    pos_count = mask.sum(dim=1)
    valid = pos_count > 0

    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (pos_count + 1e-8)
    loss = -mean_log_prob_pos[valid].mean()

    return loss


def make_bag_sampler(bags):
    labels = [int(b["label5"]) for b in bags]
    counter = Counter(labels)

    weights_per_class = {
        c: 1.0 / max(counter[c], 1)
        for c in range(NUM_CLASSES)
    }

    # 额外提高 low 类别出现概率，但不改变样本本身
    sample_weights = []
    for b in bags:
        y = int(b["label5"])
        w = weights_per_class[y]
        if int(b["is_low"]) == 1:
            w *= 1.5
        sample_weights.append(w)

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def compute_specificity(y_true_bin, y_pred_bin):
    y_true_bin = np.array(y_true_bin)
    y_pred_bin = np.array(y_pred_bin)

    tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))

    if tn + fp == 0:
        return np.nan

    return tn / (tn + fp)


def safe_pr_auc(y_true, y_score):
    if len(set(y_true)) == 2:
        return average_precision_score(y_true, y_score)
    return np.nan


def safe_roc_auc(y_true, y_score):
    if len(set(y_true)) == 2:
        return roc_auc_score(y_true, y_score)
    return np.nan


def management_pred_from_low_high(low_prob, high_prob, low_thr=LOW_THR, high_thr=HIGH_THR):
    """
    low branch has priority because Lean was previously never predicted.
    """
    if low_prob >= low_thr:
        return 0
    if high_prob >= high_thr:
        return 2
    return 1


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    y_true5 = []
    y_pred5 = []

    y_true3_from_ordinal = []
    y_pred3_from_ordinal = []

    y_true3_mgmt = []
    y_pred3_mgmt = []

    low_true = []
    low_pred = []
    low_score = []

    high_true = []
    high_pred_ordinal = []
    high_pred_branch = []
    high_score = []

    bag_ids = []
    attn_entropy = []

    for rgb, depth, labels5, labels3, is_low, is_high, ids in loader:
        rgb = rgb.to(DEVICE, non_blocking=True)
        depth = depth.to(DEVICE, non_blocking=True)
        labels5 = labels5.to(DEVICE, non_blocking=True)

        ordinal_logits, low_logit, high_logit, contrast_feat, weights = model(rgb, depth)

        pred5 = coral_predict(ordinal_logits)

        low_prob = torch.sigmoid(low_logit)
        high_prob = torch.sigmoid(high_logit)

        labels5_np = labels5.cpu().numpy().tolist()
        pred5_np = pred5.cpu().numpy().tolist()

        labels3_np = labels3.cpu().numpy().tolist()
        is_low_np = is_low.cpu().numpy().tolist()
        is_high_np = is_high.cpu().numpy().tolist()

        low_prob_np = low_prob.cpu().numpy().tolist()
        high_prob_np = high_prob.cpu().numpy().tolist()

        weights_np = weights.detach().cpu().numpy()
        ent = -np.sum(weights_np * np.log(weights_np + 1e-8), axis=1)
        attn_entropy.extend(ent.tolist())

        for yt5, yp5, yt3, lt, ht, lp, hp, bid in zip(
            labels5_np,
            pred5_np,
            labels3_np,
            is_low_np,
            is_high_np,
            low_prob_np,
            high_prob_np,
            ids,
        ):
            yt5 = int(yt5)
            yp5 = int(yp5)
            yt3 = int(yt3)

            y_true5.append(yt5)
            y_pred5.append(yp5)

            true3_ord = label5_to_label3(yt5)
            pred3_ord = label5_to_label3(yp5)

            y_true3_from_ordinal.append(true3_ord)
            y_pred3_from_ordinal.append(pred3_ord)

            pred3_mgmt = management_pred_from_low_high(lp, hp)
            y_true3_mgmt.append(yt3)
            y_pred3_mgmt.append(pred3_mgmt)

            low_true.append(int(lt))
            low_pred.append(1 if lp >= LOW_THR else 0)
            low_score.append(float(lp))

            high_true.append(int(ht))
            high_pred_ordinal.append(1 if yp5 == 4 else 0)
            high_pred_branch.append(1 if hp >= HIGH_THR else 0)
            high_score.append(float(hp))

            bag_ids.append(bid)

    y_true5 = np.array(y_true5)
    y_pred5 = np.array(y_pred5)

    y_true_raw = np.array([LABEL5_TO_RAW[int(x)] for x in y_true5])
    y_pred_raw = np.array([LABEL5_TO_RAW[int(x)] for x in y_pred5])

    abs_err = np.abs(y_true_raw - y_pred_raw)

    metrics = {}

    # 5-class ordinal metrics
    metrics["acc5"] = accuracy_score(y_true5, y_pred5)
    metrics["balanced_acc5"] = balanced_accuracy_score(y_true5, y_pred5)
    metrics["macro_f1_5"] = f1_score(y_true5, y_pred5, average="macro", zero_division=0)
    metrics["weighted_f1_5"] = f1_score(y_true5, y_pred5, average="weighted", zero_division=0)
    metrics["qwk5"] = cohen_kappa_score(y_true5, y_pred5, weights="quadratic")
    metrics["mae_bcs"] = mean_absolute_error(y_true_raw, y_pred_raw)
    metrics["within_one_acc"] = float(np.mean(abs_err <= 1))

    # 3-class from ordinal head
    metrics["acc3_ordinal"] = accuracy_score(y_true3_from_ordinal, y_pred3_from_ordinal)
    metrics["macro_f1_3_ordinal"] = f1_score(y_true3_from_ordinal, y_pred3_from_ordinal, average="macro", zero_division=0)
    metrics["qwk3_ordinal"] = cohen_kappa_score(y_true3_from_ordinal, y_pred3_from_ordinal, weights="quadratic")

    # 3-class from low/high branches
    metrics["acc3_mgmt"] = accuracy_score(y_true3_mgmt, y_pred3_mgmt)
    metrics["macro_f1_3_mgmt"] = f1_score(y_true3_mgmt, y_pred3_mgmt, average="macro", zero_division=0)
    metrics["qwk3_mgmt"] = cohen_kappa_score(y_true3_mgmt, y_pred3_mgmt, weights="quadratic")

    metrics["lean_precision_mgmt"] = precision_score(
        [1 if y == 0 else 0 for y in y_true3_mgmt],
        [1 if y == 0 else 0 for y in y_pred3_mgmt],
        zero_division=0,
    )
    metrics["lean_recall_mgmt"] = recall_score(
        [1 if y == 0 else 0 for y in y_true3_mgmt],
        [1 if y == 0 else 0 for y in y_pred3_mgmt],
        zero_division=0,
    )

    # Low branch
    metrics["low_precision"] = precision_score(low_true, low_pred, zero_division=0)
    metrics["low_recall"] = recall_score(low_true, low_pred, zero_division=0)
    metrics["low_specificity"] = compute_specificity(low_true, low_pred)
    metrics["low_pr_auc"] = safe_pr_auc(low_true, low_score)
    metrics["low_roc_auc"] = safe_roc_auc(low_true, low_score)

    # High branch
    metrics["high_precision_ordinal"] = precision_score(high_true, high_pred_ordinal, zero_division=0)
    metrics["high_recall_ordinal"] = recall_score(high_true, high_pred_ordinal, zero_division=0)
    metrics["high_specificity_ordinal"] = compute_specificity(high_true, high_pred_ordinal)

    metrics["high_precision_branch"] = precision_score(high_true, high_pred_branch, zero_division=0)
    metrics["high_recall_branch"] = recall_score(high_true, high_pred_branch, zero_division=0)
    metrics["high_specificity_branch"] = compute_specificity(high_true, high_pred_branch)
    metrics["high_pr_auc"] = safe_pr_auc(high_true, high_score)
    metrics["high_roc_auc"] = safe_roc_auc(high_true, high_score)

    metrics["attn_entropy_mean"] = float(np.mean(attn_entropy))
    metrics["attn_entropy_std"] = float(np.std(attn_entropy))

    pred_rows = []

    for bid, yt5, yp5, yt3, yp3_ord, yp3_mgmt, lt, lp, ht, hp in zip(
        bag_ids,
        y_true5,
        y_pred5,
        y_true3_mgmt,
        y_pred3_from_ordinal,
        y_pred3_mgmt,
        low_true,
        low_score,
        high_true,
        high_score,
    ):
        pred_rows.append({
            "bag_id": bid,
            "true_label5": int(yt5),
            "pred_label5": int(yp5),
            "true_bcs": LABEL5_TO_RAW[int(yt5)],
            "pred_bcs": LABEL5_TO_RAW[int(yp5)],

            "true_label3": int(yt3),
            "pred_label3_ordinal": int(yp3_ord),
            "pred_label3_mgmt": int(yp3_mgmt),
            "true_label3_name": label3_name(yt3),
            "pred_label3_ordinal_name": label3_name(yp3_ord),
            "pred_label3_mgmt_name": label3_name(yp3_mgmt),

            "true_low": int(lt),
            "low_score": float(lp),
            "true_high": int(ht),
            "high_score": float(hp),
        })

    return (
        metrics,
        pred_rows,
        y_true5,
        y_pred5,
        y_true3_mgmt,
        y_pred3_mgmt,
        y_pred3_from_ordinal,
    )


def save_confusion_matrices(fold, y_true5, y_pred5, y_true3, y_pred3_mgmt):
    cm5 = confusion_matrix(y_true5, y_pred5, labels=[0, 1, 2, 3, 4])

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm5,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["BCS2", "BCS3", "BCS4", "BCS5", "BCS6"],
        yticklabels=["BCS2", "BCS3", "BCS4", "BCS5", "BCS6"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Fold {fold} - RGB-Depth LowHigh CO-MIL 5-class")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"fold{fold}_lowhigh_cm5.png", dpi=300)
    plt.close()

    cm3 = confusion_matrix(y_true3, y_pred3_mgmt, labels=[0, 1, 2])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm3,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Lean", "Ideal", "High"],
        yticklabels=["Lean", "Ideal", "High"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Fold {fold} - Management by Low/High Branches")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"fold{fold}_lowhigh_mgmt_cm3.png", dpi=300)
    plt.close()


def train_one_fold(fold, all_bags):
    print("\n" + "=" * 120)
    print(f"Animal-wise RGB-Depth CO-MIL LowHigh | Test Fold {fold}")
    print("=" * 120)

    test_bags = [b for b in all_bags if int(b["fold_id"]) == fold]
    candidate_train_bags = [b for b in all_bags if int(b["fold_id"]) != fold]

    train_bags, val_bags = split_train_val_by_animal(
        candidate_train_bags,
        val_ratio=VAL_RATIO,
        seed=SEED + fold,
    )

    print(f"Train bags: {len(train_bags)}")
    print(f"Val bags:   {len(val_bags)}")
    print(f"Test bags:  {len(test_bags)}")

    print("Train BCS:", Counter(b["bcs_raw"] for b in train_bags))
    print("Val BCS:  ", Counter(b["bcs_raw"] for b in val_bags))
    print("Test BCS: ", Counter(b["bcs_raw"] for b in test_bags))
    print("Train management:", Counter(b["label3"] for b in train_bags))
    print("Val management:  ", Counter(b["label3"] for b in val_bags))
    print("Test management: ", Counter(b["label3"] for b in test_bags))

    train_ds = RGBDepthBagDataset(train_bags, FRAMES_PER_BAG, train=True)
    val_ds = RGBDepthBagDataset(val_bags, FRAMES_PER_BAG, train=False)
    test_ds = RGBDepthBagDataset(test_bags, FRAMES_PER_BAG, train=False)

    sampler = make_bag_sampler(train_bags)

    common_loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
    )

    if NUM_WORKERS > 0:
        common_loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        **common_loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **common_loader_kwargs,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **common_loader_kwargs,
    )

    model = RGBDepthCOMILLowHigh(num_classes=NUM_CLASSES).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    scaler = torch.cuda.amp.GradScaler(enabled=(AMP and DEVICE.type == "cuda"))

    best_val_score = -999
    best_val_qwk = -999
    best_epoch = -1
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        model.train()

        total_loss = 0.0
        total_ord = 0.0
        total_low = 0.0
        total_high = 0.0
        total_supcon = 0.0

        for rgb, depth, labels5, labels3, is_low, is_high, ids in train_loader:
            rgb = rgb.to(DEVICE, non_blocking=True)
            depth = depth.to(DEVICE, non_blocking=True)
            labels5 = labels5.to(DEVICE, non_blocking=True)
            is_low = is_low.float().to(DEVICE, non_blocking=True)
            is_high = is_high.float().to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(AMP and DEVICE.type == "cuda")):
                ordinal_logits, low_logit, high_logit, contrast_feat, weights = model(rgb, depth)

                loss_ord = coral_loss(ordinal_logits, labels5)
                loss_low = binary_focal_loss_with_logits(
                    low_logit,
                    is_low,
                    alpha=LOW_ALPHA,
                    gamma=FOCAL_GAMMA,
                )
                loss_high = binary_focal_loss_with_logits(
                    high_logit,
                    is_high,
                    alpha=HIGH_ALPHA,
                    gamma=FOCAL_GAMMA,
                )
                loss_supcon = supervised_contrastive_loss(contrast_feat, labels5)

                loss = (
                    loss_ord
                    + LAMBDA_LOW * loss_low
                    + LAMBDA_HIGH * loss_high
                    + LAMBDA_SUPCON * loss_supcon
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_ord += loss_ord.item()
            total_low += loss_low.item()
            total_high += loss_high.item()
            total_supcon += loss_supcon.item()

        scheduler.step()

        val_metrics, _, _, _, _, _, _ = evaluate(model, val_loader)

        val_qwk = val_metrics["qwk5"]
        val_mae = val_metrics["mae_bcs"]
        val_mgmt_f1 = val_metrics["macro_f1_3_mgmt"]
        val_low_recall = val_metrics["lean_recall_mgmt"]
        val_low_pr_auc = val_metrics["low_pr_auc"]
        val_high_pr_auc = val_metrics["high_pr_auc"]

        if np.isnan(val_qwk):
            val_qwk = -1.0
        if np.isnan(val_low_pr_auc):
            val_low_pr_auc = 0.0
        if np.isnan(val_high_pr_auc):
            val_high_pr_auc = 0.0

        # 保存标准更关心 Lean 是否被救起来，同时不能完全丢掉 ordinal 质量
        val_score = (
            val_qwk
            - 0.05 * val_mae
            + 0.30 * val_mgmt_f1
            + 0.20 * val_low_recall
            + 0.10 * val_low_pr_auc
            + 0.05 * val_high_pr_auc
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_val_qwk = val_metrics["qwk5"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        elapsed = time.time() - start_time

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Fold {fold} | Epoch {epoch:03d}/{EPOCHS} "
                f"| Loss {total_loss / len(train_loader):.4f} "
                f"| Ord {total_ord / len(train_loader):.4f} "
                f"| Low {total_low / len(train_loader):.4f} "
                f"| High {total_high / len(train_loader):.4f} "
                f"| SupCon {total_supcon / len(train_loader):.4f} "
                f"| Val QWK {val_metrics['qwk5']:.4f} "
                f"| Val MAE {val_metrics['mae_bcs']:.4f} "
                f"| Val MgmtF1 {val_metrics['macro_f1_3_mgmt']:.4f} "
                f"| Val LeanR {val_metrics['lean_recall_mgmt']:.4f} "
                f"| Time {elapsed:.1f}s"
            )

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    model.load_state_dict(best_state)

    test_metrics, pred_rows, y_true5, y_pred5, y_true3, y_pred3_mgmt, y_pred3_ord = evaluate(model, test_loader)

    save_confusion_matrices(fold, y_true5, y_pred5, y_true3, y_pred3_mgmt)

    model_path = OUT_DIR / f"fold{fold}_lowhigh_rgb_depth_comil_depthcrop07_best.pth"
    torch.save(model.state_dict(), model_path)

    fold_row = {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_qwk": best_val_qwk,
        "best_val_score": best_val_score,
        "test_bags": len(test_bags),

        "acc5": test_metrics["acc5"],
        "balanced_acc5": test_metrics["balanced_acc5"],
        "macro_f1_5": test_metrics["macro_f1_5"],
        "weighted_f1_5": test_metrics["weighted_f1_5"],
        "qwk5": test_metrics["qwk5"],
        "mae_bcs": test_metrics["mae_bcs"],
        "within_one_acc": test_metrics["within_one_acc"],

        "acc3_ordinal": test_metrics["acc3_ordinal"],
        "macro_f1_3_ordinal": test_metrics["macro_f1_3_ordinal"],
        "qwk3_ordinal": test_metrics["qwk3_ordinal"],

        "acc3_mgmt": test_metrics["acc3_mgmt"],
        "macro_f1_3_mgmt": test_metrics["macro_f1_3_mgmt"],
        "qwk3_mgmt": test_metrics["qwk3_mgmt"],
        "lean_precision_mgmt": test_metrics["lean_precision_mgmt"],
        "lean_recall_mgmt": test_metrics["lean_recall_mgmt"],

        "low_precision": test_metrics["low_precision"],
        "low_recall": test_metrics["low_recall"],
        "low_specificity": test_metrics["low_specificity"],
        "low_pr_auc": test_metrics["low_pr_auc"],
        "low_roc_auc": test_metrics["low_roc_auc"],

        "high_precision_ordinal": test_metrics["high_precision_ordinal"],
        "high_recall_ordinal": test_metrics["high_recall_ordinal"],
        "high_specificity_ordinal": test_metrics["high_specificity_ordinal"],

        "high_precision_branch": test_metrics["high_precision_branch"],
        "high_recall_branch": test_metrics["high_recall_branch"],
        "high_specificity_branch": test_metrics["high_specificity_branch"],
        "high_pr_auc": test_metrics["high_pr_auc"],
        "high_roc_auc": test_metrics["high_roc_auc"],

        "attn_entropy_mean": test_metrics["attn_entropy_mean"],
        "attn_entropy_std": test_metrics["attn_entropy_std"],
    }

    for r in pred_rows:
        r["fold"] = fold

    print("\nAnimal-wise RGB-Depth CO-MIL LowHigh fold test result")
    print("-" * 100)
    print(f"QWK5:                    {fold_row['qwk5']:.4f}")
    print(f"MAE:                     {fold_row['mae_bcs']:.4f}")
    print(f"Within-one Acc:          {fold_row['within_one_acc']:.4f}")
    print(f"Macro-F1 5-class:        {fold_row['macro_f1_5']:.4f}")
    print(f"Mgmt Macro-F1:           {fold_row['macro_f1_3_mgmt']:.4f}")
    print(f"Lean Precision Mgmt:     {fold_row['lean_precision_mgmt']:.4f}")
    print(f"Lean Recall Mgmt:        {fold_row['lean_recall_mgmt']:.4f}")
    print(f"Low PR-AUC:              {fold_row['low_pr_auc']:.4f}")
    print(f"High PR-AUC:             {fold_row['high_pr_auc']:.4f}")

    return fold_row, pred_rows


def summarize(rows):
    keys = [
        "acc5",
        "balanced_acc5",
        "macro_f1_5",
        "weighted_f1_5",
        "qwk5",
        "mae_bcs",
        "within_one_acc",

        "acc3_ordinal",
        "macro_f1_3_ordinal",
        "qwk3_ordinal",

        "acc3_mgmt",
        "macro_f1_3_mgmt",
        "qwk3_mgmt",
        "lean_precision_mgmt",
        "lean_recall_mgmt",

        "low_precision",
        "low_recall",
        "low_specificity",
        "low_pr_auc",
        "low_roc_auc",

        "high_precision_ordinal",
        "high_recall_ordinal",
        "high_specificity_ordinal",

        "high_precision_branch",
        "high_recall_branch",
        "high_specificity_branch",
        "high_pr_auc",
        "high_roc_auc",

        "attn_entropy_mean",
        "attn_entropy_std",
    ]

    out = []

    for k in keys:
        vals = np.array([float(r[k]) for r in rows], dtype=float)
        vals = vals[~np.isnan(vals)]

        out.append({
            "metric": k,
            "mean": float(np.mean(vals)) if len(vals) else np.nan,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        })

    return out


def save_aggregate_confusions(pred_rows):
    y_true5 = [int(r["true_label5"]) for r in pred_rows]
    y_pred5 = [int(r["pred_label5"]) for r in pred_rows]

    y_true3 = [int(r["true_label3"]) for r in pred_rows]
    y_pred3_mgmt = [int(r["pred_label3_mgmt"]) for r in pred_rows]

    cm5 = confusion_matrix(y_true5, y_pred5, labels=[0, 1, 2, 3, 4])
    cm3 = confusion_matrix(y_true3, y_pred3_mgmt, labels=[0, 1, 2])

    cm5_norm = cm5.astype(float) / np.maximum(cm5.sum(axis=1, keepdims=True), 1)
    cm3_norm = cm3.astype(float) / np.maximum(cm3.sum(axis=1, keepdims=True), 1)

    report_path = OUT_DIR / "05_lowhigh_aggregate_confusion_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Aggregate 5-class confusion matrix\n")
        f.write(str(cm5) + "\n\n")
        f.write("Row-normalized 5-class confusion matrix\n")
        f.write(str(np.round(cm5_norm, 3)) + "\n\n")
        f.write("5-class classification report\n")
        f.write(classification_report(y_true5, y_pred5, target_names=["BCS2", "BCS3", "BCS4", "BCS5", "BCS6"], zero_division=0))
        f.write("\n\nAggregate 3-class management confusion matrix\n")
        f.write(str(cm3) + "\n\n")
        f.write("Row-normalized 3-class management confusion matrix\n")
        f.write(str(np.round(cm3_norm, 3)) + "\n\n")
        f.write("3-class classification report\n")
        f.write(classification_report(y_true3, y_pred3_mgmt, target_names=["Lean", "Ideal", "High"], zero_division=0))

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm5,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["BCS2", "BCS3", "BCS4", "BCS5", "BCS6"],
        yticklabels=["BCS2", "BCS3", "BCS4", "BCS5", "BCS6"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Aggregate 5-class BCS Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_lowhigh_aggregate_cm5_counts.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm3,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Lean", "Ideal", "High"],
        yticklabels=["Lean", "Ideal", "High"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Aggregate Management Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_lowhigh_aggregate_mgmt_cm3_counts.png", dpi=300)
    plt.close()

    print("\nAggregate confusion report saved:")
    print(report_path)


def main():
    set_seed(SEED)

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Cannot find {CSV_PATH}")

    print("=" * 120)
    print("Animal-wise RGB-Depth Dual-Branch CO-MIL + Low/High branches")
    print(f"CSV: {CSV_PATH}")
    print(f"OUT: {OUT_DIR}")
    print(f"DEVICE: {DEVICE}")
    print(f"N_SPLITS: {N_SPLITS}")
    print(f"FRAMES_PER_BAG: {FRAMES_PER_BAG}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"NUM_WORKERS: {NUM_WORKERS}")
    print(f"PREFETCH_FACTOR: {PREFETCH_FACTOR}")
    print(f"AMP: {AMP}")
    print(f"DEPTH_CENTER_CROP_RATIO: {DEPTH_CENTER_CROP_RATIO}")
    print(f"LOW_THR: {LOW_THR}")
    print(f"HIGH_THR: {HIGH_THR}")
    print("=" * 120)

    rows = read_csv(CSV_PATH)
    bags = build_bags(rows)

    verify_animalwise_no_leakage(bags)

    print(f"Total bags: {len(bags)}")
    print(f"Total frames: {sum(b['frame_count'] for b in bags)}")
    print(f"Total animals: {len(set(b['animal_id'] for b in bags))}")
    print("Bag-level BCS distribution:", Counter(b["bcs_raw"] for b in bags))
    print("Management distribution:", Counter(b["label3"] for b in bags))

    fold_rows = []
    all_preds = []

    for fold in range(N_SPLITS):
        fr, preds = train_one_fold(fold, bags)
        fold_rows.append(fr)
        all_preds.extend(preds)

        partial_csv = OUT_DIR / "05_lowhigh_fold_metrics_partial.csv"
        write_csv(partial_csv, fold_rows, list(fold_rows[0].keys()))

    fold_csv = OUT_DIR / "05_lowhigh_fold_metrics.csv"
    write_csv(fold_csv, fold_rows, list(fold_rows[0].keys()))

    pred_csv = OUT_DIR / "05_lowhigh_predictions.csv"
    write_csv(pred_csv, all_preds, list(all_preds[0].keys()))

    summary_rows = summarize(fold_rows)

    summary_csv = OUT_DIR / "05_lowhigh_mean_std.csv"
    write_csv(summary_csv, summary_rows, ["metric", "mean", "std"])

    save_aggregate_confusions(all_preds)

    print("\n" + "=" * 120)
    print("Animal-wise RGB-Depth CO-MIL LowHigh 3-fold mean ± std")
    print("=" * 120)

    for r in summary_rows:
        print(f"{r['metric']:<28}: {r['mean']:.4f} ± {r['std']:.4f}")

    print("\nSaved:")
    print(fold_csv)
    print(summary_csv)
    print(pred_csv)

    print("\nTraining finished successfully.")


if __name__ == "__main__":
    main()