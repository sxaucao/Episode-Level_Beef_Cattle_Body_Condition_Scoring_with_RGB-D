import csv
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy import ndimage
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

BASE_PATH = Path(r"../dataset")
FRAME_CSV = BASE_PATH / "01_animalwise_3fold.csv"

OUT_DIR = BASE_PATH / "08e_quality_controlled_dorsal_angle"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESS_ALL_FRAMES = True
MAX_FRAMES_PER_BAG = 12

DEPTH_CENTER_CROP_RATIO = 0.70

CENTER_ROI_X1 = 0.22
CENTER_ROI_X2 = 0.78

FOREGROUND_PERCENTILES = [35, 40, 45, 50, 55, 60]
MIN_MASK_AREA_RATIO = 0.012

MIN_COMPONENT_AREA = 80
MIN_COMPONENT_WIDTH_RATIO = 0.08
MAX_CENTER_OFFSET_RATIO = 0.18

MAX_AXIS_TILT_DEG_FOR_GOOD = 35.0

WIDTH_SMOOTH_SIGMA = 2.0
TRUNK_WIDTH_RATIO = 0.68
MIN_TRUNK_RUN_ROWS = 14
MIN_TRUNK_LEN_RATIO_FOR_GOOD = 0.22

END_FRACTION = 0.12
POINTED_END_RATIO = 0.55
TAPER_STRENGTH_RATIO = 0.25
BOTTOM_GAP_LARGE = 0.10
BOTTOM_GAP_SMALL = 0.03

N_PROFILE_BANDS = 9
RIDGE_CENTER_LOW = 0.35
RIDGE_CENTER_HIGH = 0.65
MIN_VALID_PROFILES_FOR_GOOD = 3

MAX_CENTERLINE_STD_RATIO = 0.075
MAX_CENTERLINE_RANGE_RATIO = 0.22

MAX_MULTI_PEAK_COUNT = 2
MIN_RIDGE_PROMINENCE = 0.015

MAKE_DEBUG_FIGURES = True
SAMPLES_PER_BCS = 10

LABEL5_TO_RAW = {
    0: 2,
    1: 3,
    2: 4,
    3: 5,
    4: 6,
}


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def robust_percentile(values, q):
    values = np.asarray(values)
    if values.size > 50000:
        step = max(1, values.size // 50000)
        values = values[::step]
    return float(np.percentile(values, q))


def center_crop(arr, ratio):
    if ratio >= 0.999:
        return arr

    h, w = arr.shape[:2]
    new_h = int(h * ratio)
    new_w = int(w * ratio)

    y1 = max((h - new_h) // 2, 0)
    x1 = max((w - new_w) // 2, 0)

    return arr[y1:y1 + new_h, x1:x1 + new_w]


def apply_center_roi(mask):
    h, w = mask.shape
    x1 = int(w * CENTER_ROI_X1)
    x2 = int(w * CENTER_ROI_X2)

    roi = np.zeros_like(mask, dtype=bool)
    roi[:, x1:x2] = True

    return mask & roi


def normalize_closeness(depth_raw):

    arr = depth_raw.astype(np.float32)
    valid = arr > 0

    if valid.sum() <= 100:
        return np.zeros_like(arr, dtype=np.float32), valid

    vals = arr[valid]
    lo = robust_percentile(vals, 1)
    hi = robust_percentile(vals, 99)

    if hi <= lo:
        hi = lo + 1.0

    arr_clip = np.clip(arr, lo, hi)
    norm_depth = (arr_clip - lo) / (hi - lo)
    norm_depth[~valid] = 1.0

    closeness = 1.0 - norm_depth
    closeness = np.nan_to_num(closeness, nan=0.0, posinf=1.0, neginf=0.0)
    closeness = np.clip(closeness, 0.0, 1.0)

    return closeness, valid


def choose_best_body_component(mask):

    if mask.sum() == 0:
        return mask.astype(bool)

    if not HAS_SCIPY:
        return mask.astype(bool)

    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((7, 7)))
    mask = ndimage.binary_fill_holes(mask)

    labeled, n = ndimage.label(mask)

    if n == 0:
        return np.zeros_like(mask, dtype=bool)

    h, w = mask.shape
    cx0 = w / 2.0

    best_lab = None
    best_score = -1.0

    for lab in range(1, n + 1):
        comp = labeled == lab
        area = int(comp.sum())

        if area < MIN_COMPONENT_AREA:
            continue

        ys, xs = np.where(comp)

        if xs.size == 0:
            continue

        x_min, x_max = xs.min(), xs.max()
        bbox_w = x_max - x_min + 1

        if bbox_w / w < MIN_COMPONENT_WIDTH_RATIO:
            continue

        cx = xs.mean()
        center_offset = abs(cx - cx0) / w

        if center_offset > MAX_CENTER_OFFSET_RATIO:
            continue

        central_weight = np.exp(-8.0 * center_offset)
        width_weight = min(1.0, bbox_w / (0.35 * w))
        score = area * central_weight * (0.5 + 0.5 * width_weight)

        if score > best_score:
            best_score = score
            best_lab = lab

    if best_lab is None:
        return np.zeros_like(mask, dtype=bool)

    return labeled == best_lab


def segment_body_mask(depth_raw):
    depth = depth_raw.astype(np.float32)
    valid = depth > 0

    if valid.sum() <= 100:
        return np.zeros_like(valid, dtype=bool)

    valid_center = apply_center_roi(valid)

    if valid_center.sum() <= 100:
        return np.zeros_like(valid, dtype=bool)

    vals = depth[valid_center]
    best_mask = None

    for p in FOREGROUND_PERCENTILES:
        thr = robust_percentile(vals, p)

        mask = valid_center & (depth <= thr)
        mask = choose_best_body_component(mask)

        area_ratio = mask.sum() / mask.size

        if area_ratio >= MIN_MASK_AREA_RATIO:
            return mask.astype(bool)

        best_mask = mask

    if best_mask is None:
        return np.zeros_like(valid, dtype=bool)

    return best_mask.astype(bool)


def pca_body_axis(mask):

    ys, xs = np.where(mask)

    if xs.size < 50:
        return None

    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    coords = coords - coords.mean(axis=0, keepdims=True)

    cov = np.cov(coords.T)

    try:
        vals, vecs = np.linalg.eigh(cov)
    except Exception:
        return None

    main_vec = vecs[:, np.argmax(vals)]
    vx, vy = float(main_vec[0]), float(main_vec[1])

    angle_deg = np.degrees(np.arctan2(vy, vx))
    axis_angle = angle_deg % 180.0

    # distance from vertical axis
    tilt_abs = abs(axis_angle - 90.0)

    return axis_angle, tilt_abs


def rotate_mask_and_depth(mask, closeness, rotate_deg):

    if not HAS_SCIPY:
        return mask, closeness

    rotated_closeness = ndimage.rotate(
        closeness,
        rotate_deg,
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
    )

    rotated_mask = ndimage.rotate(
        mask.astype(np.uint8),
        rotate_deg,
        reshape=False,
        order=0,
        mode="constant",
        cval=0,
    ).astype(bool)

    rotated_mask = choose_best_body_component(rotated_mask)

    return rotated_mask, rotated_closeness


def rotate_to_vertical(mask, closeness):

    pca = pca_body_axis(mask)

    if pca is None:
        return mask, closeness, 0.0, 999.0, False

    axis_angle, tilt_before = pca

    if not HAS_SCIPY:
        return mask, closeness, 0.0, tilt_before, False

    candidate_angles = [
        90.0 - axis_angle,
        axis_angle - 90.0,
    ]

    best = None

    for rot in candidate_angles:
        rmask, rclose = rotate_mask_and_depth(mask, closeness, rot)

        pca_after = pca_body_axis(rmask)

        if pca_after is None:
            tilt_after = 999.0
        else:
            _, tilt_after = pca_after

        area = int(rmask.sum())

        score = tilt_after - 0.00001 * area

        if best is None or score < best["score"]:
            best = {
                "score": score,
                "rotate_deg": rot,
                "tilt_after": tilt_after,
                "mask": rmask,
                "closeness": rclose,
            }

    if best is None:
        return mask, closeness, 0.0, tilt_before, False

    return (
        best["mask"],
        best["closeness"],
        best["rotate_deg"],
        best["tilt_after"],
        True,
    )


def row_width_profile(mask):
    h, w = mask.shape
    widths = np.zeros(h, dtype=np.float32)
    x_left = np.full(h, np.nan, dtype=np.float32)
    x_right = np.full(h, np.nan, dtype=np.float32)
    x_center = np.full(h, np.nan, dtype=np.float32)

    for y in range(h):
        xs = np.where(mask[y, :])[0]
        if xs.size > 0:
            widths[y] = xs.max() - xs.min() + 1
            x_left[y] = xs.min()
            x_right[y] = xs.max()
            x_center[y] = 0.5 * (xs.min() + xs.max())

    return widths, x_left, x_right, x_center


def smooth_widths(widths):
    if HAS_SCIPY and len(widths) >= 7:
        return ndimage.gaussian_filter1d(widths, sigma=WIDTH_SMOOTH_SIGMA)
    return widths.copy()


def contiguous_runs(boolean_array):
    ys = np.where(boolean_array)[0]

    if ys.size == 0:
        return []

    runs = []
    start = ys[0]
    prev = ys[0]

    for y in ys[1:]:
        if y == prev + 1:
            prev = y
        else:
            runs.append((int(start), int(prev)))
            start = y
            prev = y

    runs.append((int(start), int(prev)))
    return runs


def estimate_phase_and_trunk(mask):
    widths, x_left, x_right, x_center = row_width_profile(mask)
    widths_smooth = smooth_widths(widths)

    nz = np.where(widths_smooth > 0)[0]

    if nz.size < MIN_TRUNK_RUN_ROWS:
        return None

    y_min = int(nz.min())
    y_max = int(nz.max())
    body_len = y_max - y_min + 1
    h, w = mask.shape

    if body_len < MIN_TRUNK_RUN_ROWS:
        return None

    body_widths = widths_smooth[y_min:y_max + 1]
    max_width = float(np.max(body_widths))

    if max_width <= 0:
        return None

    n_end = max(3, int(END_FRACTION * body_len))

    top_region = body_widths[:n_end]
    bottom_region = body_widths[-n_end:]

    top_width = float(np.mean(top_region[top_region > 0])) if np.any(top_region > 0) else 0.0
    bottom_width = float(np.mean(bottom_region[bottom_region > 0])) if np.any(bottom_region > 0) else 0.0

    top_width_ratio = top_width / (max_width + 1e-6)
    bottom_width_ratio = bottom_width / (max_width + 1e-6)

    top_gap_ratio = y_min / h
    bottom_gap_ratio = (h - 1 - y_max) / h

    # taper slope top
    first_q_end = int(max(3, 0.25 * body_len))
    first_q = body_widths[:first_q_end]

    if first_q.size >= 4:
        x = np.linspace(0, 1, first_q.size)
        try:
            slope_top = float(np.polyfit(x, first_q / (max_width + 1e-6), 1)[0])
        except Exception:
            slope_top = 0.0
    else:
        slope_top = 0.0

    # taper slope bottom
    last_q_start = int(max(0, 0.75 * body_len))
    last_q = body_widths[last_q_start:]

    if last_q.size >= 4:
        x = np.linspace(0, 1, last_q.size)
        try:
            slope_bottom = float(np.polyfit(x, last_q / (max_width + 1e-6), 1)[0])
        except Exception:
            slope_bottom = 0.0
    else:
        slope_bottom = 0.0

    pointed_top = (
        top_width_ratio <= POINTED_END_RATIO
        and slope_top >= TAPER_STRENGTH_RATIO
    )

    pointed_bottom = (
        bottom_width_ratio <= POINTED_END_RATIO
        and slope_bottom <= -TAPER_STRENGTH_RATIO
    )

    plateau = widths_smooth >= TRUNK_WIDTH_RATIO * max_width
    plateau[:y_min] = False
    plateau[y_max + 1:] = False

    runs = [
        (a, b)
        for a, b in contiguous_runs(plateau)
        if (b - a + 1) >= MIN_TRUNK_RUN_ROWS
    ]

    if not runs:
        return None

    best_run = None
    best_score = -1.0

    for a, b in runs:
        run_len = b - a + 1
        score = float(widths_smooth[a:b + 1].mean()) * run_len
        if score > best_score:
            best_score = score
            best_run = (a, b)

    trunk_a, trunk_b = best_run
    trunk_len = trunk_b - trunk_a + 1
    trunk_len_ratio = trunk_len / max(body_len, 1)

    if pointed_top and not pointed_bottom:
        phase = "head_neck_trunk"
        target_rel = 0.62
    elif pointed_bottom and not pointed_top:
        phase = "trunk_head_neck_reversed"
        target_rel = 0.38
    elif bottom_gap_ratio >= BOTTOM_GAP_LARGE and not pointed_top:
        phase = "posterior_or_trunk_partial"
        target_rel = 0.42
    elif bottom_gap_ratio <= BOTTOM_GAP_SMALL and not pointed_top:
        phase = "entry_or_trunk_with_rump"
        target_rel = 0.50
    else:
        phase = "trunk_dominant"
        target_rel = 0.50

    target_y = int(round(trunk_a + target_rel * max(trunk_len - 1, 0)))

    window_half = max(4, int(0.18 * trunk_len))
    sel_a = max(trunk_a, target_y - window_half)
    sel_b = min(trunk_b, target_y + window_half)
    selected_rows = list(range(sel_a, sel_b + 1))

    # centerline stability in selected trunk rows
    selected_centers = x_center[selected_rows]
    selected_centers = selected_centers[~np.isnan(selected_centers)]

    if selected_centers.size >= 3:
        center_std_ratio = float(np.std(selected_centers) / (max_width + 1e-6))
        center_range_ratio = float((np.max(selected_centers) - np.min(selected_centers)) / (max_width + 1e-6))
    else:
        center_std_ratio = np.nan
        center_range_ratio = np.nan

    return {
        "widths": widths,
        "widths_smooth": widths_smooth,
        "x_left": x_left,
        "x_right": x_right,
        "x_center": x_center,

        "y_min": y_min,
        "y_max": y_max,
        "body_len": body_len,
        "max_width": max_width,

        "top_width_ratio": top_width_ratio,
        "bottom_width_ratio": bottom_width_ratio,
        "top_gap_ratio": top_gap_ratio,
        "bottom_gap_ratio": bottom_gap_ratio,
        "slope_top": slope_top,
        "slope_bottom": slope_bottom,
        "pointed_top": int(pointed_top),
        "pointed_bottom": int(pointed_bottom),

        "trunk_a": trunk_a,
        "trunk_b": trunk_b,
        "trunk_len": trunk_len,
        "trunk_len_ratio": trunk_len_ratio,

        "target_y": target_y,
        "selected_rows": selected_rows,

        "centerline_std_ratio": center_std_ratio,
        "centerline_range_ratio": center_range_ratio,

        "phase": phase,
    }


def fit_line_slope(xs, ys):
    if len(xs) < 4:
        return None, None

    try:
        m, b = np.polyfit(xs, ys, 1)
        return float(m), float(b)
    except Exception:
        return None, None


def angle_from_slopes(m_left, m_right):
    v1 = np.array([-1.0, -m_left], dtype=np.float32)
    v2 = np.array([1.0, m_right], dtype=np.float32)

    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
    cosang = float(np.dot(v1, v2) / denom)
    cosang = max(-1.0, min(1.0, cosang))

    return float(np.degrees(np.arccos(cosang)))


def count_prominent_peaks(profile_y):
    if len(profile_y) < 5:
        return 0

    y = np.asarray(profile_y, dtype=np.float32)
    max_y = float(np.max(y))
    threshold = max_y - 0.08

    count = 0

    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] >= threshold:
            count += 1

    return count


def extract_profile_angle(mask, closeness, yc):
    h, w = mask.shape

    ys, xs = np.where(mask)

    if xs.size < 50:
        return None

    y_min, y_max = int(ys.min()), int(ys.max())
    body_h = y_max - y_min + 1

    band_half = max(2, int(0.015 * body_h))
    y1 = max(int(yc) - band_half, 0)
    y2 = min(int(yc) + band_half + 1, h)

    band_mask = mask[y1:y2, :]

    if band_mask.sum() < 20:
        return None

    _, bx = np.where(band_mask)

    if bx.size < 15:
        return None

    x_min = int(bx.min())
    x_max = int(bx.max())
    width = x_max - x_min + 1

    if width < 15:
        return None

    profile_x = []
    profile_y = []

    for x in range(x_min, x_max + 1):
        col_mask = band_mask[:, x]

        if col_mask.sum() == 0:
            continue

        vals = closeness[y1:y2, x][col_mask]

        if vals.size == 0:
            continue

        profile_x.append(x)
        profile_y.append(float(np.median(vals)))

    if len(profile_x) < 15:
        return None

    profile_x = np.array(profile_x, dtype=np.float32)
    profile_y = np.array(profile_y, dtype=np.float32)

    if HAS_SCIPY and len(profile_y) >= 7:
        profile_y_smooth = ndimage.gaussian_filter1d(profile_y, sigma=1.2)
    else:
        profile_y_smooth = profile_y.copy()

    ridge_idx = int(np.argmax(profile_y_smooth))
    ridge_x = float(profile_x[ridge_idx])
    ridge_y = float(profile_y_smooth[ridge_idx])

    rel_pos = (ridge_x - x_min) / max(width, 1)

    if not (RIDGE_CENTER_LOW <= rel_pos <= RIDGE_CENTER_HIGH):
        return None

    # Reject multi-peak profiles, common in head-turn/horn interference.
    peak_count = count_prominent_peaks(profile_y_smooth)

    if peak_count > MAX_MULTI_PEAK_COUNT:
        return None

    left_edge = profile_y_smooth[:max(2, len(profile_y_smooth) // 5)]
    right_edge = profile_y_smooth[-max(2, len(profile_y_smooth) // 5):]
    edge_base = float(np.median(np.concatenate([left_edge, right_edge])))
    ridge_prominence = ridge_y - edge_base

    if ridge_prominence < MIN_RIDGE_PROMINENCE:
        return None

    left_inner = ridge_x - 0.05 * width
    left_outer = ridge_x - 0.40 * width
    right_inner = ridge_x + 0.05 * width
    right_outer = ridge_x + 0.40 * width

    left_sel = (profile_x >= left_outer) & (profile_x <= left_inner)
    right_sel = (profile_x >= right_inner) & (profile_x <= right_outer)

    if left_sel.sum() < 4:
        left_sel = (profile_x >= x_min) & (profile_x < ridge_x)
    if right_sel.sum() < 4:
        right_sel = (profile_x > ridge_x) & (profile_x <= x_max)

    if left_sel.sum() < 4 or right_sel.sum() < 4:
        return None

    x_left_norm = (profile_x[left_sel] - ridge_x) / max(width, 1.0)
    y_left = profile_y_smooth[left_sel]

    x_right_norm = (profile_x[right_sel] - ridge_x) / max(width, 1.0)
    y_right = profile_y_smooth[right_sel]

    m_left, b_left = fit_line_slope(x_left_norm, y_left)
    m_right, b_right = fit_line_slope(x_right_norm, y_right)

    if m_left is None or m_right is None:
        return None

    angle = angle_from_slopes(m_left, m_right)

    if not (10 <= angle <= 180):
        return None

    return {
        "angle": angle,
        "yc": int(yc),
        "y1": int(y1),
        "y2": int(y2),
        "x_min": int(x_min),
        "x_max": int(x_max),
        "width": float(width),
        "ridge_x": ridge_x,
        "ridge_y": ridge_y,
        "ridge_rel_pos": float(rel_pos),
        "ridge_prominence": float(ridge_prominence),
        "peak_count": int(peak_count),

        "profile_x": profile_x,
        "profile_y": profile_y,
        "profile_y_smooth": profile_y_smooth,

        "left_sel": left_sel,
        "right_sel": right_sel,
        "m_left": m_left,
        "b_left": b_left,
        "m_right": m_right,
        "b_right": b_right,
    }


def extract_qc_angles_from_frame(depth_path, return_debug=False):
    raw = np.load(depth_path).astype(np.float32)
    raw = center_crop(raw, DEPTH_CENTER_CROP_RATIO)

    closeness, valid = normalize_closeness(raw)
    mask = segment_body_mask(raw)

    bad_reasons = []

    if mask.sum() < 80:
        bad_reasons.append("no_body_mask")
        if return_debug:
            return [], None, bad_reasons
        return [], bad_reasons

    rotated_mask, rotated_closeness, rotate_deg, axis_tilt_abs, rotated = rotate_to_vertical(mask, closeness)

    if rotated_mask.sum() < 80:
        bad_reasons.append("empty_after_rotation")
        if return_debug:
            return [], None, bad_reasons
        return [], bad_reasons

    if axis_tilt_abs > MAX_AXIS_TILT_DEG_FOR_GOOD:
        bad_reasons.append("excessive_axis_tilt")

    phase_info = estimate_phase_and_trunk(rotated_mask)

    if phase_info is None:
        bad_reasons.append("no_valid_trunk_phase")
        if return_debug:
            debug = {
                "raw": raw,
                "closeness": closeness,
                "mask": mask,
                "rotated_mask": rotated_mask,
                "rotated_closeness": rotated_closeness,
                "rotate_deg": rotate_deg,
                "axis_tilt_abs": axis_tilt_abs,
                "phase_info": None,
                "profiles": [],
            }
            return [], debug, bad_reasons
        return [], bad_reasons

    if phase_info["trunk_len_ratio"] < MIN_TRUNK_LEN_RATIO_FOR_GOOD:
        bad_reasons.append("short_trunk_plateau")

    if not np.isnan(phase_info["centerline_std_ratio"]):
        if phase_info["centerline_std_ratio"] > MAX_CENTERLINE_STD_RATIO:
            bad_reasons.append("unstable_centerline_std")

    if not np.isnan(phase_info["centerline_range_ratio"]):
        if phase_info["centerline_range_ratio"] > MAX_CENTERLINE_RANGE_RATIO:
            bad_reasons.append("unstable_centerline_range")

    selected_rows = phase_info["selected_rows"]

    if len(selected_rows) <= N_PROFILE_BANDS:
        y_centers = selected_rows
    else:
        idxs = np.linspace(0, len(selected_rows) - 1, N_PROFILE_BANDS).astype(int)
        y_centers = [selected_rows[int(i)] for i in idxs]

    profiles = []

    for yc in y_centers:
        prof = extract_profile_angle(rotated_mask, rotated_closeness, yc)
        if prof is not None:
            profiles.append(prof)

    angles = [p["angle"] for p in profiles]

    if len(profiles) < MIN_VALID_PROFILES_FOR_GOOD:
        bad_reasons.append("too_few_valid_profiles")

    if len(angles) == 0:
        bad_reasons.append("no_valid_angle")

    if return_debug:
        debug = {
            "raw": raw,
            "closeness": closeness,
            "mask": mask,
            "rotated_mask": rotated_mask,
            "rotated_closeness": rotated_closeness,
            "rotate_deg": rotate_deg,
            "axis_tilt_abs": axis_tilt_abs,
            "phase_info": phase_info,
            "profiles": profiles,
        }
        return angles, debug, bad_reasons

    return angles, bad_reasons


def build_bags(rows):
    bags = defaultdict(list)
    for r in rows:
        bags[r["bag_id"]].append(r)
    return bags


def sample_rows_evenly(rows, max_n):
    rows = sorted(rows, key=lambda x: str(x.get("frame_key", "")))

    if max_n is None or len(rows) <= max_n:
        return rows

    idxs = np.linspace(0, len(rows) - 1, max_n).astype(int)
    return [rows[int(i)] for i in idxs]


def safe_stats(values):
    if len(values) == 0:
        return {
            "median": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
        }

    arr = np.array(values, dtype=np.float32)

    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def extract_all_angles():
    rows = read_csv(FRAME_CSV)
    bags = build_bags(rows)

    frame_rows = []
    bag_rows = []
    bad_counter = Counter()

    print("=" * 100)
    print("Step 08e: Quality-controlled dorsal ridge angle extraction")
    print(f"FRAME_CSV: {FRAME_CSV}")
    print(f"Total frames: {len(rows)}")
    print(f"Total bags: {len(bags)}")
    print(f"PROCESS_ALL_FRAMES: {PROCESS_ALL_FRAMES}")
    print("=" * 100)

    processed_frames = 0

    for bag_idx, (bag_id, brs) in enumerate(sorted(bags.items(), key=lambda x: x[0]), start=1):
        selected = brs if PROCESS_ALL_FRAMES else sample_rows_evenly(brs, MAX_FRAMES_PER_BAG)

        meta = selected[0]

        bag_all_angles = []
        bag_good_angles = []
        bag_good_frame_medians = []
        bag_phase_counts = Counter()

        valid_angle_frame_count = 0
        good_frame_count = 0

        for r in selected:
            processed_frames += 1

            angles, debug, bad_reasons = extract_qc_angles_from_frame(
                r["depth_npy_path"],
                return_debug=True,
            )

            for reason in bad_reasons:
                bad_counter[reason] += 1

            if debug is not None and debug["phase_info"] is not None:
                phase_info = debug["phase_info"]
                phase = phase_info["phase"]
                trunk_len_ratio = phase_info["trunk_len_ratio"]
                centerline_std_ratio = phase_info["centerline_std_ratio"]
                centerline_range_ratio = phase_info["centerline_range_ratio"]
                top_width_ratio = phase_info["top_width_ratio"]
                bottom_width_ratio = phase_info["bottom_width_ratio"]
                top_gap_ratio = phase_info["top_gap_ratio"]
                bottom_gap_ratio = phase_info["bottom_gap_ratio"]
                pointed_top = phase_info["pointed_top"]
                pointed_bottom = phase_info["pointed_bottom"]
                rotate_deg = debug["rotate_deg"]
                axis_tilt_abs = debug["axis_tilt_abs"]
                bag_phase_counts[phase] += 1
            else:
                phase = "invalid"
                trunk_len_ratio = np.nan
                centerline_std_ratio = np.nan
                centerline_range_ratio = np.nan
                top_width_ratio = np.nan
                bottom_width_ratio = np.nan
                top_gap_ratio = np.nan
                bottom_gap_ratio = np.nan
                pointed_top = np.nan
                pointed_bottom = np.nan
                rotate_deg = np.nan
                axis_tilt_abs = np.nan

            is_good = len(bad_reasons) == 0 and len(angles) >= MIN_VALID_PROFILES_FOR_GOOD

            if len(angles) > 0:
                valid_angle_frame_count += 1
                arr = np.array(angles, dtype=np.float32)
                frame_angle_median = float(np.median(arr))
                frame_angle_mean = float(np.mean(arr))
                frame_angle_std = float(np.std(arr))
                bag_all_angles.extend(angles)
            else:
                frame_angle_median = np.nan
                frame_angle_mean = np.nan
                frame_angle_std = np.nan

            if is_good:
                good_frame_count += 1
                bag_good_angles.extend(angles)
                bag_good_frame_medians.append(frame_angle_median)

            frame_rows.append({
                "bag_id": r["bag_id"],
                "episode_id": r.get("episode_id", ""),
                "animal_id": r.get("animal_id", ""),
                "fold_id": int(r["fold_id"]),
                "bcs_raw": int(r["bcs_raw"]),
                "label5": int(r["label5"]),
                "label3": int(r["label3"]),
                "frame_key": r.get("frame_key", ""),
                "rgb_path": r.get("rgb_path", ""),
                "depth_npy_path": r["depth_npy_path"],

                "is_good_angle_frame": int(is_good),
                "bad_reasons": ";".join(bad_reasons),

                "n_angles": len(angles),
                "frame_angle_median": frame_angle_median,
                "frame_angle_mean": frame_angle_mean,
                "frame_angle_std": frame_angle_std,

                "anatomical_phase": phase,
                "rotate_deg": rotate_deg,
                "axis_tilt_abs": axis_tilt_abs,
                "trunk_len_ratio": trunk_len_ratio,
                "centerline_std_ratio": centerline_std_ratio,
                "centerline_range_ratio": centerline_range_ratio,

                "top_width_ratio": top_width_ratio,
                "bottom_width_ratio": bottom_width_ratio,
                "top_gap_ratio": top_gap_ratio,
                "bottom_gap_ratio": bottom_gap_ratio,
                "pointed_top": pointed_top,
                "pointed_bottom": pointed_bottom,
            })

            if processed_frames % 200 == 0:
                print(f"Processed frames: {processed_frames}")

        all_stats = safe_stats(bag_all_angles)
        good_stats = safe_stats(bag_good_angles)
        good_frame_median_stats = safe_stats(bag_good_frame_medians)

        main_phase = bag_phase_counts.most_common(1)[0][0] if bag_phase_counts else "invalid"

        bag_rows.append({
            "bag_id": bag_id,
            "episode_id": meta.get("episode_id", ""),
            "animal_id": meta.get("animal_id", ""),
            "fold_id": int(meta["fold_id"]),
            "bcs_raw": int(meta["bcs_raw"]),
            "label5": int(meta["label5"]),
            "label3": int(meta["label3"]),

            "sampled_frame_count": len(selected),
            "valid_angle_frame_count": valid_angle_frame_count,
            "good_angle_frame_count": good_frame_count,
            "good_frame_ratio": good_frame_count / max(len(selected), 1),

            "main_phase": main_phase,

            "bag_all_angle_median": all_stats["median"],
            "bag_all_angle_mean": all_stats["mean"],
            "bag_all_angle_std": all_stats["std"],

            "bag_good_angle_median": good_stats["median"],
            "bag_good_angle_mean": good_stats["mean"],
            "bag_good_angle_std": good_stats["std"],
            "bag_good_angle_p10": good_stats["p10"],
            "bag_good_angle_p25": good_stats["p25"],
            "bag_good_angle_p75": good_stats["p75"],
            "bag_good_angle_p90": good_stats["p90"],

            "bag_good_frame_median_angle": good_frame_median_stats["median"],
            "bag_good_frame_mean_angle": good_frame_median_stats["mean"],
        })

        if bag_idx % 20 == 0:
            print(f"Processed bags: {bag_idx}/{len(bags)}")

    frame_df = pd.DataFrame(frame_rows)
    bag_df = pd.DataFrame(bag_rows)

    frame_csv = OUT_DIR / "08e_qc_frame_dorsal_angle.csv"
    bag_csv = OUT_DIR / "08e_qc_bag_dorsal_angle.csv"
    good_frame_csv = OUT_DIR / "08e_good_frame_index.csv"
    bad_reason_csv = OUT_DIR / "08e_bad_reason_counts.csv"

    frame_df.to_csv(frame_csv, index=False, encoding="utf-8-sig")
    bag_df.to_csv(bag_csv, index=False, encoding="utf-8-sig")
    frame_df[frame_df["is_good_angle_frame"] == 1].to_csv(good_frame_csv, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [{"reason": k, "count": v} for k, v in bad_counter.most_common()]
    ).to_csv(bad_reason_csv, index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print(frame_csv)
    print(bag_csv)
    print(good_frame_csv)
    print(bad_reason_csv)

    return frame_df, bag_df


def summarize_and_test(frame_df, bag_df):
    valid_bag = bag_df.dropna(subset=["bag_all_angle_median"]).copy()
    good_bag = bag_df.dropna(subset=["bag_good_angle_median"]).copy()

    good_frame = frame_df[frame_df["is_good_angle_frame"] == 1].dropna(subset=["frame_angle_median"]).copy()

    summary_all = valid_bag.groupby("bcs_raw")["bag_all_angle_median"].describe()
    summary_good = good_bag.groupby("bcs_raw")["bag_good_angle_median"].describe()
    summary_frame_good = good_frame.groupby("bcs_raw")["frame_angle_median"].describe()

    summary_all_csv = OUT_DIR / "08e_all_bag_angle_summary_by_bcs.csv"
    summary_good_csv = OUT_DIR / "08e_good_bag_angle_summary_by_bcs.csv"
    summary_frame_good_csv = OUT_DIR / "08e_good_frame_angle_summary_by_bcs.csv"

    summary_all.to_csv(summary_all_csv, encoding="utf-8-sig")
    summary_good.to_csv(summary_good_csv, encoding="utf-8-sig")
    summary_frame_good.to_csv(summary_frame_good_csv, encoding="utf-8-sig")

    quality_counts = frame_df.groupby(["bcs_raw", "is_good_angle_frame"]).size().reset_index(name="count")
    phase_counts = frame_df.groupby(["bcs_raw", "anatomical_phase"]).size().reset_index(name="count")

    quality_counts_csv = OUT_DIR / "08e_good_frame_counts_by_bcs.csv"
    phase_counts_csv = OUT_DIR / "08e_phase_counts_by_bcs.csv"

    quality_counts.to_csv(quality_counts_csv, index=False, encoding="utf-8-sig")
    phase_counts.to_csv(phase_counts_csv, index=False, encoding="utf-8-sig")

    report_path = OUT_DIR / "08e_qc_dorsal_angle_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Quality-controlled dorsal angle analysis\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Total frames: {len(frame_df)}\n")
        f.write(f"Good angle frames: {int(frame_df['is_good_angle_frame'].sum())}\n")
        f.write(f"Good frame ratio: {frame_df['is_good_angle_frame'].mean():.4f}\n")
        f.write(f"Valid bags all-angle: {len(valid_bag)}\n")
        f.write(f"Valid bags good-angle: {len(good_bag)}\n\n")

        f.write("All valid bag angle summary by BCS\n")
        f.write(str(summary_all) + "\n\n")

        f.write("Good-frame bag angle summary by BCS\n")
        f.write(str(summary_good) + "\n\n")

        f.write("Good-frame frame angle summary by BCS\n")
        f.write(str(summary_frame_good) + "\n\n")

        f.write("Quality counts by BCS\n")
        f.write(str(quality_counts) + "\n\n")

        f.write("Phase counts by BCS\n")
        f.write(str(phase_counts) + "\n\n")

        if HAS_SCIPY and len(valid_bag) > 5:
            rho, p = stats.spearmanr(
                valid_bag["bcs_raw"].astype(float),
                valid_bag["bag_all_angle_median"].astype(float),
                nan_policy="omit",
            )
            f.write(f"All valid bags Spearman BCS vs angle: rho={rho:.4f}, p={p:.6g}\n")

        if HAS_SCIPY and len(good_bag) > 5:
            rho, p = stats.spearmanr(
                good_bag["bcs_raw"].astype(float),
                good_bag["bag_good_angle_median"].astype(float),
                nan_policy="omit",
            )
            f.write(f"Good-frame bags Spearman BCS vs angle: rho={rho:.4f}, p={p:.6g}\n")

            lean = good_bag[good_bag["label3"] == 0]["bag_good_angle_median"].dropna().values
            high = good_bag[good_bag["label3"] == 2]["bag_good_angle_median"].dropna().values

            if len(lean) > 0 and len(high) > 0:
                mw = stats.mannwhitneyu(lean, high, alternative="two-sided")
                f.write(f"Good-frame Lean vs High Mann-Whitney U={mw.statistic:.4f}, p={mw.pvalue:.6g}\n")
                f.write(f"Lean median angle={np.median(lean):.4f}\n")
                f.write(f"High median angle={np.median(high):.4f}\n")

    print("\nSaved summaries and report:")
    print(summary_all_csv)
    print(summary_good_csv)
    print(summary_frame_good_csv)
    print(quality_counts_csv)
    print(phase_counts_csv)
    print(report_path)

    print("\nGood-frame bag angle summary by BCS:")
    print(summary_good)

    if HAS_SCIPY and len(good_bag) > 5:
        rho, p = stats.spearmanr(
            good_bag["bcs_raw"].astype(float),
            good_bag["bag_good_angle_median"].astype(float),
            nan_policy="omit",
        )
        print(f"\nGood-frame bags Spearman BCS vs angle: rho={rho:.4f}, p={p:.6g}")

    print(f"\nGood frame ratio: {frame_df['is_good_angle_frame'].mean():.4f}")

    return valid_bag, good_bag, good_frame


def make_plots(valid_bag, good_bag, good_frame):
    if len(valid_bag) > 0:
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=valid_bag, x="bcs_raw", y="bag_all_angle_median")
        sns.stripplot(data=valid_bag, x="bcs_raw", y="bag_all_angle_median",
                      color="black", alpha=0.45, size=3)
        plt.xlabel("BCS")
        plt.ylabel("Episode median dorsal angle, all valid profiles")
        plt.title("All valid dorsal angles by BCS")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "08e_all_bag_angle_by_bcs.png", dpi=300)
        plt.close()

    if len(good_bag) > 0:
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=good_bag, x="bcs_raw", y="bag_good_angle_median")
        sns.stripplot(data=good_bag, x="bcs_raw", y="bag_good_angle_median",
                      color="black", alpha=0.45, size=3)
        plt.xlabel("BCS")
        plt.ylabel("Episode median dorsal angle from good frames")
        plt.title("Quality-controlled dorsal angles by BCS")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "08e_good_bag_angle_by_bcs.png", dpi=300)
        plt.close()

        plt.figure(figsize=(7, 5))
        sns.boxplot(data=good_bag, x="label3", y="bag_good_angle_median")
        sns.stripplot(data=good_bag, x="label3", y="bag_good_angle_median",
                      color="black", alpha=0.45, size=3)
        plt.xlabel("Management class: 0=Lean, 1=Ideal, 2=High")
        plt.ylabel("Episode median dorsal angle from good frames")
        plt.title("Quality-controlled dorsal angles by management class")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "08e_good_bag_angle_by_management.png", dpi=300)
        plt.close()

    if len(good_frame) > 0:
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=good_frame, x="bcs_raw", y="frame_angle_median")
        plt.xlabel("BCS")
        plt.ylabel("Frame median dorsal angle from good frames")
        plt.title("Good-frame dorsal angles by BCS")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "08e_good_frame_angle_by_bcs.png", dpi=300)
        plt.close()

    print("\nSaved plots in:")
    print(OUT_DIR)


def choose_debug_samples(frame_df):
    # Prefer samples from both good and bad frames for checking.
    samples = []

    for bcs in [2, 3, 4, 5, 6]:
        sub_good = frame_df[
            (frame_df["bcs_raw"].astype(int) == bcs)
            & (frame_df["is_good_angle_frame"] == 1)
        ].dropna(subset=["frame_angle_median"]).copy()

        if len(sub_good) > 0:
            sub_good = sub_good.sort_values("frame_angle_median")
            idxs = np.linspace(0, len(sub_good) - 1, min(SAMPLES_PER_BCS // 2, len(sub_good))).astype(int)
            for idx in idxs:
                samples.append(sub_good.iloc[int(idx)].to_dict())

        sub_bad = frame_df[
            (frame_df["bcs_raw"].astype(int) == bcs)
            & (frame_df["is_good_angle_frame"] == 0)
        ].copy()

        if len(sub_bad) > 0:
            sub_bad = sub_bad.sort_values(["bag_id", "frame_key"])
            idxs = np.linspace(0, len(sub_bad) - 1, min(SAMPLES_PER_BCS // 2, len(sub_bad))).astype(int)
            for idx in idxs:
                samples.append(sub_bad.iloc[int(idx)].to_dict())

    return samples


def save_debug_figure(sample, out_path):
    angles, debug, bad_reasons = extract_qc_angles_from_frame(
        sample["depth_npy_path"],
        return_debug=True,
    )

    if debug is None:
        return False

    raw = debug["raw"]
    closeness = debug["closeness"]
    mask = debug["mask"]
    rmask = debug["rotated_mask"]
    rclose = debug["rotated_closeness"]
    phase_info = debug["phase_info"]
    profiles = debug["profiles"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    rgb_path = sample.get("rgb_path", "")

    if rgb_path and Path(rgb_path).exists():
        rgb = Image.open(rgb_path).convert("RGB")
        axes[0].imshow(rgb)
    else:
        axes[0].text(0.5, 0.5, "RGB not found", ha="center", va="center")
    axes[0].set_title("RGB")
    axes[0].axis("off")

    axes[1].imshow(closeness, cmap="viridis")
    axes[1].contour(mask.astype(float), levels=[0.5], colors="lime", linewidths=1)
    axes[1].set_title("Original depth closeness + mask")
    axes[1].axis("off")

    axes[2].imshow(rclose, cmap="gray")
    axes[2].contour(rmask.astype(float), levels=[0.5], colors="lime", linewidths=1)

    if phase_info is not None:
        for y in phase_info["selected_rows"][::max(1, len(phase_info["selected_rows"]) // 20)]:
            axes[2].axhline(y, color="cyan", linewidth=0.5, alpha=0.45)
        axes[2].axhline(phase_info["target_y"], color="red", linewidth=1.2)
    axes[2].set_title("Rotated mask + selected trunk rows")
    axes[2].axis("off")

    if phase_info is not None:
        widths = phase_info["widths"]
        widths_smooth = phase_info["widths_smooth"]

        axes[3].plot(widths, np.arange(len(widths)), color="gray", alpha=0.4, label="width")
        axes[3].plot(widths_smooth, np.arange(len(widths_smooth)), color="black", linewidth=1.5, label="smoothed")
        axes[3].invert_yaxis()
        axes[3].axhline(phase_info["trunk_a"], color="blue", linestyle="--", linewidth=1)
        axes[3].axhline(phase_info["trunk_b"], color="blue", linestyle="--", linewidth=1)
        axes[3].axhline(phase_info["target_y"], color="red", linewidth=1.2)
        axes[3].set_title(
            f"Width profile\nphase={phase_info['phase']}\ncenter_std={phase_info['centerline_std_ratio']:.3f}"
        )
        axes[3].set_xlabel("Width")
        axes[3].set_ylabel("Y")
        axes[3].legend(fontsize=7)
        axes[3].grid(alpha=0.3)
    else:
        axes[3].text(0.5, 0.5, "No phase", ha="center", va="center")
        axes[3].set_title("Width profile")

    if len(profiles) > 0:
        angle_arr = np.array([p["angle"] for p in profiles], dtype=np.float32)
        median_angle = float(np.median(angle_arr))
        best_idx = int(np.argmin(np.abs(angle_arr - median_angle)))
        p = profiles[best_idx]

        profile_x = p["profile_x"]
        profile_y = p["profile_y"]
        profile_y_smooth = p["profile_y_smooth"]

        ridge_x = p["ridge_x"]
        ridge_y = p["ridge_y"]
        width = p["width"]

        axes[4].plot(profile_x, profile_y, color="gray", alpha=0.5, label="raw")
        axes[4].plot(profile_x, profile_y_smooth, color="black", linewidth=2, label="smooth")
        axes[4].plot([ridge_x], [ridge_y], "ro", label="ridge")

        x_left_plot = profile_x[p["left_sel"]]
        x_left_norm = (x_left_plot - ridge_x) / max(width, 1.0)
        y_left_fit = p["m_left"] * x_left_norm + p["b_left"]
        axes[4].plot(x_left_plot, y_left_fit, color="blue", linewidth=2, label="left fit")

        x_right_plot = profile_x[p["right_sel"]]
        x_right_norm = (x_right_plot - ridge_x) / max(width, 1.0)
        y_right_fit = p["m_right"] * x_right_norm + p["b_right"]
        axes[4].plot(x_right_plot, y_right_fit, color="green", linewidth=2, label="right fit")

        axes[4].set_title(
            f"Angle={p['angle']:.2f}°\nmedian={median_angle:.2f}, ridge_rel={p['ridge_rel_pos']:.2f}"
        )
        axes[4].set_xlabel("Transverse x")
        axes[4].set_ylabel("Closeness")
        axes[4].legend(fontsize=7)
        axes[4].grid(alpha=0.3)
    else:
        axes[4].text(0.5, 0.5, "No valid profile", ha="center", va="center")
        axes[4].set_title("Angle profile")

    title = (
        f"BCS {sample['bcs_raw']} | good={sample.get('is_good_angle_frame', '')} | "
        f"bad={';'.join(bad_reasons) if bad_reasons else 'none'} | "
        f"tilt={debug.get('axis_tilt_abs', np.nan):.1f}°, rot={debug.get('rotate_deg', np.nan):.1f}°"
    )
    fig.suptitle(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    return True


def make_debug_figures(frame_df):
    if not MAKE_DEBUG_FIGURES:
        return

    dbg_dir = OUT_DIR / "debug_profiles"
    dbg_dir.mkdir(parents=True, exist_ok=True)

    samples = choose_debug_samples(frame_df)

    rows = []

    print("\nMaking debug figures...")

    for i, s in enumerate(samples, start=1):
        bcs = int(s["bcs_raw"])
        good = int(s["is_good_angle_frame"])
        bag_id = str(s["bag_id"]).replace("\\", "_").replace("/", "_")
        frame_key = str(s.get("frame_key", "frame")).replace("\\", "_").replace("/", "_")

        out = dbg_dir / f"BCS{bcs}_good{good}_{i:03d}_{bag_id}_{frame_key}.png"

        ok = save_debug_figure(s, out)

        if ok:
            rows.append({
                "bcs_raw": bcs,
                "is_good_angle_frame": good,
                "bag_id": s["bag_id"],
                "frame_key": s.get("frame_key", ""),
                "bad_reasons": s.get("bad_reasons", ""),
                "frame_angle_median": s.get("frame_angle_median", np.nan),
                "figure_path": str(out),
            })
            print(f"[Saved] {out}")
        else:
            print(f"[Skip] {s.get('depth_npy_path', '')}")

    pd.DataFrame(rows).to_csv(
        dbg_dir / "08e_debug_sample_index.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nDebug figures saved in:")
    print(dbg_dir)


def main():
    if not FRAME_CSV.exists():
        raise FileNotFoundError(f"Cannot find {FRAME_CSV}")

    frame_df, bag_df = extract_all_angles()
    valid_bag, good_bag, good_frame = summarize_and_test(frame_df, bag_df)
    make_plots(valid_bag, good_bag, good_frame)
    make_debug_figures(frame_df)

    print("\nStep 08e finished successfully.")


if __name__ == "__main__":
    main()