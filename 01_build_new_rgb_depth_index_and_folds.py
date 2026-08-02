import os
import re
import csv
import random
import statistics
from pathlib import Path
from collections import defaultdict, Counter

BASE_PATH = Path(r"../dataset")

VALID_BCS = {"2", "3", "4", "5", "6"}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NPY_EXTS = {".npy"}

N_SPLITS = 5
SEED = 42

random.seed(SEED)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def normalize_stem(stem: str):
    """
    f_0_rgb      -> 0
    f_0_depth    -> 0
    frame_001_rgb -> 1
    """
    s = stem.lower()

    for token in [
        "rgb", "color", "colour", "depth", "dep", "npy",
        "image", "img", "frame", "aligned", "raw"
    ]:
        s = s.replace(token, "")

    s = s.replace("__", "_")
    s = s.replace("-", "_")
    s = s.strip("_")

    nums = re.findall(r"\d+", s)
    if nums:
        return nums[-1].lstrip("0") or "0"

    return s


def label5_from_bcs(bcs_raw: int):
    return bcs_raw - 2


def label3_from_bcs(bcs_raw: int):
    if bcs_raw in [2, 3]:
        return 0
    if bcs_raw in [4, 5]:
        return 1
    if bcs_raw == 6:
        return 2
    raise ValueError(bcs_raw)


def label3_name(label3: int):
    return {
        0: "Lean",
        1: "Ideal",
        2: "High-condition",
    }[label3]


def get_group_name(file_path: Path, bcs_dir: Path):
    rel = file_path.relative_to(bcs_dir)
    parts = rel.parts

    if len(parts) <= 1:
        return "root"

    return parts[0]


def canonical_animal_id(group_name: str, existing_group_names):

    m = re.match(r"^(.*)_(\d+)$", group_name)

    if not m:
        return group_name

    base = m.group(1)
    suffix = int(m.group(2))

    if suffix <= 20 and base in existing_group_names:
        return base

    return group_name


def collect_groups():

    bcs_to_group_names = defaultdict(set)

    for bcs_str in sorted(VALID_BCS, key=natural_key):
        bcs_dir = BASE_PATH / bcs_str
        if not bcs_dir.exists():
            continue

        for p in bcs_dir.rglob("*"):
            if p.is_file() and (p.suffix.lower() in IMG_EXTS or p.suffix.lower() in NPY_EXTS):
                group_name = get_group_name(p, bcs_dir)
                bcs_to_group_names[bcs_str].add(group_name)

    return bcs_to_group_names


def build_paired_index():
    if not BASE_PATH.exists():
        raise FileNotFoundError(f"BASE_PATH not found: {BASE_PATH}")

    bcs_to_group_names = collect_groups()

    frame_rows = []
    bag_dict = {}

    for bcs_str in sorted(VALID_BCS, key=natural_key):
        bcs_dir = BASE_PATH / bcs_str

        if not bcs_dir.exists():
            print(f"[Warning] Missing folder: {bcs_dir}")
            continue

        bcs_raw = int(bcs_str)
        group_names_existing = bcs_to_group_names[bcs_str]

        # group -> files
        group_files = defaultdict(lambda: {"imgs": [], "npys": []})

        files = sorted([p for p in bcs_dir.rglob("*") if p.is_file()], key=natural_key)

        for p in files:
            group_name = get_group_name(p, bcs_dir)

            if p.suffix.lower() in IMG_EXTS:
                group_files[group_name]["imgs"].append(p)
            elif p.suffix.lower() in NPY_EXTS:
                group_files[group_name]["npys"].append(p)

        for group_name, d in sorted(group_files.items(), key=lambda x: natural_key(x[0])):
            imgs = sorted(d["imgs"], key=natural_key)
            npys = sorted(d["npys"], key=natural_key)

            npy_by_norm = {}
            duplicate_npy_keys = []

            for npy_path in npys:
                key = normalize_stem(npy_path.stem)
                if key in npy_by_norm:
                    duplicate_npy_keys.append(key)
                npy_by_norm[key] = npy_path

            animal_id = canonical_animal_id(group_name, group_names_existing)

            bag_id = f"BCS{bcs_raw}__{group_name}"

            split_group_id = animal_id

            if bag_id not in bag_dict:
                bag_dict[bag_id] = {
                    "bag_id": bag_id,
                    "split_group_id": split_group_id,
                    "group_name": group_name,
                    "animal_id_guess": animal_id,
                    "bcs_raw": bcs_raw,
                    "label5": label5_from_bcs(bcs_raw),
                    "label3": label3_from_bcs(bcs_raw),
                    "is_high": 1 if bcs_raw == 6 else 0,
                    "rgb_count": len(imgs),
                    "npy_count": len(npys),
                    "paired_count": 0,
                    "missing_npy_count": 0,
                    "duplicate_npy_keys": ";".join(map(str, duplicate_npy_keys[:20])),
                }

            for img_path in imgs:
                key = normalize_stem(img_path.stem)
                npy_path = npy_by_norm.get(key, None)

                if npy_path is None:
                    bag_dict[bag_id]["missing_npy_count"] += 1
                    continue

                bag_dict[bag_id]["paired_count"] += 1

                frame_rows.append({
                    "rgb_path": str(img_path),
                    "depth_npy_path": str(npy_path),

                    "bcs_raw": bcs_raw,
                    "label5": label5_from_bcs(bcs_raw),
                    "label5_name": f"BCS_{bcs_raw}",

                    "label3": label3_from_bcs(bcs_raw),
                    "label3_name": label3_name(label3_from_bcs(bcs_raw)),

                    "is_high": 1 if bcs_raw == 6 else 0,

                    "bag_id": bag_id,
                    "split_group_id": split_group_id,
                    "group_name": group_name,
                    "animal_id_guess": animal_id,

                    "frame_key": key,
                    "rgb_file": img_path.name,
                    "depth_file": npy_path.name,
                    "rgb_ext": img_path.suffix.lower(),
                })

    bag_rows = list(bag_dict.values())
    return frame_rows, bag_rows


def build_animal_inventory(bag_rows):
    animal_dict = {}

    for b in bag_rows:
        sid = b["split_group_id"]

        if sid not in animal_dict:
            animal_dict[sid] = {
                "split_group_id": sid,
                "bag_count": 0,
                "paired_frame_count": 0,
                "bcs_values": set(),
                "bag_ids": [],
            }

        animal_dict[sid]["bag_count"] += 1
        animal_dict[sid]["paired_frame_count"] += int(b["paired_count"])
        animal_dict[sid]["bcs_values"].add(int(b["bcs_raw"]))
        animal_dict[sid]["bag_ids"].append(b["bag_id"])

    animal_rows = []

    for sid, a in sorted(animal_dict.items(), key=lambda x: natural_key(x[0])):
        animal_rows.append({
            "split_group_id": sid,
            "bag_count": a["bag_count"],
            "paired_frame_count": a["paired_frame_count"],
            "bcs_values": ",".join(map(str, sorted(a["bcs_values"]))),
            "n_bcs_values": len(a["bcs_values"]),
            "bag_ids": ";".join(a["bag_ids"]),
        })

    return animal_rows


def greedy_split_animals(animal_rows, bag_rows):

    bag_by_animal = defaultdict(list)

    for b in bag_rows:
        if int(b["paired_count"]) <= 0:
            continue
        bag_by_animal[b["split_group_id"]].append(b)

    animal_items = []

    for a in animal_rows:
        sid = a["split_group_id"]
        bags = bag_by_animal.get(sid, [])

        if not bags:
            continue

        bcs_counter = Counter()
        frame_count = 0
        bag_count = 0

        for b in bags:
            bcs_counter[int(b["bcs_raw"])] += 1
            frame_count += int(b["paired_count"])
            bag_count += 1

        rarity_score = 0
        if bcs_counter[2] > 0:
            rarity_score += 10
        if bcs_counter[6] > 0:
            rarity_score += 8
        if bcs_counter[3] > 0:
            rarity_score += 4

        animal_items.append({
            "split_group_id": sid,
            "bcs_counter": bcs_counter,
            "frame_count": frame_count,
            "bag_count": bag_count,
            "rarity_score": rarity_score,
        })

    animal_items = sorted(
        animal_items,
        key=lambda x: (-x["rarity_score"], -x["frame_count"], natural_key(x["split_group_id"]))
    )

    folds = [[] for _ in range(N_SPLITS)]
    fold_bcs_bag_counts = [Counter() for _ in range(N_SPLITS)]
    fold_frame_counts = [0 for _ in range(N_SPLITS)]
    fold_animal_counts = [0 for _ in range(N_SPLITS)]

    for item in animal_items:
        candidates = list(range(N_SPLITS))
        random.shuffle(candidates)

        def fold_score(f):

            score = 0
            for bcs in [2, 3, 4, 5, 6]:
                score += (fold_bcs_bag_counts[f][bcs] + item["bcs_counter"][bcs]) ** 2

            score += 0.05 * (fold_animal_counts[f] + 1) ** 2
            score += 0.000001 * (fold_frame_counts[f] + item["frame_count"]) ** 2
            return score

        best_fold = sorted(candidates, key=fold_score)[0]

        folds[best_fold].append(item["split_group_id"])
        fold_animal_counts[best_fold] += 1
        fold_frame_counts[best_fold] += item["frame_count"]

        for bcs, cnt in item["bcs_counter"].items():
            fold_bcs_bag_counts[best_fold][bcs] += cnt

    split_group_to_fold = {}

    for fold_id, sids in enumerate(folds):
        for sid in sids:
            split_group_to_fold[sid] = fold_id

    return split_group_to_fold


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_fold_summary(frame_rows):
    summary_rows = []

    for fold in range(N_SPLITS):
        rows = [r for r in frame_rows if int(r["fold_id"]) == fold]

        bcs_img = Counter(int(r["bcs_raw"]) for r in rows)
        bags = {}
        animals = defaultdict(set)

        for r in rows:
            bags[r["bag_id"]] = int(r["bcs_raw"])
            animals[r["split_group_id"]].add(r["bag_id"])

        bcs_bag = Counter(bags.values())

        summary_rows.append({
            "fold_id": fold,
            "frame_count": len(rows),
            "bag_count": len(bags),
            "animal_count": len(animals),

            "bcs2_frames": bcs_img[2],
            "bcs3_frames": bcs_img[3],
            "bcs4_frames": bcs_img[4],
            "bcs5_frames": bcs_img[5],
            "bcs6_frames": bcs_img[6],

            "bcs2_bags": bcs_bag[2],
            "bcs3_bags": bcs_bag[3],
            "bcs4_bags": bcs_bag[4],
            "bcs5_bags": bcs_bag[5],
            "bcs6_bags": bcs_bag[6],
        })

    return summary_rows


def verify_no_leakage(frame_rows):
    sid_to_folds = defaultdict(set)

    for r in frame_rows:
        sid_to_folds[r["split_group_id"]].add(int(r["fold_id"]))

    leaked = {
        sid: folds for sid, folds in sid_to_folds.items()
        if len(folds) > 1
    }

    if leaked:
        print("[ERROR] Split group leakage detected!")
        for sid, folds in list(leaked.items())[:20]:
            print(sid, folds)
        raise RuntimeError("Same split_group_id appears in multiple folds.")

    print("No split_group_id leakage detected.")


def main():
    print("=" * 100)
    print("Build new RGB-depth paired index and cow-wise folds")
    print(f"BASE_PATH: {BASE_PATH}")
    print("=" * 100)

    frame_rows, bag_rows = build_paired_index()
    animal_rows = build_animal_inventory(bag_rows)

    if not frame_rows:
        raise RuntimeError("No paired RGB-depth frames found.")

    split_group_to_fold = greedy_split_animals(animal_rows, bag_rows)

    new_frame_rows = []

    for r in frame_rows:
        r2 = dict(r)
        r2["fold_id"] = split_group_to_fold[r["split_group_id"]]
        new_frame_rows.append(r2)

    verify_no_leakage(new_frame_rows)

    # 输出文件
    frame_csv = BASE_PATH / "01_new_rgb_depth_frame_index.csv"
    bag_csv = BASE_PATH / "01_new_rgb_depth_bag_inventory.csv"
    animal_csv = BASE_PATH / "01_new_rgb_depth_animal_inventory.csv"
    fold_csv = BASE_PATH / "02_new_rgb_depth_5fold_cowwise.csv"
    summary_csv = BASE_PATH / "02_new_rgb_depth_fold_summary.csv"

    write_csv(frame_csv, frame_rows, list(frame_rows[0].keys()))
    write_csv(bag_csv, bag_rows, list(bag_rows[0].keys()))
    write_csv(animal_csv, animal_rows, list(animal_rows[0].keys()))
    write_csv(fold_csv, new_frame_rows, list(new_frame_rows[0].keys()))

    summary_rows = make_fold_summary(new_frame_rows)
    write_csv(summary_csv, summary_rows, list(summary_rows[0].keys()))

    # 基本统计
    bcs_frame_counter = Counter(int(r["bcs_raw"]) for r in new_frame_rows)
    bcs_bag_counter = Counter(int(b["bcs_raw"]) for b in bag_rows if int(b["paired_count"]) > 0)
    animal_bcs_multi = [a for a in animal_rows if int(a["n_bcs_values"]) > 1]

    print("\nDataset summary")
    print("-" * 100)
    print(f"Paired frames: {len(new_frame_rows)}")
    print(f"Bags:          {sum(1 for b in bag_rows if int(b['paired_count']) > 0)}")
    print(f"Animals/split groups: {len(set(r['split_group_id'] for r in new_frame_rows))}")

    print("\nFrame distribution by BCS")
    for bcs in [2, 3, 4, 5, 6]:
        print(f"  BCS {bcs}: {bcs_frame_counter[bcs]} frames")

    print("\nBag distribution by BCS")
    for bcs in [2, 3, 4, 5, 6]:
        print(f"  BCS {bcs}: {bcs_bag_counter[bcs]} bags")

    if animal_bcs_multi:
        print("\n[Warning] Some split_group_id values appear with multiple BCS labels.")
        print("They will still be kept in the same fold to avoid leakage.")
        print("First examples:")
        for a in animal_bcs_multi[:10]:
            print(f"  {a['split_group_id']}: BCS {a['bcs_values']}")
    else:
        print("\nNo split_group_id appears under multiple BCS labels.")

    print("\nFold summary")
    print("-" * 130)
    print(
        f"{'Fold':<6}"
        f"{'Frames':<10}"
        f"{'Bags':<8}"
        f"{'Animals':<10}"
        f"{'B2_fr':<8}"
        f"{'B3_fr':<8}"
        f"{'B4_fr':<8}"
        f"{'B5_fr':<8}"
        f"{'B6_fr':<8}"
        f"{'B2_bag':<8}"
        f"{'B3_bag':<8}"
        f"{'B4_bag':<8}"
        f"{'B5_bag':<8}"
        f"{'B6_bag':<8}"
    )

    for s in summary_rows:
        print(
            f"{s['fold_id']:<6}"
            f"{s['frame_count']:<10}"
            f"{s['bag_count']:<8}"
            f"{s['animal_count']:<10}"
            f"{s['bcs2_frames']:<8}"
            f"{s['bcs3_frames']:<8}"
            f"{s['bcs4_frames']:<8}"
            f"{s['bcs5_frames']:<8}"
            f"{s['bcs6_frames']:<8}"
            f"{s['bcs2_bags']:<8}"
            f"{s['bcs3_bags']:<8}"
            f"{s['bcs4_bags']:<8}"
            f"{s['bcs5_bags']:<8}"
            f"{s['bcs6_bags']:<8}"
        )

    print("\nSaved files")
    print("-" * 100)
    print(frame_csv)
    print(bag_csv)
    print(animal_csv)
    print(fold_csv)
    print(summary_csv)

    print("\nStep finished successfully.")


if __name__ == "__main__":
    main()