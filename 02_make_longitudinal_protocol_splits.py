import os
import re
import csv
import random
from pathlib import Path
from collections import defaultdict, Counter


BASE_PATH = Path(r"./dataset")
OUT_PATH = Path(r"./02_make_longitudinal_protocol_splits")
OUT_PATH.mkdir(parents=True, exist_ok=True)
INPUT_CSV = BASE_PATH / "../01_build_new_rgb_depth_index_and_folds/01_new_rgb_depth_frame_index.csv"

SEED = 42
N_SPLITS_LIST = [3, 5]

ANIMAL_ID_MODE = "first_cow_number"


def natural_key(s):
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", str(s))
    ]


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_episode_id(row):

    if "group_name" in row and row["group_name"]:
        return row["group_name"]

    bag_id = row.get("bag_id", "")
    if "__" in bag_id:
        return bag_id.split("__", 1)[1]

    return bag_id


def infer_animal_id_from_episode(episode_id):

    s = str(episode_id)

    if ANIMAL_ID_MODE == "first_cow_number":
        m = re.match(r"^(Cow_\d+)", s, flags=re.IGNORECASE)
        if m:
            # 统一 Cow 大小写
            cow_num = re.findall(r"\d+", m.group(1))[0]
            return f"Cow_{cow_num}"


    m = re.match(r"^(.*)_(\d+)$", s)
    if m:
        return m.group(1)

    return s


def add_identity_columns(rows):
    new_rows = []

    for r in rows:
        r2 = dict(r)

        episode_id = infer_episode_id(r2)
        animal_id = infer_animal_id_from_episode(episode_id)

        r2["episode_id"] = episode_id
        r2["animal_id"] = animal_id

        # 为了后续兼容，明确 bag_id
        if "bag_id" not in r2 or not r2["bag_id"]:
            r2["bag_id"] = f"BCS{r2['bcs_raw']}__{episode_id}"

        new_rows.append(r2)

    return new_rows


def build_episode_table(rows):
    episode_dict = {}

    for r in rows:
        episode_id = r["episode_id"]

        if episode_id not in episode_dict:
            episode_dict[episode_id] = {
                "episode_id": episode_id,
                "animal_id": r["animal_id"],
                "bag_id": r["bag_id"],
                "bcs_raw": int(r["bcs_raw"]),
                "label5": int(r["label5"]),
                "label3": int(r["label3"]),
                "is_high": int(r["is_high"]),
                "frame_count": 0,
                "rgb_paths": [],
                "depth_paths": [],
            }

        e = episode_dict[episode_id]

        if e["bcs_raw"] != int(r["bcs_raw"]):
            raise RuntimeError(
                f"Episode {episode_id} has multiple BCS labels: "
                f"{e['bcs_raw']} and {r['bcs_raw']}"
            )

        if e["animal_id"] != r["animal_id"]:
            raise RuntimeError(
                f"Episode {episode_id} maps to multiple animals: "
                f"{e['animal_id']} and {r['animal_id']}"
            )

        e["frame_count"] += 1
        e["rgb_paths"].append(r["rgb_path"])
        e["depth_paths"].append(r["depth_npy_path"])

    episodes = list(episode_dict.values())
    episodes = sorted(episodes, key=lambda x: natural_key(x["episode_id"]))

    return episodes


def build_animal_table(episodes):
    animal_dict = {}

    for e in episodes:
        aid = e["animal_id"]

        if aid not in animal_dict:
            animal_dict[aid] = {
                "animal_id": aid,
                "episode_ids": [],
                "bag_ids": [],
                "bcs_values": set(),
                "episode_count": 0,
                "frame_count": 0,
                "bcs_episode_counter": Counter(),
                "bcs_frame_counter": Counter(),
            }

        a = animal_dict[aid]

        a["episode_ids"].append(e["episode_id"])
        a["bag_ids"].append(e["bag_id"])
        a["bcs_values"].add(int(e["bcs_raw"]))
        a["episode_count"] += 1
        a["frame_count"] += int(e["frame_count"])
        a["bcs_episode_counter"][int(e["bcs_raw"])] += 1
        a["bcs_frame_counter"][int(e["bcs_raw"])] += int(e["frame_count"])

    animals = []

    for aid, a in sorted(animal_dict.items(), key=lambda x: natural_key(x[0])):
        animals.append({
            "animal_id": aid,
            "episode_count": a["episode_count"],
            "frame_count": a["frame_count"],
            "bcs_values": ",".join(map(str, sorted(a["bcs_values"]))),
            "n_bcs_values": len(a["bcs_values"]),
            "episode_ids": ";".join(a["episode_ids"]),
            "bag_ids": ";".join(a["bag_ids"]),

            "bcs2_episodes": a["bcs_episode_counter"][2],
            "bcs3_episodes": a["bcs_episode_counter"][3],
            "bcs4_episodes": a["bcs_episode_counter"][4],
            "bcs5_episodes": a["bcs_episode_counter"][5],
            "bcs6_episodes": a["bcs_episode_counter"][6],

            "bcs2_frames": a["bcs_frame_counter"][2],
            "bcs3_frames": a["bcs_frame_counter"][3],
            "bcs4_frames": a["bcs_frame_counter"][4],
            "bcs5_frames": a["bcs_frame_counter"][5],
            "bcs6_frames": a["bcs_frame_counter"][6],
        })

    return animals


def make_units_for_protocol(episodes, animals, protocol):

    units = []

    if protocol == "episodewise":
        for e in episodes:
            bcs_counter_episode = Counter()
            bcs_counter_frame = Counter()

            bcs_counter_episode[int(e["bcs_raw"])] += 1
            bcs_counter_frame[int(e["bcs_raw"])] += int(e["frame_count"])

            units.append({
                "unit_id": e["episode_id"],
                "unit_type": "episode",
                "animal_ids": {e["animal_id"]},
                "episode_ids": {e["episode_id"]},
                "frame_count": int(e["frame_count"]),
                "episode_count": 1,
                "bcs_episode_counter": bcs_counter_episode,
                "bcs_frame_counter": bcs_counter_frame,
                "contains_bcs2": int(bcs_counter_episode[2] > 0),
                "contains_bcs6": int(bcs_counter_episode[6] > 0),
            })

    elif protocol == "animalwise":
        episode_by_animal = defaultdict(list)

        for e in episodes:
            episode_by_animal[e["animal_id"]].append(e)

        for aid, eps in episode_by_animal.items():
            bcs_counter_episode = Counter()
            bcs_counter_frame = Counter()
            episode_ids = set()

            frame_count = 0

            for e in eps:
                bcs_counter_episode[int(e["bcs_raw"])] += 1
                bcs_counter_frame[int(e["bcs_raw"])] += int(e["frame_count"])
                episode_ids.add(e["episode_id"])
                frame_count += int(e["frame_count"])

            units.append({
                "unit_id": aid,
                "unit_type": "animal",
                "animal_ids": {aid},
                "episode_ids": episode_ids,
                "frame_count": frame_count,
                "episode_count": len(eps),
                "bcs_episode_counter": bcs_counter_episode,
                "bcs_frame_counter": bcs_counter_frame,
                "contains_bcs2": int(bcs_counter_episode[2] > 0),
                "contains_bcs6": int(bcs_counter_episode[6] > 0),
            })

    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    return units


def greedy_multilabel_split(units, n_splits):

    random.seed(SEED)

    def unit_priority(u):
        rarity = 0

        if u["contains_bcs2"]:
            rarity += 100
        if u["contains_bcs6"]:
            rarity += 50

        if u["bcs_episode_counter"][3] > 0:
            rarity += 20

        return (
            -rarity,
            -u["episode_count"],
            -u["frame_count"],
            natural_key(u["unit_id"]),
        )

    sorted_units = sorted(units, key=unit_priority)

    folds = [[] for _ in range(n_splits)]
    fold_bcs_episode = [Counter() for _ in range(n_splits)]
    fold_bcs_frame = [Counter() for _ in range(n_splits)]
    fold_frame_count = [0 for _ in range(n_splits)]
    fold_episode_count = [0 for _ in range(n_splits)]
    fold_unit_count = [0 for _ in range(n_splits)]

    for u in sorted_units:
        candidates = list(range(n_splits))
        random.shuffle(candidates)

        def score_fold(f):
            score = 0.0

            for bcs in [2, 3, 4, 5, 6]:
                new_count = fold_bcs_episode[f][bcs] + u["bcs_episode_counter"][bcs]
                score += 20.0 * (new_count ** 2)


            for bcs in [2, 3, 4, 5, 6]:
                new_frames = fold_bcs_frame[f][bcs] + u["bcs_frame_counter"][bcs]
                score += 0.00001 * (new_frames ** 2)


            score += 2.0 * ((fold_unit_count[f] + 1) ** 2)
            score += 0.5 * ((fold_episode_count[f] + u["episode_count"]) ** 2)
            score += 0.000001 * ((fold_frame_count[f] + u["frame_count"]) ** 2)

            return score

        best_f = sorted(candidates, key=score_fold)[0]

        folds[best_f].append(u["unit_id"])

        fold_unit_count[best_f] += 1
        fold_episode_count[best_f] += u["episode_count"]
        fold_frame_count[best_f] += u["frame_count"]

        for bcs in [2, 3, 4, 5, 6]:
            fold_bcs_episode[best_f][bcs] += u["bcs_episode_counter"][bcs]
            fold_bcs_frame[best_f][bcs] += u["bcs_frame_counter"][bcs]

    unit_to_fold = {}

    for fold_id, unit_ids in enumerate(folds):
        for uid in unit_ids:
            unit_to_fold[uid] = fold_id

    return unit_to_fold


def apply_protocol_folds(rows, protocol, unit_to_fold):
    new_rows = []

    for r in rows:
        r2 = dict(r)

        if protocol == "animalwise":
            unit_id = r2["animal_id"]
        elif protocol == "episodewise":
            unit_id = r2["episode_id"]
        else:
            raise ValueError(protocol)

        r2["split_protocol"] = protocol
        r2["split_unit_id"] = unit_id
        r2["fold_id"] = unit_to_fold[unit_id]

        new_rows.append(r2)

    return new_rows


def verify_no_leakage(rows, protocol):

    unit_to_folds = defaultdict(set)

    for r in rows:
        unit_to_folds[r["split_unit_id"]].add(int(r["fold_id"]))

    leaked = {
        uid: folds for uid, folds in unit_to_folds.items()
        if len(folds) > 1
    }

    if leaked:
        print(f"[ERROR] Leakage detected in {protocol}")
        for uid, folds in list(leaked.items())[:20]:
            print(uid, folds)
        raise RuntimeError(f"Leakage detected in {protocol}")

    print(f"No leakage detected for protocol: {protocol}")


def make_summary(rows, protocol, n_splits):
    summary_rows = []

    for fold in range(n_splits):
        fr = [r for r in rows if int(r["fold_id"]) == fold]

        frame_bcs = Counter(int(r["bcs_raw"]) for r in fr)

        episode_to_bcs = {}
        animal_to_episodes = defaultdict(set)
        animal_to_bcs = defaultdict(set)

        for r in fr:
            episode_to_bcs[r["episode_id"]] = int(r["bcs_raw"])
            animal_to_episodes[r["animal_id"]].add(r["episode_id"])
            animal_to_bcs[r["animal_id"]].add(int(r["bcs_raw"]))

        episode_bcs = Counter(episode_to_bcs.values())

        animal_bcs_counter = Counter()
        for aid, bcs_set in animal_to_bcs.items():
            for bcs in bcs_set:
                animal_bcs_counter[bcs] += 1

        summary_rows.append({
            "protocol": protocol,
            "n_splits": n_splits,
            "fold_id": fold,

            "frame_count": len(fr),
            "episode_count": len(episode_to_bcs),
            "animal_count": len(animal_to_episodes),

            "bcs2_frames": frame_bcs[2],
            "bcs3_frames": frame_bcs[3],
            "bcs4_frames": frame_bcs[4],
            "bcs5_frames": frame_bcs[5],
            "bcs6_frames": frame_bcs[6],

            "bcs2_episodes": episode_bcs[2],
            "bcs3_episodes": episode_bcs[3],
            "bcs4_episodes": episode_bcs[4],
            "bcs5_episodes": episode_bcs[5],
            "bcs6_episodes": episode_bcs[6],

            "bcs2_animals": animal_bcs_counter[2],
            "bcs3_animals": animal_bcs_counter[3],
            "bcs4_animals": animal_bcs_counter[4],
            "bcs5_animals": animal_bcs_counter[5],
            "bcs6_animals": animal_bcs_counter[6],
        })

    return summary_rows


def print_summary(summary_rows, protocol, n_splits):
    print("\n" + "=" * 140)
    print(f"{protocol.upper()} | {n_splits}-fold summary")
    print("=" * 140)

    print(
        f"{'Fold':<6}"
        f"{'Frames':<10}"
        f"{'Episodes':<10}"
        f"{'Animals':<10}"
        f"{'B2_fr':<8}"
        f"{'B3_fr':<8}"
        f"{'B4_fr':<8}"
        f"{'B5_fr':<8}"
        f"{'B6_fr':<8}"
        f"{'B2_ep':<8}"
        f"{'B3_ep':<8}"
        f"{'B4_ep':<8}"
        f"{'B5_ep':<8}"
        f"{'B6_ep':<8}"
    )

    for s in summary_rows:
        print(
            f"{s['fold_id']:<6}"
            f"{s['frame_count']:<10}"
            f"{s['episode_count']:<10}"
            f"{s['animal_count']:<10}"
            f"{s['bcs2_frames']:<8}"
            f"{s['bcs3_frames']:<8}"
            f"{s['bcs4_frames']:<8}"
            f"{s['bcs5_frames']:<8}"
            f"{s['bcs6_frames']:<8}"
            f"{s['bcs2_episodes']:<8}"
            f"{s['bcs3_episodes']:<8}"
            f"{s['bcs4_episodes']:<8}"
            f"{s['bcs5_episodes']:<8}"
            f"{s['bcs6_episodes']:<8}"
        )


def save_identity_audits(rows, episodes, animals):
    identity_rows = []

    for e in episodes:
        identity_rows.append({
            "episode_id": e["episode_id"],
            "animal_id": e["animal_id"],
            "bag_id": e["bag_id"],
            "bcs_raw": e["bcs_raw"],
            "label5": e["label5"],
            "label3": e["label3"],
            "frame_count": e["frame_count"],
        })

    identity_csv = OUT_PATH / "02_identity_audit.csv"
    identity_csv.touch(exist_ok=True)
    write_csv(
        identity_csv,
        identity_rows,
        [
            "episode_id",
            "animal_id",
            "bag_id",
            "bcs_raw",
            "label5",
            "label3",
            "frame_count",
        ],
    )

    multibcs_rows = [
        a for a in animals
        if int(a["n_bcs_values"]) > 1
    ]

    multibcs_csv = OUT_PATH / "02_multibcs_animal_audit.csv"

    if multibcs_rows:
        write_csv(multibcs_csv, multibcs_rows, list(multibcs_rows[0].keys()))
    else:
        # 写空表
        write_csv(
            multibcs_csv,
            [],
            [
                "animal_id",
                "episode_count",
                "frame_count",
                "bcs_values",
                "n_bcs_values",
                "episode_ids",
                "bag_ids",
                "bcs2_episodes",
                "bcs3_episodes",
                "bcs4_episodes",
                "bcs5_episodes",
                "bcs6_episodes",
                "bcs2_frames",
                "bcs3_frames",
                "bcs4_frames",
                "bcs5_frames",
                "bcs6_frames",
            ],
        )

    print("\nSaved identity audit:")
    print(identity_csv)
    print(multibcs_csv)

    if multibcs_rows:
        print("\n[Info] Animals appearing in multiple BCS values:")
        for a in multibcs_rows[:20]:
            print(
                f"  {a['animal_id']}: BCS {a['bcs_values']}, "
                f"episodes={a['episode_count']}, frames={a['frame_count']}"
            )
    else:
        print("\nNo animal_id appears in multiple BCS values under current parsing.")


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_CSV}. "
            f"Run 01_build_new_rgb_depth_index_and_folds.py first."
        )

    print("=" * 140)
    print("Step 03: Longitudinal animal / episode split protocols")
    print(f"Input: {INPUT_CSV}")
    print(f"Animal ID mode: {ANIMAL_ID_MODE}")
    print("=" * 140)

    raw_rows = read_csv(INPUT_CSV)
    rows = add_identity_columns(raw_rows)

    episodes = build_episode_table(rows)
    animals = build_animal_table(episodes)

    print("\nDataset identity summary")
    print("-" * 100)
    print(f"Frames:   {len(rows)}")
    print(f"Episodes: {len(episodes)}")
    print(f"Animals:  {len(animals)}")

    print("\nEpisode-level BCS distribution:")
    ep_bcs = Counter(int(e["bcs_raw"]) for e in episodes)
    for bcs in [2, 3, 4, 5, 6]:
        print(f"  BCS {bcs}: {ep_bcs[bcs]} episodes")

    print("\nAnimal-level BCS presence:")
    animal_bcs_presence = Counter()
    for a in animals:
        bcs_set = [int(x) for x in a["bcs_values"].split(",") if x]
        for bcs in bcs_set:
            animal_bcs_presence[bcs] += 1

    for bcs in [2, 3, 4, 5, 6]:
        print(f"  BCS {bcs}: {animal_bcs_presence[bcs]} animals containing this BCS")

    save_identity_audits(rows, episodes, animals)

    saved_files = []

    for protocol in ["animalwise", "episodewise"]:
        units = make_units_for_protocol(episodes, animals, protocol)

        for n_splits in N_SPLITS_LIST:
            unit_to_fold = greedy_multilabel_split(units, n_splits=n_splits)
            protocol_rows = apply_protocol_folds(rows, protocol, unit_to_fold)

            verify_no_leakage(protocol_rows, protocol)

            summary_rows = make_summary(protocol_rows, protocol, n_splits)
            print_summary(summary_rows, protocol, n_splits)

            if protocol == "animalwise":
                prefix = "02A_animalwise"
            else:
                prefix = "02B_episodewise"

            out_csv = OUT_PATH  / f"{prefix}_{n_splits}fold.csv"
            out_summary_csv = OUT_PATH  / f"{prefix}_{n_splits}fold_summary.csv"

            write_csv(out_csv, protocol_rows, list(protocol_rows[0].keys()))
            write_csv(out_summary_csv, summary_rows, list(summary_rows[0].keys()))

            saved_files.append(out_csv)
            saved_files.append(out_summary_csv)

    print("\nSaved split files")
    print("-" * 100)
    for p in saved_files:
        print(p)

    print("\nStep 03 finished successfully.")


if __name__ == "__main__":
    main()