import argparse
import csv
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


SEED = 42


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.size = {item: 1 for item in items}

    def find(self, item):
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            item, self.parent[item] = self.parent[item], root
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def read_classes(classes_file: Path) -> list[str]:
    return [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_main_dataset(workspace: Path) -> Path:
    for labels_dir in workspace.rglob("labels"):
        if "prepared_" in str(labels_dir):
            continue
        classes_file = labels_dir / "classes.txt"
        root = labels_dir.parent
        if not classes_file.exists():
            continue
        if not (root / "Annotations").is_dir() or not (root / "JPEGImages").is_dir():
            continue
        classes = read_classes(classes_file)
        if "platen-off" in classes and "switch-center" in classes:
            return root
    raise FileNotFoundError("Could not find the main control-cabinet dataset.")


def choose_clean_label_dir(workspace: Path, dataset: Path) -> Path:
    qc_labels = workspace / "reports" / "main_annotation_qc" / "cleaned_yolo_labels_from_xml"
    if qc_labels.exists():
        return qc_labels
    return dataset / "labels"


def parse_xml(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    classes = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name:
            classes.append(name)
    return {"stem": xml_path.stem, "classes": classes}


def dhash(image_path: Path, hash_size: int = 16) -> int:
    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def build_similarity_clusters(records: list[dict], image_dir: Path, threshold: int) -> tuple[dict[str, int], list[tuple[str, str, int]]]:
    stems = [record["stem"] for record in records]
    hashes = {stem: dhash(image_dir / f"{stem}.jpg") for stem in stems}
    uf = UnionFind(stems)
    edges = []

    buckets = defaultdict(list)
    for stem in stems:
        # Use the high 24 bits as a coarse bucket, then compare neighboring candidates
        # by full Hamming distance. This keeps 3000-image clustering fast and local.
        bucket = hashes[stem] >> (16 * 16 - 24)
        buckets[bucket].append(stem)

    # Compare all pairs too: 3000 images is small enough, and full comparison gives a
    # more reliable leakage barrier than approximate bucketing alone.
    for i, left in enumerate(stems):
        h_left = hashes[left]
        for right in stems[i + 1:]:
            dist = hamming(h_left, hashes[right])
            if dist <= threshold:
                uf.union(left, right)
                edges.append((left, right, dist))

    roots = {}
    root_to_id = {}
    for stem in stems:
        root = uf.find(stem)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        roots[stem] = root_to_id[root]
    return roots, edges


def split_clusters(records: list[dict], cluster_ids: dict[str, int], ratios=(0.7, 0.15, 0.15)) -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    total_counts = Counter()
    for record in records:
        total_counts.update(record["classes"])

    clusters = defaultdict(list)
    for record in records:
        clusters[cluster_ids[record["stem"]]].append(record)

    cluster_items = list(clusters.values())

    def rarity_score(items: list[dict]) -> float:
        classes = set()
        for record in items:
            classes.update(record["classes"])
        return sum(1.0 / max(total_counts[cls], 1) for cls in classes)

    rng.shuffle(cluster_items)
    cluster_items.sort(key=rarity_score, reverse=True)

    split_names = ["train", "val", "test"]
    targets = {name: ratios[i] for i, name in enumerate(split_names)}
    target_images = {name: int(round(len(records) * targets[name])) for name in split_names}
    target_images["train"] = len(records) - target_images["val"] - target_images["test"]
    target_class_counts = {
        name: {cls: total * targets[name] for cls, total in total_counts.items()}
        for name in split_names
    }

    splits = {name: [] for name in split_names}
    split_counts = {name: Counter() for name in split_names}

    for items in cluster_items:
        item_counts = Counter()
        for record in items:
            item_counts.update(record["classes"])

        best_name = None
        best_score = None
        for name in split_names:
            fill = len(splits[name]) / max(target_images[name], 1)
            if fill > 1.15:
                continue
            class_deficit_score = 0.0
            for cls, amount in item_counts.items():
                desired = target_class_counts[name][cls]
                deficit = desired - split_counts[name][cls]
                if deficit > 0:
                    class_deficit_score += min(deficit, amount) / max(total_counts[cls], 1)
            size_penalty = abs((len(splits[name]) + len(items)) - target_images[name]) / max(target_images[name], 1)
            score = class_deficit_score - size_penalty * 0.05
            if best_score is None or score > best_score:
                best_name = name
                best_score = score
        if best_name is None:
            best_name = min(split_names, key=lambda name: len(splits[name]) / max(target_images[name], 1))
        splits[best_name].extend(items)
        split_counts[best_name].update(item_counts)
    return splits


def repeat_factor(record: dict, class_counts: Counter) -> int:
    factor = 1
    for cls in set(record["classes"]):
        count = class_counts[cls]
        if count <= 100:
            factor = max(factor, 8)
        elif count <= 200:
            factor = max(factor, 5)
        elif count <= 400:
            factor = max(factor, 3)
        elif count <= 900:
            factor = max(factor, 2)
    return factor


def copy_pair(stem: str, src_images: Path, src_labels: Path, out_root: Path, split: str) -> Path:
    image_src = src_images / f"{stem}.jpg"
    label_src = src_labels / f"{stem}.txt"
    image_dst = out_root / "images" / split / image_src.name
    label_dst = out_root / "labels" / split / label_src.name
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_src, image_dst)
    shutil.copy2(label_src, label_dst)
    return image_dst


def write_yaml(workspace: Path, out_root: Path, classes: list[str]) -> None:
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(classes))
    dataset_path = out_root.resolve().as_posix()
    yaml = f"""# Generated by tools/prepare_yolov8s_clustered_dataset.py
path: {dataset_path}
train: train_oversampled.txt
val: images/val
test: images/test
names:
{names}
"""
    (out_root / "data.yaml").write_text(yaml, encoding="utf-8")


def write_reports(out_root: Path, splits: dict[str, list[dict]], cluster_ids: dict[str, int], edges: list[tuple[str, str, int]]) -> None:
    with (out_root / "split_class_counts.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class", "count"])
        for split, records in splits.items():
            counts = Counter()
            for record in records:
                counts.update(record["classes"])
            for cls, count in counts.most_common():
                writer.writerow([split, cls, count])

    stem_to_split = {}
    for split, records in splits.items():
        for record in records:
            stem_to_split[record["stem"]] = split

    with (out_root / "clusters.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "cluster_id", "split"])
        for stem, cid in sorted(cluster_ids.items(), key=lambda row: (row[1], row[0])):
            writer.writerow([stem, cid, stem_to_split[stem]])

    leakage_edges = []
    for left, right, dist in edges:
        if stem_to_split[left] != stem_to_split[right]:
            leakage_edges.append((left, right, dist, stem_to_split[left], stem_to_split[right]))

    with (out_root / "similarity_edges_cross_split.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image_a", "image_b", "hash_distance", "split_a", "split_b"])
        writer.writerows(leakage_edges)

    cluster_sizes = Counter(cluster_ids.values())
    with (out_root / "cluster_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "size", "split"])
        for cid, size in cluster_sizes.most_common():
            members = [stem for stem, member_cid in cluster_ids.items() if member_cid == cid]
            writer.writerow([cid, size, stem_to_split[members[0]]])

    print(f"cross_split_similarity_edges={len(leakage_edges)}")
    print(f"clusters={len(cluster_sizes)} largest_cluster={max(cluster_sizes.values()) if cluster_sizes else 0}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="prepared_yolov8s_clustered")
    parser.add_argument("--threshold", type=int, default=10, help="dHash Hamming distance for grouping near-duplicates.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    workspace = Path.cwd()
    dataset = find_main_dataset(workspace)
    src_labels = choose_clean_label_dir(workspace, dataset)
    classes = read_classes(src_labels / "classes.txt")
    records = [parse_xml(path) for path in sorted((dataset / "Annotations").glob("*.xml"))]

    out_root = workspace / args.out
    if out_root.exists():
        if not args.force:
            raise FileExistsError(f"{out_root} exists. Use --force to overwrite.")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    class_counts = Counter()
    for record in records:
        class_counts.update(record["classes"])

    cluster_ids, edges = build_similarity_clusters(records, dataset / "JPEGImages", args.threshold)
    splits = split_clusters(records, cluster_ids)

    image_paths_by_split = {name: [] for name in splits}
    for split, split_records in splits.items():
        for record in split_records:
            image_dst = copy_pair(record["stem"], dataset / "JPEGImages", src_labels, out_root, split)
            image_paths_by_split[split].append(image_dst)

    oversampled_train = []
    for record, image_path in zip(splits["train"], image_paths_by_split["train"]):
        oversampled_train.extend([image_path] * repeat_factor(record, class_counts))
    rng = random.Random(SEED)
    rng.shuffle(oversampled_train)

    (out_root / "train_oversampled.txt").write_text(
        "\n".join(path.resolve().as_posix() for path in oversampled_train) + "\n",
        encoding="utf-8",
    )
    (out_root / "train_original.txt").write_text(
        "\n".join(path.resolve().as_posix() for path in image_paths_by_split["train"]) + "\n",
        encoding="utf-8",
    )

    write_yaml(workspace, out_root, classes)
    write_reports(out_root, splits, cluster_ids, edges)

    print(f"source_dataset={dataset}")
    print(f"source_labels={src_labels}")
    print(f"out={out_root}")
    print(f"images train/val/test={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"oversampled_train_items={len(oversampled_train)}")
    print(f"data_yaml={out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
