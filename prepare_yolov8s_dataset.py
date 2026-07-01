import argparse
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


SEED = 42


def read_classes(classes_file: Path) -> list[str]:
    return [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_main_dataset(workspace: Path) -> Path:
    for labels_dir in workspace.rglob("labels"):
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


def parse_xml(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    objects = [obj.findtext("name", "").strip() for obj in root.findall("object")]
    return {"stem": xml_path.stem, "classes": [name for name in objects if name]}


def choose_clean_label_dir(workspace: Path, dataset: Path) -> Path:
    qc_labels = workspace / "reports" / "main_annotation_qc" / "cleaned_yolo_labels_from_xml"
    if qc_labels.exists():
        return qc_labels
    return dataset / "labels"


def record_group(record: dict, group_size: int) -> str:
    match = re.search(r"(\d+)$", record["stem"])
    if not match:
        return record["stem"]
    return f"group_{int(match.group(1)) // group_size:04d}"


def multilabel_split(records: list[dict], ratios=(0.7, 0.15, 0.15), group_size: int = 50) -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    total_counts = Counter()
    for record in records:
        total_counts.update(record["classes"])

    groups = {}
    for record in records:
        groups.setdefault(record_group(record, group_size), []).append(record)

    def rarity_score(record: dict) -> float:
        return sum(1.0 / max(total_counts[name], 1) for name in set(record["classes"]))

    group_items = list(groups.values())
    rng.shuffle(group_items)
    group_items.sort(key=lambda items: sum(rarity_score(record) for record in items), reverse=True)

    split_names = ["train", "val", "test"]
    targets = {name: ratios[i] for i, name in enumerate(split_names)}
    splits = {name: [] for name in split_names}
    split_counts = {name: Counter() for name in split_names}
    target_images = {name: int(round(len(records) * targets[name])) for name in split_names}
    target_images["train"] = len(records) - target_images["val"] - target_images["test"]
    target_class_counts = {
        name: {cls: total * targets[name] for cls, total in total_counts.items()}
        for name in split_names
    }

    for group_records in group_items:
        group_classes = Counter()
        for record in group_records:
            group_classes.update(record["classes"])
        best_name = None
        best_score = None
        for name in split_names:
            if len(splits[name]) + len(group_records) > target_images[name] + group_size:
                continue
            fill_bonus = 1.0 - (len(splits[name]) / max(target_images[name], 1))
            class_deficit_score = 0.0
            for cls, amount in group_classes.items():
                desired = target_class_counts[name][cls]
                deficit = desired - split_counts[name][cls]
                if deficit > 0:
                    class_deficit_score += min(deficit, amount) / max(total_counts[cls], 1)
            score = class_deficit_score + fill_bonus * 0.05
            if best_score is None or score > best_score:
                best_name = name
                best_score = score
        if best_name is None:
            best_name = min(split_names, key=lambda item: len(splits[item]) / max(target_images[item], 1))
        splits[best_name].extend(group_records)
        split_counts[best_name].update(group_classes)
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


def write_yaml(out_root: Path, classes: list[str]) -> None:
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(classes))
    yaml = f"""# Generated by tools/prepare_yolov8s_dataset.py
path: .
train: train_oversampled.txt
val: images/val
test: images/test
names:
{names}
"""
    (out_root / "data.yaml").write_text(yaml, encoding="utf-8")


def write_count_report(out_root: Path, splits: dict[str, list[dict]], class_counts: Counter) -> None:
    lines = ["split,class,count"]
    for split, records in splits.items():
        counts = Counter()
        for record in records:
            counts.update(record["classes"])
        for cls, count in counts.most_common():
            lines.append(f"{split},{cls},{count}")
    lines.append("")
    lines.append("all,class,count")
    for cls, count in class_counts.most_common():
        lines.append(f"all,{cls},{count}")
    (out_root / "split_class_counts.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="prepared_yolov8s_main", help="Output YOLO dataset directory.")
    parser.add_argument("--force", action="store_true", help="Overwrite output directory.")
    parser.add_argument("--group-size", type=int, default=50, help="Keep nearby numbered images in the same split.")
    args = parser.parse_args()

    workspace = Path.cwd()
    dataset = find_main_dataset(workspace)
    src_labels = choose_clean_label_dir(workspace, dataset)
    classes = read_classes(src_labels / "classes.txt")
    records = [parse_xml(path) for path in sorted((dataset / "Annotations").glob("*.xml"))]
    class_counts = Counter()
    for record in records:
        class_counts.update(record["classes"])

    out_root = workspace / args.out
    if out_root.exists():
        if not args.force:
            raise FileExistsError(f"{out_root} exists. Use --force to overwrite.")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    splits = multilabel_split(records, group_size=args.group_size)
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
        "\n".join(path.relative_to(out_root).as_posix() for path in oversampled_train) + "\n",
        encoding="utf-8",
    )
    (out_root / "train_original.txt").write_text(
        "\n".join(path.relative_to(out_root).as_posix() for path in image_paths_by_split["train"]) + "\n",
        encoding="utf-8",
    )

    write_yaml(out_root, classes)
    write_count_report(out_root, splits, class_counts)

    print(f"source_dataset={dataset}")
    print(f"source_labels={src_labels}")
    print(f"out={out_root}")
    print(f"images train/val/test={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"oversampled_train_items={len(oversampled_train)}")
    print(f"data_yaml={out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
