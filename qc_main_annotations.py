import csv
import html
import math
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


CLASS_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
]


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


def read_classes(classes_file: Path) -> list[str]:
    return [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_xml(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width", "0"))) if size is not None else 0
    height = int(float(size.findtext("height", "0"))) if size is not None else 0
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin", "0"))
        ymin = float(bnd.findtext("ymin", "0"))
        xmax = float(bnd.findtext("xmax", "0"))
        ymax = float(bnd.findtext("ymax", "0"))
        boxes.append({"class": name, "box": (xmin, ymin, xmax, ymax)})
    return {"stem": xml_path.stem, "xml": xml_path, "width": width, "height": height, "boxes": boxes}


def norm_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = box
    return (xmin / width, ymin / height, xmax / width, ymax / height)


def yolo_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = box
    xmin = min(max(xmin, 0.0), float(width))
    xmax = min(max(xmax, 0.0), float(width))
    ymin = min(max(ymin, 0.0), float(height))
    ymax = min(max(ymax, 0.0), float(height))
    x = ((xmin + xmax) / 2.0) / width
    y = ((ymin + ymax) / 2.0) / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return tuple(min(max(v, 0.0), 1.0) for v in (x, y, w, h))


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


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


def compare_annotations(a: dict, b: dict) -> dict:
    a_boxes = [(box["class"], norm_box(box["box"], a["width"], a["height"])) for box in a["boxes"]]
    b_boxes = [(box["class"], norm_box(box["box"], b["width"], b["height"])) for box in b["boxes"]]
    used_b = set()
    class_mismatches = []
    matched_same = 0

    for ai, (acls, abox) in enumerate(a_boxes):
        best = None
        for bi, (bcls, bbox) in enumerate(b_boxes):
            if bi in used_b:
                continue
            score = iou(abox, bbox)
            if best is None or score > best[0]:
                best = (score, bi, bcls, bbox)
        if best and best[0] >= 0.55:
            used_b.add(best[1])
            if acls == best[2]:
                matched_same += 1
            else:
                class_mismatches.append((acls, best[2], best[0]))

    unmatched_a = len(a_boxes) - matched_same - len(class_mismatches)
    unmatched_b = len(b_boxes) - len(used_b)
    count_delta = sum((Counter(c for c, _ in a_boxes) - Counter(c for c, _ in b_boxes)).values())
    count_delta += sum((Counter(c for c, _ in b_boxes) - Counter(c for c, _ in a_boxes)).values())
    score = len(class_mismatches) * 4 + unmatched_a + unmatched_b + count_delta * 0.5
    return {
        "score": score,
        "matched_same": matched_same,
        "class_mismatches": class_mismatches,
        "unmatched_a": unmatched_a,
        "unmatched_b": unmatched_b,
        "count_delta": count_delta,
    }


def draw_preview(image_path: Path, record: dict, class_to_id: dict[str, int], out_path: Path) -> None:
    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return
    scale = min(1.0, 900 / max(img.shape[:2]))
    if scale < 1.0:
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    for item in record["boxes"]:
        cls = item["class"]
        xmin, ymin, xmax, ymax = [int(round(v * scale)) for v in item["box"]]
        color = CLASS_COLORS[class_to_id.get(cls, 0) % len(CLASS_COLORS)]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.rectangle(img, (xmin, max(0, ymin - 20)), (xmin + min(220, 8 * len(cls) + 8), ymin), color, -1)
        cv2.putText(img, cls, (xmin + 3, max(14, ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    ok, data = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if ok:
        data.tofile(str(out_path))


def make_pair_preview(left_path: Path, right_path: Path, out_path: Path) -> None:
    left = cv2.imdecode(np.fromfile(str(left_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    right = cv2.imdecode(np.fromfile(str(right_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if left is None or right is None:
        return
    height = max(left.shape[0], right.shape[0])
    def pad(img):
        if img.shape[0] == height:
            return img
        pad_h = height - img.shape[0]
        return cv2.copyMakeBorder(img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    joined = np.concatenate([pad(left), pad(right)], axis=1)
    ok, data = cv2.imencode(".jpg", joined, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if ok:
        data.tofile(str(out_path))


def write_clean_yolo(records: list[dict], classes: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    class_to_id = {name: i for i, name in enumerate(classes)}
    for record in records:
        lines = []
        for item in record["boxes"]:
            if item["class"] not in class_to_id:
                continue
            vals = yolo_box(item["box"], record["width"], record["height"])
            lines.append(f"{class_to_id[item['class']]} " + " ".join(f"{v:.6f}" for v in vals))
        (out_dir / f"{record['stem']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    workspace = Path.cwd()
    dataset = find_main_dataset(workspace)
    report_dir = workspace / "reports" / "main_annotation_qc"
    preview_dir = report_dir / "previews"
    pair_dir = report_dir / "similar_pairs"
    clean_label_dir = report_dir / "cleaned_yolo_labels_from_xml"
    if report_dir.exists():
        shutil.rmtree(report_dir)
    preview_dir.mkdir(parents=True)
    pair_dir.mkdir(parents=True)

    classes = read_classes(dataset / "labels" / "classes.txt")
    class_to_id = {name: i for i, name in enumerate(classes)}
    records = [parse_xml(path) for path in sorted((dataset / "Annotations").glob("*.xml"))]
    by_stem = {r["stem"]: r for r in records}

    class_counts = Counter()
    rule_issues = []
    duplicate_issues = []
    xml_yolo_issues = []
    yolo_range_issues = []

    for record in records:
        seen = []
        for idx, item in enumerate(record["boxes"]):
            cls = item["class"]
            class_counts[cls] += 1
            xmin, ymin, xmax, ymax = item["box"]
            if cls not in class_to_id:
                rule_issues.append((record["stem"], "unknown_class", cls, item["box"]))
            if record["width"] <= 0 or record["height"] <= 0 or xmax <= xmin or ymax <= ymin:
                rule_issues.append((record["stem"], "invalid_box", cls, item["box"]))
            if xmin < 0 or ymin < 0 or xmax > record["width"] or ymax > record["height"]:
                rule_issues.append((record["stem"], "box_out_of_bounds", cls, item["box"]))
            nbox = norm_box(item["box"], record["width"], record["height"])
            for prev_cls, prev_box in seen:
                if cls == prev_cls and iou(nbox, prev_box) > 0.95:
                    duplicate_issues.append((record["stem"], cls, round(iou(nbox, prev_box), 4)))
            seen.append((cls, nbox))

        txt_path = dataset / "labels" / f"{record['stem']}.txt"
        expected = []
        for item in record["boxes"]:
            if item["class"] in class_to_id:
                expected.append((class_to_id[item["class"]], yolo_box(item["box"], record["width"], record["height"])))
        found = []
        if txt_path.exists():
            for line_no, line in enumerate(txt_path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                parts = line.split()
                if len(parts) == 5:
                    cls_id = int(float(parts[0]))
                    vals = tuple(float(x) for x in parts[1:])
                    found.append((cls_id, vals))
                    x, y, w, h = vals
                    edge_violation = max(0.0, -(x - w / 2), -(y - h / 2), x + w / 2 - 1, y + h / 2 - 1)
                    value_violation = max(0.0, max(vals) - 1, -min(vals), -w, -h)
                    violation = max(edge_violation, value_violation)
                    if violation > 0:
                        level = "minor_rounding" if violation <= 1e-4 else "severe_out_of_bounds"
                        yolo_range_issues.append((record["stem"], line_no, level, f"{violation:.8f}", line))
        if len(expected) != len(found):
            xml_yolo_issues.append((record["stem"], "count_mismatch", len(expected), len(found)))
        else:
            for exp, got in zip(expected, found):
                if exp[0] != got[0] or max(abs(a - b) for a, b in zip(exp[1], got[1])) > 1e-4:
                    xml_yolo_issues.append((record["stem"], "value_mismatch", exp[0], got[0]))
                    break

    hashes = {}
    image_paths = {}
    for record in records:
        image_path = dataset / "JPEGImages" / f"{record['stem']}.jpg"
        if image_path.exists():
            image_paths[record["stem"]] = image_path
            hashes[record["stem"]] = dhash(image_path)

    stems = sorted(hashes)
    candidates = []
    for i, left in enumerate(stems):
        left_hash = hashes[left]
        close = []
        for right in stems[i + 1:]:
            dist = hamming(left_hash, hashes[right])
            if dist <= 18:
                close.append((dist, right))
        for dist, right in sorted(close)[:8]:
            cmp = compare_annotations(by_stem[left], by_stem[right])
            if cmp["score"] >= 3:
                candidates.append((cmp["score"], dist, left, right, cmp))

    candidates.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
    top_candidates = candidates[:150]

    with (report_dir / "class_counts.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "count"])
        for cls, count in class_counts.most_common():
            writer.writerow([cls, count])

    with (report_dir / "rule_issues.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "issue", "class", "box"])
        writer.writerows(rule_issues)

    with (report_dir / "duplicate_box_issues.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class", "iou"])
        writer.writerows(duplicate_issues)

    with (report_dir / "xml_yolo_mismatch.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "issue", "xml_value", "txt_value"])
        writer.writerows(xml_yolo_issues)

    with (report_dir / "yolo_range_issues.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "line", "level", "max_violation", "raw_label"])
        writer.writerows(yolo_range_issues)

    with (report_dir / "similar_annotation_conflicts.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "score", "image_hash_distance", "image_a", "image_b", "class_mismatches", "unmatched_a", "unmatched_b", "count_delta"])
        for rank, (score, dist, left, right, cmp) in enumerate(top_candidates, 1):
            mismatches = "; ".join(f"{a}->{b}@{ov:.2f}" for a, b, ov in cmp["class_mismatches"])
            writer.writerow([rank, score, dist, left, right, mismatches, cmp["unmatched_a"], cmp["unmatched_b"], cmp["count_delta"]])

    preview_records = set()
    for _, _, left, right, _ in top_candidates[:80]:
        preview_records.add(left)
        preview_records.add(right)
    for stem in sorted(preview_records):
        draw_preview(image_paths[stem], by_stem[stem], class_to_id, preview_dir / f"{stem}.jpg")
    for rank, (_, _, left, right, _) in enumerate(top_candidates[:80], 1):
        make_pair_preview(preview_dir / f"{left}.jpg", preview_dir / f"{right}.jpg", pair_dir / f"{rank:03d}_{left}__{right}.jpg")

    write_clean_yolo(records, classes, clean_label_dir)

    rows = []
    for rank, (score, dist, left, right, cmp) in enumerate(top_candidates[:80], 1):
        mismatches = html.escape("; ".join(f"{a} -> {b} (IoU {ov:.2f})" for a, b, ov in cmp["class_mismatches"]) or "none")
        img = f"similar_pairs/{rank:03d}_{left}__{right}.jpg"
        rows.append(
            f"<tr><td>{rank}</td><td>{score:.1f}</td><td>{dist}</td><td>{html.escape(left)}</td>"
            f"<td>{html.escape(right)}</td><td>{mismatches}</td><td>{cmp['unmatched_a']}</td>"
            f"<td>{cmp['unmatched_b']}</td><td><img src='{html.escape(img)}'></td></tr>"
        )

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Main Dataset Annotation QC</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d9e2ec; padding: 6px 8px; vertical-align: top; font-size: 13px; }}
th {{ background: #f0f4f8; position: sticky; top: 0; }}
img {{ max-width: 100%; }}
.summary {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin-bottom: 18px; }}
.summary div {{ border: 1px solid #d9e2ec; padding: 12px; border-radius: 6px; background: #f8fafc; }}
</style>
</head>
<body>
<h1>主数据集标注质检报告</h1>
<div class="summary">
<div><strong>图片数</strong><br>{len(records)}</div>
<div><strong>目标数</strong><br>{sum(class_counts.values())}</div>
<div><strong>规则异常</strong><br>{len(rule_issues)}</div>
<div><strong>相似图疑似冲突</strong><br>{len(top_candidates)}</div>
</div>
<p>建议优先查看 score 高的相似图对。左、右图都叠加了当前 XML 标注；如果画面状态实际一致但标签不同，再人工修改对应 XML。</p>
<p>CSV 文件：class_counts.csv、rule_issues.csv、duplicate_box_issues.csv、xml_yolo_mismatch.csv、similar_annotation_conflicts.csv。</p>
<p>YOLO 范围检查：yolo_range_issues.csv。minor_rounding 通常是贴边框四舍五入导致，已在清洗标签副本中裁剪到合法范围。</p>
<p>已根据 XML 重新生成合法 YOLO 标签副本：cleaned_yolo_labels_from_xml/。</p>
<table>
<thead><tr><th>rank</th><th>score</th><th>hash距离</th><th>图A</th><th>图B</th><th>同位置类别冲突</th><th>A未匹配</th><th>B未匹配</th><th>预览</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>"""
    (report_dir / "index.html").write_text(html_doc, encoding="utf-8")

    print(f"dataset={dataset}")
    print(f"report={report_dir}")
    print(f"images={len(records)} objects={sum(class_counts.values())} classes={len(class_counts)}")
    print(f"rule_issues={len(rule_issues)} duplicate_issues={len(duplicate_issues)} xml_yolo_mismatch={len(xml_yolo_issues)} yolo_range_issues={len(yolo_range_issues)}")
    print(f"similar_conflict_candidates={len(top_candidates)}")


if __name__ == "__main__":
    main()
