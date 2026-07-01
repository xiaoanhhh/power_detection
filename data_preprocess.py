import os
import cv2
import shutil
from pathlib import Path
from PIL import Image
import numpy as np


def check_bbox_validity(bbox, img_w, img_h):
    """检查边界框是否合法"""
    x_center, y_center, w, h = bbox
    if w <= 0 or h <= 0:
        return False
    if x_center - w/2 < 0 or x_center + w/2 > img_w:
        return False
    if y_center - h/2 < 0 or y_center + h/2 > img_h:
        return False
    return True


def clip_bbox(bbox):
    """裁剪边界框到[0,1]范围内"""
    x_center, y_center, w, h = bbox
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return [x_center, y_center, w, h]


def remove_duplicate_boxes(boxes, iou_threshold=0.95):
    """移除重复的边界框"""
    if len(boxes) <= 1:
        return boxes
    
    keep = []
    boxes = sorted(boxes, key=lambda x: x[3] * x[4], reverse=True)
    
    while len(boxes) > 0:
        current = boxes.pop(0)
        keep.append(current)
        
        remaining = []
        for box in boxes:
            if box[0] != current[0]:
                remaining.append(box)
                continue
            
            iou = calculate_iou(current[1:], box[1:])
            if iou < iou_threshold:
                remaining.append(box)
        
        boxes = remaining
    
    return keep


def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    x1_min, x1_max = x1 - w1/2, x1 + w1/2
    y1_min, y1_max = y1 - h1/2, y1 + h1/2
    x2_min, x2_max = x2 - w2/2, x2 + w2/2
    y2_min, y2_max = y2 - h2/2, y2 + h2/2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def clean_dataset(images_dir, labels_dir, output_dir):
    """清洗数据集"""
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    image_files = list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.png'))
    
    cleaned_count = 0
    issue_count = 0
    
    for img_path in image_files:
        label_path = Path(labels_dir) / (img_path.stem + '.txt')
        
        if not label_path.exists():
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]
            
            bbox = clip_bbox(bbox)
            
            if check_bbox_validity(bbox, 1.0, 1.0):
                boxes.append([class_id] + bbox)
        
        boxes = remove_duplicate_boxes(boxes)
        
        if len(boxes) != len(lines):
            issue_count += 1
        
        output_label_path = Path(output_dir) / 'labels' / (img_path.stem + '.txt')
        with open(output_label_path, 'w') as f:
            for box in boxes:
                f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
        
        shutil.copy(img_path, os.path.join(output_dir, 'images', img_path.name))
        cleaned_count += 1
    
    print(f"清洗完成: 处理了 {cleaned_count} 张图片，发现 {issue_count} 个有问题的标注")
    return cleaned_count


def analyze_dataset(labels_dir, num_classes=16):
    """分析数据集"""
    class_counts = [0] * num_classes
    small_objects = 0
    total_objects = 0
    
    label_files = list(Path(labels_dir).glob('*.txt'))
    
    for label_path in label_files:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            w, h = float(parts[3]), float(parts[4])
            area = w * h
            
            if class_id < num_classes:
                class_counts[class_id] += 1
            total_objects += 1
            
            if area < 0.01:
                small_objects += 1
    
    print(f"总目标数: {total_objects}")
    print(f"小目标数: {small_objects} ({small_objects/total_objects*100:.1f}%)")
    print(f"各类别分布:")
    for i, count in enumerate(class_counts):
        print(f"  类别 {i}: {count}")
    
    return class_counts


if __name__ == '__main__':
    images_dir = 'data/images'
    labels_dir = 'data/labels'
    output_dir = 'data/cleaned'
    
    print("开始数据清洗...")
    clean_dataset(images_dir, labels_dir, output_dir)
    
    print("\n数据集分析:")
    analyze_dataset(os.path.join(output_dir, 'labels'))
