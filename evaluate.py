import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_model(model_path, data_yaml, imgsz=960, batch=4, device=''):
    """评估模型性能"""
    model = YOLO(model_path)
    
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        save_json=True,
        save_hybrid=False,
        conf=0.001,
        iou=0.6,
        max_det=300,
        half=False,
        dnn=False,
        plots=True,
        rect=False,
        split='val',
    )
    
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    
    return results


def get_predictions(model_path, images_dir, conf_threshold=0.25, iou_threshold=0.45):
    """获取预测结果"""
    model = YOLO(model_path)
    
    image_files = list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.png'))
    
    all_predictions = []
    
    for img_path in image_files:
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    pred = {
                        'image': img_path.name,
                        'class': int(box.cls[0]),
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].cpu().numpy().tolist()
                    }
                    all_predictions.append(pred)
    
    return all_predictions


def calculate_confusion_matrix(model_path, data_yaml, class_names, conf_threshold=0.25):
    """计算混淆矩阵"""
    model = YOLO(model_path)
    
    results = model.val(
        data=data_yaml,
        conf=conf_threshold,
        plots=False,
        verbose=False
    )
    
    cm = results.confusion_matrix.matrix
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    
    print("混淆矩阵已保存: confusion_matrix.png")
    
    return cm


def analyze_class_performance(results, class_names):
    """分析每个类别的性能"""
    print("\n各类别性能分析:")
    print("-" * 80)
    print(f"{'类别':<20} {'P':<8} {'R':<8} {'mAP50':<8} {'mAP50-95':<8}")
    print("-" * 80)
    
    for i, name in enumerate(class_names):
        p = results.box.p[i] if i < len(results.box.p) else 0
        r = results.box.r[i] if i < len(results.box.r) else 0
        ap50 = results.box.ap50[i] if i < len(results.box.ap50) else 0
        ap = results.box.ap[i] if i < len(results.box.ap) else 0
        
        print(f"{name:<20} {p:<8.3f} {r:<8.3f} {ap50:<8.3f} {ap:<8.3f}")
    
    print("-" * 80)


def plot_pr_curve(results, class_names, output_path='pr_curve.png'):
    """绘制PR曲线"""
    plt.figure(figsize=(10, 8))
    
    for i, name in enumerate(class_names):
        if i < len(results.box.p_curve):
            precision = results.box.p_curve[i]
            recall = results.box.r_curve[i]
            plt.plot(recall, precision, label=f'{name}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PR曲线已保存: {output_path}")


def detect_small_objects(model_path, images_dir, output_dir='output/small_objects', 
                         conf_threshold=0.25, area_threshold=0.01):
    """检测小目标并可视化"""
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO(model_path)
    image_files = list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.png'))
    
    small_obj_count = 0
    
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            verbose=False
        )
        
        has_small = False
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    area = (x2 - x1) * (y2 - y1) / (w * h)
                    
                    if area < area_threshold:
                        has_small = True
                        small_obj_count += 1
                        
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        label = f'small: {float(box.conf[0]):.2f}'
                        cv2.putText(img, label, (int(x1), int(y1)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if has_small:
            output_path = os.path.join(output_dir, img_path.name)
            cv2.imwrite(output_path, img)
    
    print(f"小目标检测完成，共检测到 {small_obj_count} 个小目标")
    print(f"可视化结果已保存到: {output_dir}")


if __name__ == '__main__':
    class_names = [
        'switch-left', 'switch-center', 'switch-right',
        'platen-on', 'platen-off',
        'red', 'green', 'red-green', 'green-green-red',
        'transformer-on', 'transformer-off',
        'button-on', 'button-off',
        'display-on', 'display-off',
        'other'
    ]
    
    model_path = 'runs/power_panel_yolov8s/weights/best.pt'
    data_yaml = 'data.yaml'
    
    print("开始模型评估...")
    results = evaluate_model(model_path, data_yaml, imgsz=960)
    
    analyze_class_performance(results, class_names)
    plot_pr_curve(results, class_names)
