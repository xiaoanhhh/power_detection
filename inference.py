import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time


def load_model(model_path):
    """加载模型"""
    model = YOLO(model_path)
    print(f"模型已加载: {model_path}")
    return model


def predict_single_image(model, image_path, conf_threshold=0.25, iou_threshold=0.45, 
                         save_result=True, output_dir='output'):
    """对单张图片进行预测"""
    if save_result:
        os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        save=save_result,
        project=output_dir,
        name='predict',
        exist_ok=True,
        verbose=False
    )
    
    inference_time = time.time() - start_time
    
    result = results[0]
    boxes = result.boxes
    
    predictions = []
    if boxes is not None:
        for box in boxes:
            pred = {
                'class': int(box.cls[0]),
                'class_name': model.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox_xyxy': box.xyxy[0].cpu().numpy().tolist(),
                'bbox_xywh': box.xywh[0].cpu().numpy().tolist()
            }
            predictions.append(pred)
    
    print(f"检测到 {len(predictions)} 个目标")
    print(f"推理时间: {inference_time:.3f}s")
    
    for i, pred in enumerate(predictions):
        print(f"  目标 {i+1}: {pred['class_name']} (置信度: {pred['confidence']:.3f})")
    
    return predictions, inference_time


def predict_batch(model, images_dir, conf_threshold=0.25, iou_threshold=0.45,
                  save_result=True, output_dir='output'):
    """批量预测图片"""
    if save_result:
        os.makedirs(output_dir, exist_ok=True)
    
    image_files = list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.png'))
    print(f"找到 {len(image_files)} 张图片")
    
    all_predictions = []
    total_time = 0
    
    for i, img_path in enumerate(image_files):
        print(f"\n处理第 {i+1}/{len(image_files)} 张: {img_path.name}")
        
        predictions, inference_time = predict_single_image(
            model, str(img_path), conf_threshold, iou_threshold,
            save_result=False
        )
        
        all_predictions.append({
            'image': img_path.name,
            'predictions': predictions,
            'inference_time': inference_time
        })
        
        total_time += inference_time
    
    print(f"\n批量预测完成")
    print(f"总时间: {total_time:.3f}s")
    print(f"平均每张: {total_time/len(image_files):.3f}s")
    
    return all_predictions


def predict_video(model, video_path, conf_threshold=0.25, iou_threshold=0.45,
                  save_result=True, output_dir='output'):
    """对视频进行预测"""
    if save_result:
        os.makedirs(output_dir, exist_ok=True)
    
    results = model.predict(
        source=video_path,
        conf=conf_threshold,
        iou=iou_threshold,
        save=save_result,
        project=output_dir,
        name='video_predict',
        exist_ok=True,
        stream=True
    )
    
    return results


def draw_predictions(image, predictions, class_names, colors=None):
    """在图片上绘制预测结果"""
    if colors is None:
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(class_names), 3)).tolist()
    
    for pred in predictions:
        x1, y1, x2, y2 = map(int, pred['bbox_xyxy'])
        class_id = pred['class']
        confidence = pred['confidence']
        class_name = pred['class_name']
        
        color = colors[class_id]
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label = f'{class_name}: {confidence:.2f}'
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def count_objects_by_class(predictions, class_names):
    """按类别统计目标数量"""
    counts = {name: 0 for name in class_names}
    
    for pred in predictions:
        class_name = pred['class_name']
        if class_name in counts:
            counts[class_name] += 1
    
    return counts


def export_predictions_to_txt(predictions, output_path):
    """导出预测结果为TXT格式（YOLO格式）"""
    with open(output_path, 'w') as f:
        for pred in predictions:
            class_id = pred['class']
            x_center, y_center, w, h = pred['bbox_xywh']
            confidence = pred['confidence']
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {confidence:.6f}\n")
    
    print(f"预测结果已导出: {output_path}")


def real_time_detection(model, source=0, conf_threshold=0.25, iou_threshold=0.45):
    """实时检测（摄像头）"""
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("按 'q' 退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        
        annotated_frame = results[0].plot()
        
        cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('YOLO Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model_path = 'runs/power_panel_yolov8s/weights/best.pt'
    image_path = 'data/test/images/sample.jpg'
    
    print("加载模型...")
    model = load_model(model_path)
    
    print("\n单张图片预测:")
    predictions, _ = predict_single_image(model, image_path, conf_threshold=0.25)
    
    print("\n按类别统计:")
    counts = count_objects_by_class(predictions, model.names)
    for name, count in counts.items():
        if count > 0:
            print(f"  {name}: {count}")
