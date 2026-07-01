import os
import yaml
from ultralytics import YOLO
import torch


def create_data_yaml(train_dir, val_dir, test_dir, nc, names, output_path='data.yaml'):
    """创建数据集配置文件"""
    data = {
        'train': train_dir,
        'val': val_dir,
        'test': test_dir,
        'nc': nc,
        'names': names
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)
    
    print(f"数据集配置文件已创建: {output_path}")
    return output_path


def train_yolov8(model_size='s', data_yaml='data.yaml', epochs=30, imgsz=960, batch=4, 
                 project='runs', name='yolov8_train', pretrained=True, resume=False):
    """训练YOLOv8模型"""
    
    model_path = f'yolov8{model_size}.pt' if pretrained else f'yolov8{model_size}.yaml'
    
    model = YOLO(model_path)
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        exist_ok=True,
        patience=40,
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=1.0,
        mixup=0.05,
        copy_paste=0.0,
        auto_augment='randaugment',
        erasing=0.4,
        crop_fraction=1.0,
        amp=True,
        fraction=1.0,
        profile=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True,
        save=True,
        save_period=-1,
        cache=False,
        device='',
        workers=8,
        close_mosaic=20,
        resume=resume,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=True,
        close_mosaic_nms=False,
    )
    
    return results


def train_with_freeze(model_size='s', data_yaml='data.yaml', epochs=30, imgsz=960, batch=4,
                      project='runs', name='yolov8_freeze'):
    """分阶段训练：先冻结骨干网络，再解冻微调"""
    
    model = YOLO(f'yolov8{model_size}.pt')
    
    print("第一阶段：冻结骨干网络训练...")
    results1 = model.train(
        data=data_yaml,
        epochs=10,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=f'{name}_stage1',
        exist_ok=True,
        freeze=10,
        patience=20,
        cos_lr=True,
        close_mosaic=5,
    )
    
    print("第二阶段：全网络微调...")
    best_model_path = os.path.join(project, f'{name}_stage1', 'weights', 'best.pt')
    model = YOLO(best_model_path)
    
    results2 = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=f'{name}_stage2',
        exist_ok=True,
        patience=30,
        cos_lr=True,
        close_mosaic=15,
        lr0=0.001,
    )
    
    return results1, results2


def export_model(model_path, format='onnx', imgsz=960):
    """导出模型"""
    model = YOLO(model_path)
    
    exported_path = model.export(
        format=format,
        imgsz=imgsz,
        half=False,
        opset=17,
        simplify=True,
    )
    
    print(f"模型已导出: {exported_path}")
    return exported_path


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
    
    data_yaml = create_data_yaml(
        train_dir='data/train/images',
        val_dir='data/val/images',
        test_dir='data/test/images',
        nc=16,
        names=class_names,
        output_path='data.yaml'
    )
    
    print("开始训练YOLOv8...")
    results = train_yolov8(
        model_size='s',
        data_yaml=data_yaml,
        epochs=30,
        imgsz=960,
        batch=4,
        project='runs',
        name='power_panel_yolov8s'
    )
    
    print("训练完成！")
