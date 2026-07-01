import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="prepared_yolov8s_clustered/data.yaml")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="training_runs/main_yolov8s")
    parser.add_argument("--name", default="yolov8s_clustered_strict")
    args = parser.parse_args()

    from ultralytics import YOLO

    data = Path(args.data)
    if not data.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data}")
    data = data.resolve()

    project = Path(args.project)
    if not project.is_absolute():
        project = Path.cwd() / project

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name=args.name,
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        patience=40,
        close_mosaic=20,
        hsv_h=0.01,
        hsv_s=0.35,
        hsv_v=0.25,
        degrees=3.0,
        translate=0.08,
        scale=0.35,
        shear=0.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.0,
        mosaic=1.0,
        mixup=0.05,
        copy_paste=0.0,
        cache=False,
        amp=True,
        seed=42,
        deterministic=False,
        plots=True,
    )


if __name__ == "__main__":
    main()
