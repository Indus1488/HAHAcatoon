from ultralytics import YOLO


def main():

    print("=" * 60)
    print("AI PCB Defect Detection - Final Model Training")
    print("=" * 60)

    # Load pretrained YOLOv8 Nano model
    model = YOLO("yolov8n.pt")

    # Train model
    model.train(
        data="datasets/data.yaml",
        epochs=52,
        imgsz=640,
        batch=8,
        workers=0,
        device="0",
        optimizer="auto",
        patience=30,
        close_mosaic=10,
        project="runs/detect",
        name="pcb_yolov8_final",
        exist_ok=True,
        save=True,
        save_period=10,
        verbose=True,
        plots=True,
        cache=False,
        seed=42
    )

    print("\nTraining Completed Successfully!")
    print("Best model saved at:")
    print("runs/detect/pcb_yolov8_final/weights/best.pt")


if __name__ == "__main__":
    main()