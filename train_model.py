from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")

model.train(
    data="data",
    epochs=20,
    imgsz=224,
    batch=8,
    name="packaging_classifier"
)
