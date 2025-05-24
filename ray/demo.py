# import ray
from ultralytics import YOLO

# ray.init(address="ray://localhost:10001")
model = YOLO("yolo11n.pt")
result_grid = model.tune(data="coco8.yaml", device=0, use_ray=True)
