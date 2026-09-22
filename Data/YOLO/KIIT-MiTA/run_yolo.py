from ultralytics import YOLO

if __name__ == "__main__":
    # Load a pretrained model. Change the file name to the model you want.
    model = YOLO("yolov8n.pt")

    # Train on GPU 0. data.yaml is in this folder.
    results = model.train(data="data.yaml", epochs=100, imgsz=640, device=0)
