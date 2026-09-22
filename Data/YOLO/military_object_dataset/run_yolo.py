from ultralytics import YOLO

if __name__ == "__main__":
    # Load a pretrained model. Change the file name to the model you want.
    model = YOLO("yolo26m.pt")

    # Train on GPU 0. data.yaml is in this folder.
    results = model.train(data="data.yaml", epochs=100, imgsz=640, device=0)
