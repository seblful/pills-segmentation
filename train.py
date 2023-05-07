import os
import ultralytics
from ultralytics import YOLO
import torch

# Define constant variables
HOME = os.getcwd()
DATASET_PATH = os.path.join(HOME, 'datasets') # dataset prepared for training

def main():
    # Check libraries
    ultralytics.checks()
    print(f"Torch version is {torch.__version__}.")

    # Instantiate device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Current device is {device}")

    # Load a model
    model = YOLO("yolov8l-seg.yaml")  # build a new model from scratch
    model = YOLO("yolov8l-seg.pt")  # load a pretrained model

    # Train the model
    model.train(data=DATASET_PATH + '/data.yaml', epochs=20, imgsz=640, batch=1, device=device)

    # Evaluate model performance on the validation set
    metrics = model.val()

    # Export the model to ONNX format
    success = model.export(format="onnx")    


if __name__ == "__main__":
    main()