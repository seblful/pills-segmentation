# train.py

import os
import ultralytics
from ultralytics import YOLO
import torch
import argparse

# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

# Get an arg for yolo model_size
parser.add_argument("--model_size", 
                     default='m', 
                     type=str, 
                     help="Size of yolo segmentation model (n, s, m, l, xl)")

# Get an arg for epochs
parser.add_argument("--epochs", 
                     default=10, 
                     type=int,
                     help="the number of epochs to train for")

# Get an arg for batch_size
parser.add_argument("--batch_size", 
                     default=32, 
                     type=int, 
                     help="the number of batch size to train for")

# Get our arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
MODEL_SIZE = args.model_size
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

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


    # Model type
    model_yaml = f"yolov8{MODEL_SIZE}-seg.yaml"
    model_type = f"yolov8{MODEL_SIZE}-seg.pt"

    # Load a model
    model = YOLO(model_yaml)  # build a new model from scratch
    model = YOLO(model_type)  # load a pretrained model

    # Train the model
    model.train(data=DATASET_PATH + '/data.yaml', epochs=EPOCHS, imgsz=640, batch=BATCH_SIZE)

    # Evaluate model performance on the validation set
    metrics = model.val()

    # Export the model to ONNX format
    success = model.export(format="onnx")    


if __name__ == "__main__":
    main()