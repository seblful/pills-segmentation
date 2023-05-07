# preparing_data.py
from utils import augment_images_and_labels, check_if_labels_right,\
                  partitionate_data, create_data_yaml

import os
import imgaug.augmenters as iaa
import argparse

# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

# Get an arg for factor_tablets
parser.add_argument("--factor_tablets", 
                     default=3, 
                     type=int, 
                     help="Factor how many images with tablets should be after augmentation.")

# Get an arg for factor_capsules
parser.add_argument("--factor_capsules", 
                     default=3, 
                     type=int, 
                     help="Factor how many images with capsules should be after augmentation.")

# Get an arg for vizualize_augmentation
parser.add_argument("--vizualize_augmentation", 
                     default=True, 
                     type=bool, 
                     help="Vizualize how image is changing after several step of augmentation.")

# Get an arg for check_labels
parser.add_argument("--check_labels", 
                     default=True, 
                     type=bool, 
                     help="Vizualize if polygon is right after augmentation on several random images.")

# Get an arg for number_images_for_check
parser.add_argument("--number_images_for_check", 
                     default=5,
                     type=int,
                     help="How many images should check after augmentation. Works only if check_labels==True.")


# Get our arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
FACTOR_TABLETS = args.factor_tablets
FACTOR_CAPSULES = args.factor_capsules
VIZUALIZE_AUGMENTATION = args.vizualize_augmentation
CHECK_LABELS = args.check_labels
NUMBER_IMAGES_FOR_CHECK = args.number_images_for_check


def main():
    # Create path variables and create folder for dataset
    # Define constant variables
    HOME = os.getcwd()
    RAW_DATA_PATH = os.path.join(HOME, 'raw_data') # data from label studio
    DATA_PATH = os.path.join(HOME, 'data') # augmented data
    DATASET_PATH = os.path.join(HOME, 'datasets') # dataset prepared for training

    # Make folder for DATA_PATH
    os.makedirs(DATA_PATH, exist_ok=True)
    # Make folder for DATASET_PATH
    os.makedirs(DATASET_PATH, exist_ok=True)
        
    print(f"\nHOME directory is {HOME}.")
    print(f"Folder with labelled images is {RAW_DATA_PATH}.")
    print(f"Folder with augmented images is {DATA_PATH}.")
    print(f"Folder with prepared dataset is {DATASET_PATH}.")


    # Define the augmenter
    augmenter = iaa.SomeOf((4, None), [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.7),
                iaa.Affine(rotate=(-20, 20)),
                iaa.Affine(shear=(-16, 16)),
                iaa.Dropout(p=(0, 0.1)),
                iaa.ImpulseNoise(0.1),
                iaa.GaussianBlur(sigma=(0.0, 2.0)),
                iaa.MultiplyAndAddToBrightness(mul=(0.7, 1.3), add=(-20, 20)),
                iaa.MultiplyHueAndSaturation(mul_hue=(0.7, 1.3)),
                iaa.Add((-40, 40), per_channel=0.5),
                iaa.GammaContrast((0.5, 2.0))
    ])
    print("Augmenter has created.")


    # Augmenting and save images and labels
    augment_images_and_labels(RAW_DATA_PATH, 
                              DATA_PATH, 
                              augmenter, 
                              augmented_factor_tablets=FACTOR_TABLETS, 
                              augmented_factor_capsules=FACTOR_CAPSULES, 
                              visualize=VIZUALIZE_AUGMENTATION)
    print("Images has augmented.")
    if CHECK_LABELS==True:
        # Check if labels after augmentation is right
        check_if_labels_right(DATA_PATH, number_of_images=NUMBER_IMAGES_FOR_CHECK)
        print("Labels has checked, see image 'check_augmented_polygons.jpg' in directory.")

    # Create train, validation and test data and put them to dataset folder
    partitionate_data(DATA_PATH, DATASET_PATH, train_size=0.75)
    print("Train, validation, test dataset has created.")

    # Create data.yaml
    create_data_yaml(DATA_PATH, DATASET_PATH)
    print("data.yaml file has created.")


if __name__ == "__main__":
    main()