# Project Overview
Instance Pills Segmentation** using YOLOv8 with Preparing and Augmenting Data is a computer vision project that aims to accurately detect and segment instances of pills (capsules and tablets) in images. The project leverages the **YOLOv8** object detection algorithm, which is known for its fast and accurate object detection capabilities. To improve the accuracy of the model, the project also includes a comprehensive pipeline for data preparation and augmentation.


# Data Preparation and Augmentation
The project includes a **preparing_data.py** script that is responsible for preparing the data and augmenting it. The script uses **Label Studio** data to generate augmented images with randomly transformed polygons, rotation, and noise. The **utils.py** file contains helper functions for preparing and augmenting the data, including functions for loading and saving images, reading label-studio data, generating polygons, and augmenting images.

Example of usage:

```python preparing_data.py --factor_tablets 3 --factor_capsules 3 --vizualize_augmentation True --check_labels True --number_images_for_check 5```

The data preparation and augmentation pipeline is crucial for improving the accuracy of the **YOLOv8** model. By generating augmented data, the model is exposed to a wider range of images, making it more robust to variations in lighting, rotation, and other factors.


# Training the YOLO Model
The **train.py** script is used to train the YOLO model using the prepared and augmented data. It loads the images and labels, initializes the YOLO model, and performs the training, saving the best model weights to disk.

Example of usage:

```python train.py --model_size m --epochs 30 --batch_size 32```

# Project Structure
The project includes several files and folders, including:

**README.md**: A file that provides instructions and information about the project.

**.gitignore**: A file that specifies which files and directories to exclude from version control.

**requirements.txt**: A file that lists the necessary packages to run the scripts.

**raw_data**: A folder that contains the Label Studio data used for data preparation and augmentation.

**utils.py**: A file that contains helper functions for preparing and augmenting the data.

**preparing_data.py**: A script for preparing and augmenting the data.

**train.py**: A script for training the YOLO model.

**example_polygons.jpg**: An example of augmented polygons generated by the preparing_data.py script.

**check_augmented_polygons.jpg**: An image used to check if the augmentation was done correctly.


# Conclusion
**Instance Pills Segmentation** using YOLOv8 with Preparing and Augmenting Data is a comprehensive computer vision project that offers a complete solution for detecting and segmenting pills in images. The project's focus on data preparation and augmentation is critical for improving the accuracy of the YOLO model, making it more robust to variations in lighting, rotation, and other factors. With this project, you can easily train your own YOLO model for instance pills segmentation with augmented data.
