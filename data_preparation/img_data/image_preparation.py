import os
import copy
import json
import shutil
import argparse

import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from tqdm import tqdm
from natsort import natsorted
from IPython.display import display
from typing import Optional, List, Dict, Tuple


def crop_image(img: np.ndarray, crop_percent: int = 10) -> np.ndarray:
    """
    Crop 10% of the image length from both left and right sides.

    Args:
        img: Input image (NumPy array).

    Returns:
        Cropped image (NumPy array).
    """
    if img is None:
        raise ValueError("Input image is None.")

    # Get the dimensions of the input image
    height, width, _ = img.shape

    # Calculate the number of pixels to crop from each side (10% of the width)
    crop_width = width // crop_percent

    # Crop the image
    cropped_img = img[:, crop_width:-crop_width]

    return cropped_img


def resize_image(img: np.ndarray, new_shape: Tuple = (160, 224)) -> np.ndarray:
    return cv2.resize(img, (new_shape[1], new_shape[0]))


def select_major_emotion(emotion_vector: np.array) -> np.array:
    """
    Returns the major emotion based on significance while retaining the original order.

    :param emotion_vector: List of binary values representing emotions.
    :return: A new binary vector with only the major emotion set to 1.
    """

    # Original order of emotions
    original_order = ["Curiosity", "Uncertainty", "Excitement", "Happiness",
                     "Surprise", "Disgust", "Fear", "Frustration"]

    # New order of significance
    new_order = ["Happiness", "Excitement", "Disgust", "Fear", "Frustration",
                 "Surprise", "Curiosity", "Uncertainty"]

    # Extract emotions from initial vector
    indicated_emotions = [original_order[i] for i, e in enumerate(emotion_vector) if e == 1]

    # Determine the most significant emotion from the new order
    for emotion in new_order:
        if emotion in indicated_emotions:
            major = emotion
            break

    # Create and return the new vector based on original order
    return np.array([1 if emotion == major else 0 for emotion in original_order])


def prepare_image_data(crop_percent: int, new_shape: Tuple, major_emotion: bool) -> None:
    with open(os.path.join(os.getcwd(), 'data_path_config.json'), 'r') as file:
        config = json.load(file)

    path_to_save_datasets = os.path.join(os.getcwd(), 'datasets')
    os.makedirs(path_to_save_datasets, exist_ok=True)

    specific_path = os.path.join(path_to_save_datasets,
                                 f"{crop_percent}_perc_crop_{'_'.join([str(el) for el in new_shape])}_shape")
    if major_emotion:
        specific_path += "_major_emotion"
    os.makedirs(specific_path, exist_ok=True)

    data_split_types = [os.path.basename(el).split('.')[0] for el in
                        glob.glob(os.path.join(config["datasets_videos"], "*.csv"))]

    for split_type in data_split_types:
        dataset_name = f"{split_type}_dataset"

        df = pd.read_csv(os.path.join(config["datasets_images"], f"{split_type}.csv"))
        image_folder_path_list = df["file_path"].to_list()

        dataset_data = []
        for img_idx, img_folder in enumerate(tqdm(image_folder_path_list, desc=f"Processing {split_type} data")):
            _label = np.array(df.iloc[img_idx, 1:-1].values, dtype=int)
            if np.max(_label) != 0:
                if major_emotion:
                    _label = select_major_emotion(emotion_vector=_label)

                if split_type == "train":
                    for _img_path in natsorted(glob.glob(os.path.join(img_folder, "*.jpg")))[2:-2]:
                        img = cv2.imread(_img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # Crop image
                        img = crop_image(img=img, crop_percent=crop_percent)

                        # Resize image
                        img = resize_image(img=img, new_shape=new_shape)

                        dataset_data.append([torch.from_numpy(img).permute(2, 0, 1), _label])
                else:
                    _img_path = natsorted(glob.glob(os.path.join(img_folder, "*.jpg")))[4]
                    img = cv2.imread(_img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Crop image
                    img = crop_image(img=img, crop_percent=crop_percent)

                    # Resize image
                    img = resize_image(img=img, new_shape=new_shape)

                    dataset_data.append([torch.from_numpy(img).permute(2, 0, 1), _label])

        # Save 'processed_model_data' data
        torch.save(
            dataset_data,
            os.path.join(specific_path, f"{dataset_name}.pt"),
        )


if __name__ == '__main__':
    # Read input parameters
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-crop_percent', '--crop_percent', type=int,
                        required=False, default=10,
                        help="Percentage of image's width to crop from sides")
    parser.add_argument('-new_shape', '--new_shape', type=tuple,
                        required=False, default=(160, 224),
                        help="New image shape")
    parser.add_argument('-major_emotion', '--major_emotion', type=bool,
                        required=False, default=True,
                        help="Percentage of image's width to crop from sides")

    args = parser.parse_args()

    prepare_image_data(
        crop_percent=args.crop_percent, new_shape=args.new_shape,
        major_emotion=args.major_emotion
    )
