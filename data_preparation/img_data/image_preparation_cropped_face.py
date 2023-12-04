import os
import json
import argparse

import cv2
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from facenet_pytorch import MTCNN

from tqdm import tqdm
from natsort import natsorted
from typing import Optional, List, Tuple
from sklearn.model_selection import train_test_split


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


def crop_face_from_image(mtcnn, img: np.ndarray, extend_box_dimensions: float = 0.05) -> Optional[np.ndarray]:
    _boxes, *_ = mtcnn.detect(img)

    if _boxes is None:
        return None

    # Convert the coordinates to integers
    box = [int(coord) for coord in _boxes[0]]

    # Extract the corner points of the bounding box
    x1, y1, x2, y2 = box

    # # Draw the expanded bounding box on the image
    # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # (0, 255, 0) is the color (green), and 2 is the thickness
    # plt.imshow(img)
    # plt.show()

    # Calculate the width and height of the box
    width = x2 - x1
    height = y2 - y1

    # Increase the dimensions by 5%
    x1 = np.maximum(int(x1 - width * extend_box_dimensions), 0)
    y1 = np.maximum(int(y1 - height * (extend_box_dimensions / 2)), 0)
    x2 = np.maximum(int(x2 + width * extend_box_dimensions), 0)
    y2 = np.maximum(int(y2 + height * (extend_box_dimensions / 2)), 0)

    # Crop the image
    cropped_face = img[y1:y2, x1:x2]
    # print(cropped_img.shape)

    return cropped_face


def resize_image(img: np.ndarray, new_shape: Tuple = (160, 160)) -> np.ndarray:
    """
    Resize an image to the target size without distortion.
    Pads with 0s if needed.

    Parameters:
        img (numpy.array): The input image.
        target_size (tuple): The desired dimensions as (height, width).

    Returns:
        numpy.array: The resized image.
    """

    # Calculate aspect ratio
    aspect = img.shape[1] / float(img.shape[0])
    if (aspect > 1):
        # landscape orientation - wide image
        res = int(new_shape[0] / aspect)
        scaled_img = cv2.resize(img, (new_shape[1], res))
    else:
        # portrait orientation - tall image
        res = int(new_shape[1] * aspect)
        scaled_img = cv2.resize(img, (res, new_shape[0]))

    # Padding to get the target shape
    delta_w = new_shape[1] - scaled_img.shape[1]
    delta_h = new_shape[0] - scaled_img.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    padded_img = cv2.copyMakeBorder(scaled_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return padded_img


def down_sample_class(data: List, to_keep: int = 100) -> List:
    target_label = np.array([0, 0, 0, 1, 0, 0, 0, 0])

    # Filter out samples of the target class
    target_class_samples = [sample for sample in data if np.array_equal(sample[1], target_label)]

    # Randomly select a subset of samples from the target class
    selected_samples = random.sample(target_class_samples, to_keep)

    # Construct the downsampled dataset
    downsampled_data = [sample for sample in data if not np.array_equal(sample[1], target_label)]
    downsampled_data.extend(selected_samples)
    random.shuffle(downsampled_data)

    return downsampled_data


def prepare_image_data(crop_percent: int, new_shape: Tuple, major_emotion: bool) -> None:
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    )

    with open(os.path.join(os.getcwd(), 'data_path_config.json'), 'r') as file:
        config = json.load(file)

    path_to_save_datasets = os.path.join(os.getcwd(), 'datasets')
    os.makedirs(path_to_save_datasets, exist_ok=True)

    specific_path = os.path.join(path_to_save_datasets,
                                 f"{crop_percent}_perc_crop_{'_'.join([str(el) for el in new_shape])}_shape")
    if major_emotion:
        specific_path += "_major_emotion"

    specific_path += "_cropped_face"
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
                    files_to_process = natsorted(glob.glob(os.path.join(img_folder, "*.jpg")))[2:-2]
                else:
                    files_to_process = natsorted(glob.glob(os.path.join(img_folder, "*.jpg")))[4:6]

                for _img_path in files_to_process:
                    img = cv2.imread(_img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Extract face from image
                    img = crop_face_from_image(
                        mtcnn,
                        img=img
                    )

                    if img is None:
                        continue

                    # Resize image
                    img = resize_image(img=img)

                    dataset_data.append([torch.from_numpy(img).permute(2, 0, 1), _label, _img_path])

        # Down-sample class
        if split_type == "train":
            dataset_data = down_sample_class(data=dataset_data, to_keep=330)

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
                        required=False, default=(160, 160),
                        help="New image shape")
    parser.add_argument('-major_emotion', '--major_emotion', type=bool,
                        required=False, default=True,
                        help="Percentage of image's width to crop from sides")

    args = parser.parse_args()

    prepare_image_data(
        crop_percent=args.crop_percent, new_shape=args.new_shape,
        major_emotion=args.major_emotion
    )
