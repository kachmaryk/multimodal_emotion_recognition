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

from tqdm import tqdm
from typing import Optional, List
from IPython.display import display


# Function to calculate the Laplacian variance of an image
def laplacian_variance(image: np.ndarray) -> float:
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return laplacian.var()


def select_best_frames(video_path: str, path_to_save: str, num_frames: int = 10) -> str:
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    # Get video info: duration and frame count
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # print(f"FPS: {fps}")
    # print(f"Total Frames: {total_frames}")

    # Calculate frames per chunk
    frames_per_chunk = total_frames // num_frames

    best_frames = []
    best_focus_values = [0] * num_frames
    current_chunk = 0

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for focus measurement
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # Calculate the Laplacian variance
        focus_value = laplacian_variance(gray)

        # Check if this frame has better focus than the best one in the current chunk
        if focus_value > best_focus_values[current_chunk]:
            best_focus_values[current_chunk] = focus_value
            if len(best_frames) <= current_chunk:
                best_frames.append(frame)
            else:
                best_frames[current_chunk] = frame

        # Move to the next chunk if needed
        if (i + 1) % frames_per_chunk == 0:
            current_chunk += 1
            if current_chunk >= num_frames:
                break

    # Release video capture
    cap.release()

    path_to_save_photos = os.path.join(path_to_save, os.path.splitext(os.path.basename(video_path))[0])
    os.makedirs(path_to_save_photos)

    # Save the best frames
    for i, frame in enumerate(best_frames):
        cv2.imwrite(os.path.join(path_to_save_photos, f'frame_{i+1}.jpg'), frame)

    return path_to_save_photos


def extract_images(num_frames: Optional[int]) -> None:
    if num_frames is None:
        num_frames = 10

    with open(os.path.join(os.getcwd(), 'data_path_config.json'), 'r') as file:
        config = json.load(file)

    config["images"] = os.path.join(config["processed_data"], "images")
    if os.path.exists(config["images"]):
        shutil.rmtree(config["images"])
    os.makedirs(config["images"], exist_ok=True)

    config['datasets_images'] = os.path.join(config["datasets"], "images")
    os.makedirs(config["datasets_images"], exist_ok=True)

    with open(os.path.join(os.getcwd(), 'data_path_config.json'), 'w') as file:
        json.dump(config, file, indent=2)

    data_split_types = [os.path.basename(el).split('.')[0] for el in
                        glob.glob(os.path.join(config["datasets_videos"], "*.csv"))]

    for split_type in data_split_types:
        path_to_save = os.path.join(config["images"], split_type)
        os.makedirs(path_to_save)

        df = pd.read_csv(os.path.join(config["datasets_videos"], f"{split_type}.csv"))

        tqdm.pandas(desc=f"Extracting images from videos from {split_type} dataset")
        df['file_path'] = df['file_path'].progress_apply(
            select_best_frames,
            path_to_save=path_to_save,
            num_frames=num_frames
        )

        df.to_csv(
            os.path.join(config["datasets_images"], f"{split_type}.csv"),
            index=False
        )


if __name__ == '__main__':
    # Read input parameters
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-num_frames', '--num_frames', type=int,
                        required=False, default=None,
                        help='Amount of frames to be extracted from video')

    args = parser.parse_args()

    extract_images(num_frames=args.num_frames)
