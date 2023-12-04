import os
import json
import argparse

import cv2
import glob
import wave
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from tqdm import tqdm
from natsort import natsorted
from typing import Optional, List, Tuple
from sklearn.model_selection import train_test_split

from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import sobel


def log_spectrogram(audio, sample_rate, window_size=20,
                    step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))

    freqs, _, spec = signal.spectrogram(
        audio,
        fs=sample_rate,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False
    )

    return freqs, np.log(spec.T.astype(np.float32) + eps)


def audio2spectrogram(filepath: str) -> np.ndarray:
    samplerate, test_sound = wavfile.read(filepath, mmap=True)

    _, spectrogram = log_spectrogram(test_sound, samplerate)

    return spectrogram


def audio2wave(filepath) -> np.ndarray:
    _, test_sound = wavfile.read(filepath, mmap=True)

    return test_sound


def gradients_channels(grayscale_array: np.ndarray) -> np.ndarray:
    gradient_x = sobel(grayscale_array, axis=0)
    gradient_y = sobel(grayscale_array, axis=1)

    return np.stack([grayscale_array, gradient_x, gradient_y], axis=-1)


def get_3d_spectrogram(Sxx_in, moments=None):
    if moments is not None:
        (base_mean, base_std, delta_mean, delta_std,
             delta2_mean, delta2_std) = moments
    else:
        base_mean, delta_mean, delta2_mean = (0, 0, 0)
        base_std, delta_std, delta2_std = (1, 1, 1)

    h, w = Sxx_in.shape

    right1 = np.concatenate([Sxx_in[:, 0].reshape((h, -1)), Sxx_in], axis=1)[:, :-1]
    delta = (Sxx_in - right1)[:, 1:]
    delta_pad = delta[:, 0].reshape((h, -1))
    delta = np.concatenate([delta_pad, delta], axis=1)

    right2 = np.concatenate([delta[:, 0].reshape((h, -1)), delta], axis=1)[:, :-1]
    delta2 = (delta - right2)[:, 1:]
    delta2_pad = delta2[:, 0].reshape((h, -1))
    delta2 = np.concatenate([delta2_pad, delta2], axis=1)

    base = (Sxx_in - base_mean) / base_std
    delta = (delta - delta_mean) / delta_std
    delta2 = (delta2 - delta2_mean) / delta2_std

    stacked = [arr.reshape((h, w, 1)) for arr in (base, delta, delta2)]

    return np.concatenate(stacked, axis=2)


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


def pad_image_top(img: np.ndarray, target_height: int = 224) -> np.ndarray:
    """
    Pad the image vertically from the top.

    Args:
        img: Input image (NumPy array).

    Returns:
        Padded image (NumPy array).
    """
    height_padding = target_height - img.shape[1]

    return np.pad(img, ((0, 0), (height_padding, 0), (0, 0)), mode='constant')


def pad_image_both(img: np.ndarray, target_height: int = 224) -> np.ndarray:
    """
    Pad the image vertically.

    Args:
        img: Input image (NumPy array).

    Returns:
        Padded image (NumPy array).
    """
    height_padding = target_height - img.shape[1]

    pad_top = height_padding // 2
    pad_bottom = height_padding - pad_top

    return np.pad(img, ((0, 0), (pad_top, pad_bottom), (0, 0)), mode='constant')


def sliding_window_overlap(image, window_size=224, overlap=168):
    """
    Extract patches from the image using a sliding window with overlap, ensuring even the last patch is of size 224.
    """

    patches = []

    stride = window_size - overlap
    for i in range(0, image.shape[0], stride):
        # If we're at the last window, and it's going to be smaller than 224
        if i + window_size > image.shape[0]:
            patches.append(image[-window_size:, :])
            break
        patches.append(image[i:i + window_size, :])

    return patches


def extract_central_parts(image, window_size=224, overlap=168):
    """
    Extract the central part(s) of the image.

    Args:
        image (np.ndarray): The input image.
        window_size (int): The size of the window to extract.
        overlap (int): Overlap size for extracting two windows. This is only used if two windows are extracted.

    Returns:
        list: A list containing one or two windows, depending on the image size.
    """
    # Compute the center of the image
    center = image.shape[0] // 2

    # Single central part
    start_single = center - (window_size // 2)
    end_single = start_single + window_size
    single_central_part = image[start_single:end_single, :]

    # Two overlapping central parts
    stride = window_size - overlap
    start_double1 = center - (window_size + stride // 2)
    end_double1 = start_double1 + window_size

    start_double2 = center - (stride // 2)
    end_double2 = start_double2 + window_size

    double_central_part1 = image[start_double1:end_double1, :]
    double_central_part2 = image[start_double2:end_double2, :]

    # Decide whether to return one or two parts based on the image size
    if center > window_size + stride // 2:
        return [double_central_part1, double_central_part2]
    else:
        return [single_central_part]


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


def prepare_audio_data(
        chan_gen_type: str, overlap_size: int,
        is_padding: bool, padding_type: str,
        major_emotion: bool
) -> None:
    with open(os.path.join(os.getcwd(), 'data_path_config.json'), 'r') as file:
        config = json.load(file)

    path_to_save_datasets = os.path.join(os.getcwd(), 'datasets')
    os.makedirs(path_to_save_datasets, exist_ok=True)

    if chan_gen_type == "sobel":
        specific_path = os.path.join(path_to_save_datasets,
                                     f"shuffle_3d_by_sobel_sliding_window")
    elif chan_gen_type == "3d_spectrogram":
        specific_path = os.path.join(path_to_save_datasets,
                                     f"shuffle_3d_spectrogram_sliding_window")
    else:
        raise NotImplementedError(f"There is no option as {chan_gen_type} for channels creation")

    if major_emotion:
        specific_path += "_major_emotion"
    if is_padding:
        specific_path += f"_{padding_type}_pad"

    os.makedirs(specific_path, exist_ok=True)

    data_split_types = [os.path.basename(el).split('.')[0] for el in
                        glob.glob(os.path.join(config["datasets_videos"], "*.csv"))]

    for split_type in data_split_types:
        dataset_name = f"{split_type}_dataset"

        df = pd.read_csv(os.path.join(config["datasets_audio_216k"], f"{split_type}.csv"))
        image_folder_path_list = df["file_path"].to_list()

        dataset_data = []
        for audio_idx, audio_file in enumerate(tqdm(image_folder_path_list, desc=f"Processing {split_type} data")):
            _label = np.array(df.iloc[audio_idx, 1:-1].values, dtype=int)
            if np.max(_label) != 0:
                if major_emotion:
                    _label = select_major_emotion(emotion_vector=_label)

                _audio_spectrogram = audio2spectrogram(filepath=audio_file)

                if chan_gen_type == "sobel":
                    _audio_spectrogram_3d = gradients_channels(_audio_spectrogram)
                elif chan_gen_type == "3d_spectrogram":
                    _audio_spectrogram_3d = get_3d_spectrogram(_audio_spectrogram)
                else:
                    raise NotImplementedError(f"There is no option as {chan_gen_type} for channels creation")

                if is_padding:
                    if padding_type == "both":
                        _audio_spectrogram_3d = pad_image_both(_audio_spectrogram_3d)
                    elif padding_type == "top":
                        _audio_spectrogram_3d = pad_image_top(_audio_spectrogram_3d)
                    else:
                        raise NotImplementedError(f"There is no such padding as {padding_type}")

                if split_type == "train":
                    _audio_spectrogram_patches = sliding_window_overlap(
                        _audio_spectrogram_3d,
                        overlap=overlap_size
                    )
                else:
                    _audio_spectrogram_patches = extract_central_parts(
                        _audio_spectrogram_3d,
                        overlap=overlap_size
                    )

                for _patch in _audio_spectrogram_patches:
                    dataset_data.append(
                        [torch.from_numpy(_patch).permute(2, 1, 0), _label, audio_file]
                    )

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
    parser.add_argument('-chan_gen_type', '--chan_gen_type', type=str,
                        required=False, default="sobel",  # 3d_spectrogram | sobel
                        help="Percentage of image's width to crop from sides")
    parser.add_argument('-overlap_size', '--overlap_size', type=int,
                        required=False, default=168,
                        help="Percentage of image's width to crop from sides")
    parser.add_argument('-is_padding', '--is_padding', type=str,
                        required=False, default=True,
                        help="Type of padding")
    parser.add_argument('-padding_type', '--padding_type', type=str,
                        required=False, default="both",
                        help="Type of padding")
    parser.add_argument('-major_emotion', '--major_emotion', type=bool,
                        required=False, default=True,
                        help="Whether to keep only major emotion")

    args = parser.parse_args()

    prepare_audio_data(
        chan_gen_type=args.chan_gen_type, overlap_size=args.overlap_size,
        is_padding=args.is_padding, padding_type=args.padding_type,
        major_emotion=args.major_emotion
    )
