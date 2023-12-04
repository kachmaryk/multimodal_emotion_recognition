import os
import sys
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
from torch import nn
from facenet_pytorch import MTCNN

from tqdm import tqdm
from natsort import natsorted
from typing import Optional, List, Tuple
from sklearn.model_selection import train_test_split

from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import sobel

# Add the trainig_pipelines/image_models/models directory to the system path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.dirname(os.getcwd())),
            "training_pipelines"
        )
    ),
)
from audio_models.utils.get_model import get_model as get_audio_model
from image_models.utils.get_model import get_model as get_image_model


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


def prepare_audio_encoder(path_to_saved_weights: str):
    audio_encoder = get_audio_model(
        model_type="densenet121",
        num_classes=8,
        is_multilabel=False,
        is_fastai_head=False,
        pretrained=False,
        freeze_all_except_last=False,
        unfreeze_first=False
    )

    # Redefining the classifier without the last two layers
    # Keep only the first layer of the original classifier
    audio_encoder.classifier = nn.Sequential(
        *list(audio_encoder.classifier.children())[:-2]
    )

    state_dict = torch.load(path_to_saved_weights, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    audio_encoder.load_state_dict(state_dict, strict=False)

    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    audio_encoder.to(device)

    # Set model to evaluation mode
    audio_encoder.eval()

    return audio_encoder


def prepare_image_encoder(path_to_saved_weights: str):
    image_encoder = get_image_model(
        model_type="InceptionResnetV1Encoder",
        num_classes=8,
        is_multilabel=False,
        is_fastai_head=False,
        pretrained='vggface2',
        freeze_all_except_last=False,
        unfreeze_first=False
    )

    state_dict = torch.load(path_to_saved_weights, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    image_encoder.load_state_dict(state_dict, strict=False)

    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_encoder.to(device)

    # Set model to evaluation mode
    image_encoder.eval()

    return image_encoder


def process_audio_data(_label, audio_file, chan_gen_type,
                       is_padding, padding_type, split_type, overlap_size) -> List:
    audio_data: List = []
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
        audio_data.append(
            [torch.from_numpy(_patch).permute(2, 1, 0), _label, audio_file]
        )

    return audio_data


def process_image_data(_label, split_type, img_folder, mtcnn) -> List:
    image_data: List = []

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

        image_data.append([torch.from_numpy(img).permute(2, 0, 1), _label, _img_path])

    return image_data


def prepare_multimodal_data(
        crop_percent: str, new_shape: Tuple,
        chan_gen_type: str, overlap_size: int,
        is_padding: bool, padding_type: str,
        major_emotion: bool,
        audio_model_path: str, image_model_path: str
) -> None:
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    )

    audio_encoder = prepare_audio_encoder(audio_model_path)
    image_encoder = prepare_image_encoder(image_model_path)

    # Move the models to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(os.path.join(os.getcwd(), 'data_path_config.json'), 'r') as file:
        config = json.load(file)

    path_to_save_datasets = os.path.join(os.getcwd(), 'datasets')
    os.makedirs(path_to_save_datasets, exist_ok=True)

    if chan_gen_type == "sobel":
        specific_path = os.path.join(path_to_save_datasets,
                                     f"multimodal_3d_by_sobel_sliding_window")
    elif chan_gen_type == "3d_spectrogram":
        specific_path = os.path.join(path_to_save_datasets,
                                     f"multimodal_3d_spectrogram_sliding_window")
    else:
        raise NotImplementedError(f"There is no option as {chan_gen_type} for channels creation")

    if is_padding:
        specific_path += f"_{padding_type}_pad_cropped_face"
    if major_emotion:
        specific_path += "_major_emotion"

    os.makedirs(specific_path, exist_ok=True)

    data_split_types = [os.path.basename(el).split('.')[0] for el in
                        glob.glob(os.path.join(config["datasets_videos"], "*.csv"))]

    for split_type in data_split_types:
        dataset_name = f"{split_type}_dataset"

        df = pd.read_csv(os.path.join(config["datasets_audio_216k"], f"{split_type}.csv"))
        audio_folder_path_list = df["file_path"].to_list()

        df = pd.read_csv(os.path.join(config["datasets_images"], f"{split_type}.csv"))
        image_folder_path_list = df["file_path"].to_list()

        dataset_data = []
        for image_idx, img_folder in enumerate(tqdm(image_folder_path_list, desc=f"Processing {split_type} data")):
            image_idx, audio_file = image_idx, audio_folder_path_list[image_idx]

            _label = np.array(df.iloc[image_idx, 1:-1].values, dtype=int)

            if np.max(_label) != 0:
                if major_emotion:
                    _label = select_major_emotion(emotion_vector=_label)

                processed_audio_data = process_audio_data(
                    _label, audio_file, chan_gen_type, is_padding,
                    padding_type, split_type, overlap_size
                )

                processed_image_data = process_image_data(
                    _label, split_type, img_folder, mtcnn
                )

                for _proc_img_idx, _proc_img in enumerate(processed_image_data):
                    if _proc_img >= len(processed_audio_data):
                        _proc_audio = processed_audio_data[-1]
                    else:
                        _proc_audio = processed_audio_data[_proc_img_idx]

                    _proc_img, _proc_audio = _proc_img.to(device), _proc_audio.to(device)

                    # Forward pass through both models
                    _encoded_audio = audio_encoder(_proc_audio)
                    _encoded_img = image_encoder(_proc_img)

                    # Concatenate the output vectors
                    concatenated_output = torch.cat((_encoded_audio, _encoded_img), dim=1)

                    dataset_data.append(
                        [concatenated_output, _label, audio_file]
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
    parser.add_argument('-crop_percent', '--crop_percent', type=int,
                        required=False, default=10,
                        help="Percentage of image's width to crop from sides")
    parser.add_argument('-new_shape', '--new_shape', type=tuple,
                        required=False, default=(160, 160),
                        help="New image shape")

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

    parser.add_argument('-audio_model_path', '--audio_model_path',
                        type=str, required=True,
                        help="Path to the trained audio model weights")
    parser.add_argument('-image_model_path', '--image_model_path',
                        type=str, required=True,
                        help="Path to the trained image model weights")

    args = parser.parse_args()

    prepare_multimodal_data(
        crop_percent=args.crop_percent, new_shape=args.new_shape,
        chan_gen_type=args.chan_gen_type, overlap_size=args.overlap_size,
        is_padding=args.is_padding, padding_type=args.padding_type,
        major_emotion=args.major_emotion,
        audio_model_path=args.audio_model_path, image_model_path=args.image_model_path
    )
