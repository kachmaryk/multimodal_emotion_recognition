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


# Replace with the directory containing ffmpeg.exe
ffmpeg_dir = "D:\\Installers\\ffmpeg\\bin"
# Add the ffmpeg directory to the PATH
os.environ["PATH"] += os.pathsep + ffmpeg_dir


def _extract_original_audio(video_path: str, path_to_save: str) -> str:
    path_to_output_file = os.path.join(path_to_save, os.path.splitext(os.path.basename(video_path))[0] + ".wav")

    exit_code = os.system("ffmpeg -i " + video_path + " -q:a 0 " + path_to_output_file)
    # print(exit_code)

    return path_to_output_file


def _extract_audio_216k(video_path: str, path_to_save: str) -> str:
    path_to_output_file = os.path.join(path_to_save, os.path.splitext(os.path.basename(video_path))[0] + ".wav")

    exit_code = os.system("ffmpeg -i " + video_path + " -ar 16k -ac 1 " + path_to_output_file)
    # print(exit_code)

    return path_to_output_file


def extract_original_audio() -> None:
    with open(os.path.join(os.getcwd(), 'data_path_config.json'), 'r') as file:
        config = json.load(file)

    config["audio"] = os.path.join(config["processed_data"], "audio")
    if os.path.exists(config["audio"]):
        shutil.rmtree(config["audio"])
    os.makedirs(config["audio"], exist_ok=True)

    config['datasets_audio'] = os.path.join(config["datasets"], "audio")
    os.makedirs(config["datasets_audio"], exist_ok=True)

    with open(os.path.join(os.getcwd(), 'data_path_config.json'), 'w') as file:
        json.dump(config, file, indent=2)

    data_split_types = [os.path.basename(el).split('.')[0] for el in
                        glob.glob(os.path.join(config["datasets_videos"], "*.csv"))]

    for split_type in data_split_types:
        path_to_save = os.path.join(config["audio"], split_type)
        os.makedirs(path_to_save)

        df = pd.read_csv(os.path.join(config["datasets_videos"], f"{split_type}.csv"))

        tqdm.pandas(desc=f"Extracting original from videos from {split_type} dataset")
        df['file_path'] = df['file_path'].progress_apply(
            _extract_original_audio,
            path_to_save=path_to_save
        )

        df.to_csv(
            os.path.join(config["datasets_audio"], f"{split_type}.csv"),
            index=False
        )


def extract_audio_216k() -> None:
    with open(os.path.join(os.getcwd(), 'data_path_config.json'), 'r') as file:
        config = json.load(file)

    config["audio_216k"] = os.path.join(config["processed_data"], "audio_216k")
    if os.path.exists(config["audio_216k"]):
        shutil.rmtree(config["audio_216k"])
    os.makedirs(config["audio_216k"], exist_ok=True)

    config['datasets_audio_216k'] = os.path.join(config["datasets"], "audio_216k")
    os.makedirs(config["datasets_audio_216k"], exist_ok=True)

    with open(os.path.join(os.getcwd(), 'data_path_config.json'), 'w') as file:
        json.dump(config, file, indent=2)

    data_split_types = [os.path.basename(el).split('.')[0] for el in
                        glob.glob(os.path.join(config["datasets_videos"], "*.csv"))]

    for split_type in data_split_types:
        path_to_save = os.path.join(config["audio_216k"], split_type)
        os.makedirs(path_to_save)

        df = pd.read_csv(os.path.join(config["datasets_videos"], f"{split_type}.csv"))

        tqdm.pandas(desc=f"Extracting original from videos from {split_type} dataset")
        df['file_path'] = df['file_path'].progress_apply(
            _extract_audio_216k,
            path_to_save=path_to_save
        )

        df.to_csv(
            os.path.join(config["datasets_audio_216k"], f"{split_type}.csv"),
            index=False
        )


if __name__ == '__main__':
    extract_original_audio()

    extract_audio_216k()
