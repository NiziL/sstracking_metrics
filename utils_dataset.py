"""
#
# Software Name : sstracking_metrics
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE" file for more details or https://opensource.org/license/MIT
#
"""

import json

import librosa
import numpy as np
import torch
from einops import rearrange


def read_audio(path: str, FS):
    """
    Loads an audio file from the given path using librosa.

    Args:
        path (str): Path to the audio file.

    Returns:
        numpy.ndarray: Audio waveform with sample rate of 16kHz in mono format.
    """
    wav = librosa.load(path, sr=FS, mono=False)[0]
    return wav


def read_array(path: str):
    """
    Reads a numpy array from a file.

    Args:
        path (str): Path to the numpy array file (.npy).

    Returns:
        numpy.ndarray: The loaded array.
    """
    array = np.load(path)
    return array


def time_to_frame(tensor: torch.Tensor, spectro_params: dict):
    """
    Converts time-domain signal to frame-level representation using librosa stft parameters.
    (https://librosa.org/doc/0.11.0/generated/librosa.stft.html)

    Args:
        tensor (torch.Tensor): Input tensor with shape [time, ...].
        spectro_params (dict): Dictionary containing spectral parameters including:
            - n_fft: FFT size
            - win_length: Window length
            - hop_length: Hop size between frames
            - center: Whether to pad signal on both sides
            - pad_mode: Type of padding to use

    Returns:
        torch.Tensor: Frame-level representation with shape [frames, ...].
    """
    tensor = rearrange(tensor.clone(), "t ... -> ... t")

    # to comply with stft "center" parameter
    # we pad 0 in the begining and end of vad to have the frame t centered at t*hop_size
    if spectro_params["center"]:
        tensor = torch.nn.functional.pad(
            tensor,
            pad=[spectro_params["n_fft"] // 2, spectro_params["n_fft"] // 2],
            mode=spectro_params["pad_mode"],
        )

    tensor_per_frame = tensor.unfold(
        dimension=-1,
        size=spectro_params["win_length"],
        step=spectro_params["hop_length"],
    ).clone()  # [..., frames, frame_size]

    out = torch.nanmean(tensor_per_frame, dim=-1)

    return rearrange(out.clone(), " ... t -> t ...")


def load_json(path: str, path_root: str):
    """
    Loads a JSON file and replaces all instances of "{root}" with the provided root path.

    Args:
        path (str): Path to the JSON file.
        path_root (str): Root path to replace "{root}" placeholders.

    Returns:
        dict: Loaded JSON data with replaced root paths.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return replace_root(data, path_root)


def replace_root(input, path_root: str):
    """
    Recursively replaces all instances of "{root}" with the provided root path.

    Args:
        input: Input (can be dict, list, str, or other type).
        path_root (str): Root path to replace "{root}" placeholders.

    Returns:
        Same type as input obj with all "{root}" instances replaced.
    """
    # replace "{root}" instances with path_root
    if isinstance(input, dict):
        return {k: replace_root(v, path_root) for k, v in input.items()}
    elif isinstance(input, list):
        return [replace_root(item, path_root) for item in input]
    elif isinstance(input, str):
        return input.replace("{root}", path_root)
    else:
        return input
