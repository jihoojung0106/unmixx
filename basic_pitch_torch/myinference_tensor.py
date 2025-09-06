#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2022 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import enum
import json
import os
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from basic_pitch_torch.model import BasicPitchTorch
import numpy as np
import librosa
import pretty_midi
import matplotlib.pyplot as plt
from basic_pitch_torch.constants import (
    AUDIO_SAMPLE_RATE,
    AUDIO_N_SAMPLES,
    ANNOTATIONS_FPS,
    FFT_HOP,
)
from basic_pitch_torch import note_creation as infer

def visualize_salience_map(contour,path):
    
    plt.figure(figsize=(12, 6))
    plt.imshow(contour.T, origin='lower', aspect='auto', cmap='magma')
    plt.colorbar(label='Salience')
    plt.xlabel('Frame')
    plt.ylabel('Frequency Bin')
    plt.title('Pitch Contour (Salience Map)')
    plt.savefig(path)
    print(f"Saved salience map to {path}")
    
def frame_with_pad_torch(x: torch.Tensor, frame_length: int, hop_size: int) -> torch.Tensor:
    """
    Frame a 1D tensor signal with padding at the end (torch version).
    Args:
        x: (T,)
    Returns:
        framed_audio: (n_windows, frame_length)
    """
    n_frames = (x.shape[-1] - frame_length + hop_size) // hop_size
    needed_len = n_frames * hop_size + frame_length
    pad_len = needed_len - x.shape[-1]
    
    if pad_len > 0:
        x = torch.nn.functional.pad(x, (0, pad_len))
    
    framed_audio = x.unfold(0, frame_length, hop_size)  # shape: (n_windows, frame_length)
    return framed_audio


def window_audio_file_torch(audio_original_with_batch: torch.Tensor, hop_size: int) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    """
    Args:
        audio_original_with_batch: (batch, time)
    Returns:
        audio_windowed: (batch, n_windows, frame_length)
    """
    audio_windowed_list = []
    window_times_list = []
    for i in range(audio_original_with_batch.shape[0]):
        audio_original = audio_original_with_batch[i]
        audio_windowed = frame_with_pad_torch(audio_original, AUDIO_N_SAMPLES, hop_size)
        
        window_times = [
            {
                "start": t_start,
                "end": t_start + (AUDIO_N_SAMPLES / AUDIO_SAMPLE_RATE),
            }
            for t_start in torch.arange(audio_windowed.shape[0]) * hop_size / AUDIO_SAMPLE_RATE
        ]
        audio_windowed_list.append(audio_windowed)
        window_times_list.append(window_times)
    
    audio_windowed = torch.stack(audio_windowed_list, dim=0)  # (batch, n_windows, frame_length)
    return audio_windowed, window_times_list

def get_audio_input_torch(audio_original: torch.Tensor, overlap_len: int, hop_size: int) -> Tuple[torch.Tensor, List[Dict[str, float]], int]:
    """
    Args:
        audio_original: (batch, time)
    Returns:
        audio_windowed: (batch, n_windows, frame_length)
    """
    assert overlap_len % 2 == 0, f"overlap_len must be even, got {overlap_len}"

    original_length = audio_original.shape[-1]
    padding = torch.zeros(audio_original.shape[0], overlap_len // 2, device=audio_original.device)
    audio_original = torch.cat([padding, audio_original], dim=1)  # (batch, time + pad)
    
    audio_windowed, window_times_list = window_audio_file_torch(audio_original, hop_size)
    
    return audio_windowed, window_times_list, original_length

def unwrap_output_torch(output: torch.Tensor, audio_original_length: int, n_overlapping_frames: int, batch_size: int) -> torch.Tensor:
    """
    Args:
        output: (n_batches, n_times_short, n_freqs)
    Returns:
        (batch_size, n_times, n_freqs)
    """
    if output.ndim != 3:
        return None

    n_olap = n_overlapping_frames // 2
    if n_olap > 0:
        output = output[:, n_olap:-n_olap, :]
    
    output_shape = output.shape
    per = output_shape[0] // batch_size
    n_output_frames_original = int((audio_original_length * ANNOTATIONS_FPS) / AUDIO_SAMPLE_RATE)
    
    output_list = []
    for batch_idx in range(batch_size):
        raw_output = output[batch_idx * per:(batch_idx + 1) * per] #(2,142,264)
        unwrapped_output = raw_output.reshape(-1, output_shape[2])
        output_list.append(unwrapped_output[:n_output_frames_original])
    
    return torch.stack(output_list, dim=0)  # (batch_size, n_times, n_freqs)

def run_inference(audio_tensor: torch.Tensor, model: nn.Module) -> torch.Tensor:
    n_overlapping_frames = 30
    overlap_len = n_overlapping_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    audio_windowed, _, audio_original_length = get_audio_input_torch(audio_tensor, overlap_len, hop_size)
    batch_size = audio_windowed.shape[0]

    audio_windowed = audio_windowed.view(-1, AUDIO_N_SAMPLES).to(audio_tensor.device)  # flatten batches
    output = model(audio_windowed)["contour"]  # (n_batches, n_times_short, n_freqs)

    unwrapped_output = unwrap_output_torch(output, audio_original_length, n_overlapping_frames, batch_size=batch_size)
    return unwrapped_output


def mypredict(
    audio_tensor,
    model_path: Union[pathlib.Path, str] = "assets/basic_pitch_pytorch_icassp_2022.pth",
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 127.70,
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
    multiple_pitch_bends: bool = False,
    melodia_trick: bool = True,
    debug_file: Optional[pathlib.Path] = None,
    midi_tempo: float = 120,
) -> Tuple[Dict[str, np.array], pretty_midi.PrettyMIDI, List[Tuple[float, float, int, float, Optional[List[int]]]],]:
    """Run a single prediction.

    Args:
        audio_tensor: (batch,time).
        model_or_model_path: Path to load the Keras saved model from. Can be local or on GCS.
        onset_threshold: Minimum energy required for an onset to be considered present.
        frame_threshold: Minimum energy requirement for a frame to be considered present.
        minimum_note_length: The minimum allowed note length in milliseconds.
        minimum_freq: Minimum allowed output frequency, in Hz. If None, all frequencies are used.
        maximum_freq: Maximum allowed output frequency, in Hz. If None, all frequencies are used.
        multiple_pitch_bends: If True, allow overlapping notes in midi file to have pitch bends.
        melodia_trick: Use the melodia post-processing step.
        debug_file: An optional path to output debug data to. Useful for testing/verification.
    Returns:
        The model output, midi data and note events from a single prediction
    """
    model = BasicPitchTorch()
    model.load_state_dict(torch.load(str(model_path)))
    model.eval() 
    if torch.cuda.is_available():
        model.cuda()

    #print(f"Predicting MIDI for {audio_path}...")

    model_output = run_inference(audio_tensor, model, debug_file) #(2,249, 264),더 길면 (2,798,264)
    visualize_salience_map(model_output[0].detach().cpu().numpy(),path="salience1.png")
    visualize_salience_map(model_output[1].detach().cpu().numpy(),path="salience2.png")
    return model_output,None,None