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
    
def frame_with_pad(x: np.array, frame_length: int, hop_size: int) -> np.array:
    """
    Extends librosa.util.frame with end padding if required, similar to 
    tf.signal.frame(pad_end=True).

    Returns:
        framed_audio: tensor with shape (n_windows, AUDIO_N_SAMPLES)
    """
    n_frames = int(np.ceil((x.shape[0] - frame_length) / hop_size)) + 1
    n_pads = (n_frames - 1) * hop_size + frame_length - x.shape[0]
    x = np.pad(x, (0, n_pads), mode="constant")
    framed_audio = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_size)
    return framed_audio


def window_audio_file(audio_original_with_batch: np.array, hop_size: int) -> Tuple[np.array, List[Dict[str, int]]]:
    """
    Pad appropriately an audio file, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

    Returns:
        audio_windowed: tensor with shape (n_windows, AUDIO_N_SAMPLES, 1)
            audio windowed into fixed length chunks
        window_times: list of {'start':.., 'end':...} objects (times in seconds)

    """
    audio_windowed_list = []
    window_times_list = []
    for i in range(audio_original_with_batch.shape[0]):
        audio_original=audio_original_with_batch[i]
        audio_windowed = frame_with_pad(audio_original, AUDIO_N_SAMPLES, hop_size)
        window_times = [
            {
                "start": t_start,
                "end": t_start + (AUDIO_N_SAMPLES / AUDIO_SAMPLE_RATE),
            }
            for t_start in np.arange(audio_windowed.shape[0]) * hop_size / AUDIO_SAMPLE_RATE
        ]
        audio_windowed_list.append(audio_windowed)
        window_times_list.append(window_times)
    return audio_windowed_list, window_times_list


def get_audio_input(
    audio_original, overlap_len: int, hop_size: int
) -> Tuple[Tensor, List[Dict[str, int]], int]:
    """
    Read wave file (as mono), pad appropriately, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

    Returns:
        audio_windowed: tensor with shape (n_windows, AUDIO_N_SAMPLES, 1)
            audio windowed into fixed length chunks
        window_times: list of {'start':.., 'end':...} objects (times in seconds)
        audio_original_length: int
            length of original audio file, in frames, BEFORE padding.

    """
    assert overlap_len % 2 == 0, "overlap_length must be even, got {}".format(overlap_len)


    original_length = audio_original.shape[-1]
    padding = np.zeros((audio_original.shape[0], int(overlap_len / 2)), dtype=np.float32)
    audio_original = np.concatenate([padding, audio_original], axis=1) #(2,67840)
    #audio_original = np.concatenate([np.zeros((int(overlap_len / 2),), dtype=np.float32), audio_original])
    audio_windowed_list, window_times_list = window_audio_file(audio_original, hop_size)
    #audio_windowed = np.stack(audio_windowed_list, axis=0)  # (n_windows, AUDIO_N_SAMPLES, 1)
    return audio_windowed_list, window_times_list, original_length


def unwrap_output(output: Tensor, audio_original_length: int, n_overlapping_frames: int,batch_size:int) -> np.array:
    """Unwrap batched model predictions to a single matrix.

    Args:
        output: array (n_batches, n_times_short, n_freqs)
        audio_original_length: length of original audio signal (in samples)
        n_overlapping_frames: number of overlapping frames in the output

    Returns:
        array (n_times, n_freqs)
    """
    raw_output = output.cpu().detach().numpy()
    if len(raw_output.shape) != 3:
        return None

    n_olap = int(0.5 * n_overlapping_frames)
    if n_olap > 0:
        # remove half of the overlapping frames from beginning and end
        raw_output = raw_output[:, n_olap:-n_olap, :]
        #raw_output : (n_batches*2,times, n_freqs)
    output_shape = raw_output.shape 
    per_=output_shape[0]//batch_size #2
    n_output_frames_original = int(np.floor(audio_original_length * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
    output_list=[]
    for batch_idx in range(batch_size):
        raw_output_=raw_output[batch_idx*per_:(batch_idx+1)*per_,:,:]
        unwrapped_output = raw_output_.reshape(per_ * output_shape[1], output_shape[2])
        output_list.append(unwrapped_output[:n_output_frames_original, :])
    output_list_expanded = [np.expand_dims(x, axis=0) for x in output_list]
    return np.concatenate(output_list_expanded, axis=0)
    #return np.concatenate(output_list, axis=0)  # (n_times, n_freqs)


def run_inference(
    audio_tensor: Tensor,
    model: nn.Module,
    debug_file: Optional[pathlib.Path] = None,
) -> Dict[str, np.array]:
    """Run the model on the input audio path.

    Args:
        audio_tensor: (batch,time).
        model: A loaded keras model to run inference with.
        debug_file: An optional path to output debug data to. Useful for testing/verification.

    Returns:
       A dictionary with the notes, onsets and contours from model inference.
    """
    # overlap 30 frames
    n_overlapping_frames = 30
    overlap_len = n_overlapping_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len
    audio_windowed_list, _, audio_original_length = get_audio_input(audio_tensor, overlap_len, hop_size)
    batch_size=len(audio_windowed_list)
    
    audio_windowed=np.concatenate(audio_windowed_list, axis=-1)
    assert len(audio_windowed.shape) == 2, "audio_windowed should be 2D"
    audio_windowed = torch.from_numpy(audio_windowed).T
    if torch.cuda.is_available():
        audio_windowed = audio_windowed.cuda()

    output = model(audio_windowed)["contour"] #[]torch.Size([3, 172, 264])
    #output["contour"]
    #unwrapped_output = {k: unwrap_output(output[k], audio_original_length, n_overlapping_frames) for k in output}
    unwrapped_output = unwrap_output(output, audio_original_length, n_overlapping_frames,batch_size=batch_size) #(time, freq)
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
    visualize_salience_map(model_output[0],path="salience1.png")
    visualize_salience_map(model_output[1],path="salience2.png")
    return model_output,None,None