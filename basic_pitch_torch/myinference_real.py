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
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage.measure import regionprops
import networkx as nx
import csv
import enum
import json
import os
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects
from basic_pitch_torch.algo import *
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
from basic_pitch_torch.algo import *

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
        raw_output = output[batch_idx * per:(batch_idx + 1) * per]
        unwrapped_output = raw_output.reshape(-1, output_shape[2])
        output_list.append(unwrapped_output[:n_output_frames_original])
    
    return torch.stack(output_list, dim=0)  # (batch_size, n_times, n_freqs)
def generate_random_string(length=3): 
    import random
    import string

    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for i in range(length))
def run_inference(audio_tensor: torch.Tensor, model: nn.Module, debug_file: Optional[pathlib.Path] = None) -> torch.Tensor:
    n_overlapping_frames = 30
    overlap_len = n_overlapping_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    audio_windowed, _, audio_original_length = get_audio_input_torch(audio_tensor, overlap_len, hop_size) #(2,6,43844),204800
    batch_size = audio_windowed.shape[0]

    audio_windowed = audio_windowed.view(-1, audio_windowed.shape[-1]).to(audio_tensor.device)  # flatten batches
    output = model(audio_windowed.to("cuda"))["contour"]  # (n_batches, n_times_short, n_freqs)

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
    midi_tempo: float = 120,path=None,name=None,
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
    save_folder=os.path.join("sample3",name)
    os.makedirs(save_folder, exist_ok=True)
    model_output = run_inference(audio_tensor, model, debug_file) #(2,249, 264),더 길면 (2,798,264)
    return model_output


def detect_swapped_timestamps(est1, est2, mix, timestamp_upper_lower, threshold=0.1):
    """
    est1/est2가 반대로 뒤바뀌었다고 판단되는 timestamp만 반환

    Args:
    - est1, est2: np.ndarray (freq, time)
    - mix: np.ndarray (freq, time)
    - timestamp_upper_lower: dict[t] = {'upper': (y_slice, x_slice), 'lower': (y_slice, x_slice)}
    - threshold: dist2가 dist1보다 더 작을 때 margin

    Returns:
    - swapped_timestamps: List[int]
    """
    swapped_timestamps = []

    for t, v in timestamp_upper_lower.items():
        if v['upper'] is None or v['lower'] is None:
            continue

        _, y_upper = v['upper'] #앞이 time,뒤가 freq
        _, y_lower = v['lower']

        # 추정 및 mix에서 upper/lower 영역 추출 (해당 timestamp의 freq만)
        mix_upper = mix[t,y_upper]
        mix_lower = mix[t,y_lower]

        est1_upper = est1[t,y_upper]
        est2_upper = est2[t,y_upper]
        est1_lower = est1[t,y_lower]
        est2_lower = est2[t,y_lower]

        # 두 가지 case 거리 비교
        dist1 = np.linalg.norm(est1_upper - mix_upper) + np.linalg.norm(est2_lower - mix_lower)
        dist2 = np.linalg.norm(est2_upper - mix_upper) + np.linalg.norm(est1_lower - mix_lower)

        if dist2 + threshold < dist1:
            swapped_timestamps.append(t)

    return swapped_timestamps





def refine_salience_map(mixed_saliences,est1_saliences,est2_saliences,save_folder=None):
    output_mixed_saliences=[]
    output_est1_saliences=[]
    output_est2_saliences=[]
    for i in range(mixed_saliences.shape[0]):
        mixed_salience=mixed_saliences[i].detach().cpu().numpy()
        est1_salience=est1_saliences[i].detach().cpu().numpy()
        est2_salience=est2_saliences[i].detach().cpu().numpy()
        
        objects_mixed=fine_continous(mixed_salience)
        objects_mixed=filter_objects_by_height(objects_mixed,min_height=3)
        overlaps_mixed=check_overlap_time_histogram(objects_mixed,min_overlap_frames=10)
        if save_folder is not None:
            visualize_salience_map(mixed_salience, path=os.path.join(save_folder,f"mixed.png"), objects=objects_mixed, overlaps=overlaps_mixed)
        objects_est1=fine_continous(est1_salience)
        objects_est1=filter_objects_by_height(objects_est1,min_height=3)
        overlaps_est1=check_overlap_time_histogram(objects_est1,min_overlap_frames=10)

        objects_est2=fine_continous(est2_salience)
        objects_est2=filter_objects_by_height(objects_est2,min_height=3)
        overlaps_est2=check_overlap_time_histogram(objects_est2,min_overlap_frames=10)
        if len(overlaps_est1)>=1 and len(overlaps_est2)>=1:
            est1_salience, est2_salience =joint_process(est1_salience, est2_salience,objects_est1,objects_est2, overlaps_est1,overlaps_est2)
            if save_folder is not None:
                visualize_salience_map(est1_salience, path=os.path.join(save_folder,f"joint1.png"))
                visualize_salience_map(est2_salience, path=os.path.join(save_folder,f"joint2.png"))
        if len(overlaps_est1)>=1 or len(overlaps_est2)>=1:
            est1_salience, est2_salience=single_process(est1_salience, est2_salience,objects_est1,objects_est2, overlaps_est1,overlaps_est2)
            if save_folder is not None:
                visualize_salience_map(est1_salience, path=os.path.join(save_folder,f"single1.png"))
                visualize_salience_map(est2_salience, path=os.path.join(save_folder,f"single2.png"))
        
        isover_mixed=is_overlap_80_percent(overlaps_mixed,objects_mixed)
        if isover_mixed:
            timestamp_upper_lower=get_upper_lower_bboxes_per_timestamp(objects_mixed, mixed_salience)
            if save_folder is not None:
                visualize_salience_map_with_upper_lower(mixed_salience, path=os.path.join(save_folder,f"upper_lower.png"), objects=objects_mixed, overlaps=overlaps_mixed,timestamp_upper_lower=timestamp_upper_lower)
            est1_salience, est2_salience=apply_upper_lower_to_salience(est1_salience, est2_salience, mixed_salience,timestamp_upper_lower)
            if save_folder is not None:
                visualize_salience_map(est1_salience, path=os.path.join(save_folder,f"up_down1.png"))
                visualize_salience_map(est2_salience, path=os.path.join(save_folder,f"up_down2.png"))
        output_mixed_saliences.append(mixed_salience)
        output_est1_saliences.append(est1_salience)
        output_est2_saliences.append(est2_salience)
    return output_mixed_saliences, output_est1_saliences, output_est2_saliences
def refine_salience_map_amp(mixed_saliences, est1_saliences, est2_saliences, save_folder=None):
    output_mixed_saliences = []
    output_est1_saliences = []
    output_est2_saliences = []

    for i in range(mixed_saliences.shape[0]):
        mixed_salience = mixed_saliences[i].detach().cpu().numpy()
        est1_salience = est1_saliences[i].detach().cpu().numpy()
        est2_salience = est2_saliences[i].detach().cpu().numpy()

        # --- STEP 1. 객체 검출 및 시각화 ---
        objects_mixed = fine_continous(mixed_salience)
        objects_mixed = filter_objects_by_height(objects_mixed, min_height=3)
        overlaps_mixed = check_overlap_time_histogram(objects_mixed, min_overlap_frames=10)

        if save_folder is not None:
            visualize_salience_map(mixed_salience, path=os.path.join(save_folder, f"mixed.png"), objects=objects_mixed, overlaps=overlaps_mixed)

        est1_updated = np.zeros_like(est1_salience)
        est2_updated = np.zeros_like(est2_salience)

        for bbox in objects_mixed:
            if bbox is None or len(bbox) != 2:
                continue

            y_slice, x_slice = bbox
            energy1 = np.sum(est1_salience[y_slice, x_slice])
            energy2 = np.sum(est2_salience[y_slice, x_slice])

            if energy1 >= energy2:
                est1_updated[y_slice, x_slice] = est1_salience[y_slice, x_slice]
                est2_updated[y_slice, x_slice] = 0.0
            else:
                est2_updated[y_slice, x_slice] = est2_salience[y_slice, x_slice]
                est1_updated[y_slice, x_slice] = 0.0

        # isover_mixed=is_overlap_80_percent(overlaps_mixed,objects_mixed, threshold=0.4)
        # if isover_mixed:
        #     timestamp_upper_lower=get_upper_lower_bboxes_per_timestamp(objects_mixed, mixed_salience)
        #     if save_folder is not None:
        #         visualize_salience_map_with_upper_lower(mixed_salience, path=os.path.join(save_folder,f"upper_lower.png"), objects=objects_mixed, overlaps=overlaps_mixed,timestamp_upper_lower=timestamp_upper_lower)
        #     est1_updated, est2_updated=apply_upper_lower_to_salience(est1_updated, est2_updated, mixed_salience,timestamp_upper_lower)
        #     if save_folder is not None:
        #         visualize_salience_map(est1_updated, path=os.path.join(save_folder,f"up_down1.png"))
        #         visualize_salience_map(est2_updated, path=os.path.join(save_folder,f"up_down2.png"))
        if save_folder is not None:
            visualize_salience_map(est1_updated, path=os.path.join(save_folder, f"refined_est1.png"))
            visualize_salience_map(est2_updated, path=os.path.join(save_folder, f"refined_est2.png"))
        
        output_mixed_saliences.append(mixed_salience)
        output_est1_saliences.append(est1_updated)
        output_est2_saliences.append(est2_updated)

    return output_mixed_saliences, output_est1_saliences, output_est2_saliences


def get_mixed_contour(mixed_saliences,est1_saliences, est2_saliences,target1_pitch=None,target2_pitch=None,save_folder=None):
    #output_mixed_saliences = []
    istest=target1_pitch==None
    x_list=[]
    y_list=[]
    for i in range(mixed_saliences.shape[0]):
        x_list_per_batch=[]
        y_list_per_batch=[]
        mixed_salience = mixed_saliences[i].detach().cpu().numpy() #(249,140)
        est1_salience= est1_saliences[i]
        est2_salience= est2_saliences[i]
        if not istest:
            target1_pitch_=target1_pitch[i]
            target2_pitch_=target2_pitch[i]
        # --- STEP 1. 객체 검출 및 시각화 ---
        objects_mixed = fine_continous(mixed_salience) #15개 list
        objects_mixed = filter_objects_by_height(objects_mixed, min_height=3)#15개 list
        #overlaps_mixed = check_overlap_time_histogram(objects_mixed, min_overlap_frames=10)
        for y_slice, x_slice in objects_mixed: #y_slice :time, x_slice : freq
            ridge_mask = torch.zeros_like(est1_salience)
            ridge_mask[y_slice, x_slice] = 1.0
            mix_masked = torch.from_numpy(mixed_salience).to(ridge_mask.device) * ridge_mask #(249,140)
            est1_masked = est1_salience * ridge_mask
            est2_masked = est2_salience * ridge_mask
            x = torch.stack([mix_masked, est1_masked, est2_masked, ridge_mask], dim=0) #(4, 249, 140)
            x_list_per_batch.append(x.permute(0, 2, 1))
            if not istest:
                target1_masked = target1_pitch_ * ridge_mask
                target2_masked = target2_pitch_ * ridge_mask
                t1 = target1_masked.sum().item()
                t2 = target2_masked.sum().item()
                y = 0 if t1 > t2 else 1
                y_list_per_batch.append(y)
        x_list.append(x_list_per_batch) # (4, 249, 264) -> (4, 264, 249)
        y_list.append(y_list_per_batch)
    if istest:
        return x_list,None
    return x_list, y_list

def get_mixed_contour_with_bbox(mixed_saliences,est1_saliences, est2_saliences,target1_pitch=None,target2_pitch=None,save_folder=None):
    #output_mixed_saliences = []
    istest=target1_pitch==None
    x_list=[]
    y_list=[]
    bboxes_list=[]
    
    for i in range(mixed_saliences.shape[0]):
        x_list_per_batch=[]
        y_list_per_batch=[]
        bboxes_list_per_batch=[]
        mixed_salience = mixed_saliences[i].detach().cpu().numpy() #(249,140)
        est1_salience= est1_saliences[i]
        est2_salience= est2_saliences[i]
        if not istest:
            target1_pitch_=target1_pitch[i]
            target2_pitch_=target2_pitch[i]
        # --- STEP 1. 객체 검출 및 시각화 ---
        objects_mixed = fine_continous(mixed_salience) #15개 list
        objects_mixed = filter_objects_by_height(objects_mixed, min_height=3)#15개 list
        #overlaps_mixed = check_overlap_time_histogram(objects_mixed, min_overlap_frames=10)
        for y_slice, x_slice in objects_mixed: #y_slice :time, x_slice : freq
            ridge_mask = torch.zeros_like(est1_salience)
            bboxes_list_per_batch.append(
                torch.tensor([x_slice.start/140, x_slice.stop/140, y_slice.start/249, y_slice.stop/249], dtype=torch.float32).to(ridge_mask.device)
            )
            ridge_mask[y_slice, x_slice] = 1.0
            mix_masked = torch.from_numpy(mixed_salience).to(ridge_mask.device) * ridge_mask #(249,140)
            est1_masked = est1_salience * ridge_mask
            est2_masked = est2_salience * ridge_mask
            x = torch.stack([mix_masked, est1_masked, est2_masked, ridge_mask], dim=0) #(4, 249, 140)
            x_list_per_batch.append(x.permute(0, 2, 1))
            if not istest:
                target1_masked = target1_pitch_ * ridge_mask
                target2_masked = target2_pitch_ * ridge_mask
                t1 = target1_masked.sum().item()
                t2 = target2_masked.sum().item()
                y = 0 if t1 > t2 else 1
                y_list_per_batch.append(y)
        x_list.append(x_list_per_batch) # 
        y_list.append(y_list_per_batch)
        bboxes_list.append(bboxes_list_per_batch)
    if istest:
        return x_list,bboxes_list,None
    return x_list, bboxes_list,y_list