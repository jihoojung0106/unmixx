import os
import math
import json
import random
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
import pyloudnorm as pyln
import os
import torch
import random
import librosa as audio_lib
import numpy as np
import torchaudio
from look2hear.utils import change_pitch_and_formant_random
from speechbrain.inference.speaker import EncoderClassifier
import torch.nn.functional as F 
import math
#from utils import util_dataset
from pytorch_lightning import LightningDataModule
# from pytorch_lightning.core.mixins import HyperparametersMixin
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from typing import Dict, Iterable, List, Iterator
from rich import print
from pytorch_lightning.utilities import rank_zero_only
import json
from torch.utils.data import Dataset, DataLoader
import glob
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
import pandas as pd
import functools

import torchaudio.functional as AF  # or any alias
from look2hear.utils import (
    load_wav_arbitrary_position_mono,
    load_wav_specific_position_mono,
    db2linear,
    loudness_match_and_norm,
    loudness_normal_match_and_norm,
    loudnorm,
    change_pitch_and_formant_random,
    worker_init_fn,
    change_pitch_and_formant,
)
debug=False

def to_tensor(audio):
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    return audio.float()

def save_audio(audio, path,sample_rate=24000):
    audio = to_tensor(audio)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)  # mono
    elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
        audio = audio.T  # (time, channel) -> (channel, time)
    torchaudio.save(path, audio, sample_rate)
# Singing dataset + LibriSpeech dataset
def load_or_create_wav_list(data_dirs, txt_path):
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            wav_paths = [line.strip() for line in f if line.strip()]
    else:
        wav_paths = []
        for path in data_dirs:
            wav_paths.extend(glob.glob(f"{path}/*.wav"))
        with open(txt_path, "w") as f:
            for wav in wav_paths:
                f.write(wav + "\n")
    return wav_paths

class DuetSingingSpeechMixTraining(Dataset):
    """Dataset class for duet singing voice separation tasks.

    Args:
        singing_data_dir (List): The paths of the directories of singing data.
        speech_data_dir (List) : The paths of the directories of speech data.
        song_length_dict_path (str) : The path that contains the length information of singing train data.
        same_song_dict_path (List of list) : The list of lists. Each list is made of
            [
                'root path of singing data',
                'path of json file that contains the dictionary of same song's list'
                'dataset name'
            ]
        same_singer_dict_path (List of list) : The list of lists. Each list is made of
            [
                'root path of singing data',
                'path of json file that contains the dictionary of same singer's songs'
                'dataset name'
            ]
        same_speaker_dict_path (List of list) : The list of lists. Each list is made of
            [
                'root path of speech data',
                'path of json file that contains the dictionary of same speaker's speeches',
                'dataset name
            ]
        sample_rate (int) : The sample rate of the sources and mixtures.
        n_src (int) : The number of sources in the mixture.
        segment (float) : The desired sources and mixtures length in [second].
        unison_prob (float) : Probability of applying unison data augmentation. 0 <= unison_prob <=1
        pitch_formant_augment_prob (float) : Probability of applying pitch and formant shift augmentation. 0 <= prob <=1
        augment (bool) : If true, the volume of two input sources are roughly matched and the loudness of mixture is normalized to -24 LUFS
        part_of_data (float) : Use reduced amount of training data. If part_of_data == 0.1, only 10% of training data will be used.
        sing_sing_ratio (float) : Case 1. Ratio of 'different singing + singing' in training data sampling process.
        sing_speech_ratio (float) : Case 2. Ratio of 'different singing + speech' in training data sampling process.
        same_song_ratio (float) : Case 3. Ratio of 'same song of different singers’ in training data sampling process.
        same_singer_ratio (float) : Case 4. Ratio of 'different songs of same singer’ in training data sampling process.
        same_speaker_ratio (float) : Case 5. Ratio of 'different speeches of same speaker’ in training data sampling process.
        speech_speech_ratio (not specified) : Case 6. Ratio of 'different speech + speech’ in training data sampling process. 
                                            This is not specified by arguments, but automatically calculated by ‘1 - (sum_of_rest_arguments)’.
    Notes:
        sum_of_ratios = (sing_sing_ratio
            + sing_speech_ratio
            + same_song_ratio
            + same_singer_ratio
            + same_speaker_ratio)
        should be smaller than 1
        speech_speech_ratio (different speech + speech, Case 6) will be automatically calculated as 1 - sum_of_ratios
    """

    dataset_name = "singing_with_speech"

    def __init__(
        self,
        singing_data_dir=None,
        speech_data_dir=None,
        song_length_dict_path=None,
        same_song_dict_path=None,
        same_singer_dict_path=None,
        same_speaker_dict_path=None,
        sample_rate=24000,
        n_src=2,
        segment=4,
        unison_prob=0.1,
        pitch_formant_augment_prob=0.2,
        augment=True,
        part_of_data=None,
        reduced_training_data_ratio=0.03,
        sing_sing_ratio=0.2, #0.2
        sing_speech_ratio=0.2, #0.2
        same_song_ratio=0.15,
        same_singer_ratio=0.15,
        same_speaker_ratio=0.15, #0.15
        # speech_speech_ratio=0.15
    ):
        singing_data_dir = [
                "duet_svs/24k/CSD",
                "duet_svs/24k/NUS",
                "duet_svs/24k/VocalSet",
                "duet_svs/24k/jsut-song_ver1",
                "duet_svs/24k/jvs_music_ver1",
                "duet_svs/24k/musdb_a_train",
                "duet_svs/24k/OpenSinger",
                "duet_svs/24k/k_multisinger",
                "duet_svs/24k/k_multitimbre",
            ]
        speech_data_dir = [
                "duet_svs/24k/LibriSpeech_train-clean-360",
                "duet_svs/24k/LibriSpeech_train-clean-100",
            ]
        song_length_dict_path = "duet_svs/24k/json/song_length_dict_24k_mine.json"
        same_song_dict_path = [
                [
                    "duet_svs/24k/k_multisinger",
                    "duet_svs/24k/json/same_song/same_song_k_multisinger_filtered.json",
                    "k_multisinger",
                ],
                [
                    'duet_svs/24k/OpenSinger',
                    'duet_svs/24k/json/same_song/same_song_dict_OpenSinger_.json',
                    "OpenSinger",
                ]
                
            ]
        same_singer_dict_path = [
                [
                    "duet_svs/24k/OpenSinger",
                    "duet_svs/24k/json/same_singer/same_singer_OpenSinger_filtered.json",
                    "OpenSinger",
                ],
                [
                    "duet_svs/24k/k_multisinger",
                    "duet_svs/24k/json/same_singer/same_singer_k_multisinger_filtered.json",
                    "k_multisinger",
                ],
                [
                    "duet_svs/24k/CSD",
                    "duet_svs/24k/json/same_singer/same_singer_CSD_filtered.json",
                    "CSD",
                ],
                [
                    "duet_svs/24k/jsut-song_ver1",
                    "duet_svs/24k/json/same_singer/same_singer_jsut-song_ver1_filtered.json",
                    "jsut-song_ver1",
                ],
                [
                    "duet_svs/24k/jvs_music_ver1",
                    "duet_svs/24k/json/same_singer/same_singer_jvs_music_ver1_filtered.json",
                    "jvs_music_ver1",
                ],
                [
                    "duet_svs/24k/k_multitimbre",
                    "duet_svs/24k/json/same_singer/same_singer_k_multitimbre_filtered.json",
                    "k_multitimbre",
                ],
                [
                    "duet_svs/24k/musdb_a_train",
                    "duet_svs/24k/json/same_singer/same_singer_musdb_a_train_filtered.json",
                    "musdb_a_train",
                ],
                [
                    "duet_svs/24k/NUS",
                    "duet_svs/24k/json/same_singer/same_singer_NUS_filtered.json",
                    "NUS",
                ],
                [
                    "duet_svs/24k/VocalSet",
                    "duet_svs/24k/json/same_singer/same_singer_VocalSet_filtered.json",
                    "VocalSet",
                ],
            ]
        same_speaker_dict_path=[
            [
                "duet_svs/24k/LibriSpeech_train-clean-100",
                "duet_svs/24k/json/same_speaker/same_singer_LibriSpeech_train-clean-100.json",
                "LibriSpeech_train-clean-100",
            ],
            [
                "duet_svs/24k/LibriSpeech_train-clean-360",
                "duet_svs/24k/json/same_speaker/same_singer_LibriSpeech_train-clean-360.json",
                "LibriSpeech_train-clean-360",
            ],
        ]
        self.segment = segment  # segment is length of input segment
        self.sample_rate = sample_rate
        self.n_src = n_src
        self.augment = augment
        self.unison_prob = unison_prob
        self.pitch_formant_augment_prob = pitch_formant_augment_prob
        self.meter = pyln.Meter(self.sample_rate)
        self.reduced_training_data_ratio = reduced_training_data_ratio

        # load singing_data_list from the list of singing data dirs
        singing_txt = "duet_svs/24k/json/singing_wavs.txt"
        speech_txt = "duet_svs/24k/json/speech_wavs.txt"

        self.singing_wav_paths = load_or_create_wav_list(singing_data_dir, singing_txt)
        self.speech_wav_paths = load_or_create_wav_list(speech_data_dir, speech_txt)

        # to check the influence of the training data size, reduce the number of training data.
        if part_of_data != None:
            print("before number of singing data  :", len(self.singing_wav_paths))
            self.singing_wav_paths = random.sample(
                self.singing_wav_paths, int(len(self.singing_wav_paths) * part_of_data)
            )
            print("after number of singing data :", len(self.singing_wav_paths))

            print("before number of speech data  :", len(self.speech_wav_paths))
            self.speech_wav_paths = random.sample(
                self.speech_wav_paths, int(len(self.speech_wav_paths) * part_of_data)
            )
            print("after number of speech data :", len(self.speech_wav_paths))

        song_name_path_dict = {}
        for data_path in self.singing_wav_paths:
            song_name_path_dict[os.path.basename(data_path)] = data_path

        # We have to load a long song more than a short song.
        # Therefore, we will make a singing_train_list, which contains training data paths,
        # a long song more often than a short song.
        with open(song_length_dict_path, "r") as json_file:
            song_length_dict = json.load(json_file)
        # sort dict by descending order
        song_length_dict = dict(
            sorted(song_length_dict.items(), key=lambda x: x[1], reverse=True)
        )
        song_names = []
        song_lengths = []
        for key, value in song_length_dict.items():
            if key not in song_name_path_dict:
                pass
            else:
                song_names.append(key)
                song_lengths.append(value)

        # Determine how many times to load one audio file during one epoch
        train_list_number = np.array(song_lengths) / (self.segment * self.sample_rate)
        self.singing_train_list = []
        for i, num_seg in enumerate(list(train_list_number)):
            try:
                self.singing_train_list.extend(
                    [song_name_path_dict[song_names[i]]] * math.ceil(num_seg)
                )
            except KeyError:  # some songs might not be in the self.song_name_path_dict
                pass

        self.len_singing_train_list = len(self.singing_train_list)

        self.sing_sing_ratio_cum = sing_sing_ratio
        self.sing_speech_ratio_cum = self.sing_sing_ratio_cum + sing_speech_ratio
        self.same_song_ratio_cum = self.sing_speech_ratio_cum + same_song_ratio
        self.same_singer_ratio_cum = self.same_song_ratio_cum + same_singer_ratio
        self.same_speaker_ratio_cum = self.same_singer_ratio_cum + same_speaker_ratio

        self.same_song_dict = {}  # {'songname':[...], ...}
        self.same_song_dataname_path_dict = (
            {}
        )  # {'OpenSinger':OpenSinger root path, ...}
        self.same_song_list = (
            []
        )  # [{'filename':filename,'dataset':dataname,'songname':songname}, ...]
        if same_song_dict_path != None:
            # same_song_dict_path [[data_root,data_dict_path, data_name], ...]
            for path in same_song_dict_path:
                with open(path[1], "r") as json_file:
                    same_song_dict_temp = json.load(json_file)
                self.same_song_dict.update(same_song_dict_temp)
                self.same_song_dataname_path_dict[path[2]] = path[0]
                for same_song_key, same_song_value in same_song_dict_temp.items():
                    for same_song_value_item in same_song_value:
                        self.same_song_list.append(
                            {
                                "filename": same_song_value_item,  # in case of 'same song', key 'filename' contains item ['filename', 'audio_length']
                                "dataset": path[2],
                                "songname": same_song_key,
                            }
                        )

        self.same_singer_dict = {}  # {'singername':[...], ...}
        self.same_singer_dataname_path_dict = (
            {}
        )  # {'OpenSinger':OpenSinger root path, ...}
        self.same_singer_list = (
            []
        )  # [{'filename':filename,'dataset':dataname,'songname':songname}, ...]
        if same_singer_dict_path != None:
            # same_singer_dict_path [[data_root,data_dict_path, data_name]]
            for path in same_singer_dict_path:
                with open(path[1], "r") as json_file:
                    same_singer_dict_temp = json.load(json_file)
                self.same_singer_dict.update(same_singer_dict_temp)
                self.same_singer_dataname_path_dict[path[2]] = path[0]
                for same_singer_key, same_singer_value in same_singer_dict_temp.items():
                    for same_singer_value_item in same_singer_value:
                        self.same_singer_list.append(
                            {
                                "filename": same_singer_value_item,
                                "dataset": path[2],
                                "singername": same_singer_key,
                            }
                        )

        self.same_speaker_dict = {}  # {'speakername':[...], ...}
        self.same_speaker_dataname_path_dict = (
            {}
        )  # {'LibriSpeech_train-clean-100':LibriSpeech_train-clean-100 root path, ...}
        self.same_speaker_list = (
            []
        )  # [{'filename':filename,'dataset':dataname,'songname':songname}, ...]
        if same_speaker_dict_path != None:
            # same_speaker_dict_path [[data_root,data_dict_path, data_name]]
            for path in same_speaker_dict_path:
                with open(path[1], "r") as json_file:
                    same_speaker_dict_temp = json.load(json_file)
                self.same_speaker_dict.update(same_speaker_dict_temp)
                self.same_speaker_dataname_path_dict[path[2]] = path[0]
                for (
                    same_speaker_key,
                    same_speaker_value,
                ) in same_speaker_dict_temp.items():
                    for same_speaker_value_item in same_speaker_value:
                        self.same_speaker_list.append(
                            {
                                "filename": same_speaker_value_item,
                                "dataset": path[2],
                                "speakername": same_speaker_key,
                            }
                        )

    def __len__(self):
        debug=False
        if debug:
            return 2
        return int(self.len_singing_train_list * self.reduced_training_data_ratio)

    def return_paths(self, idx, rand_prob):
        # Case 1. different singing + singing
        if rand_prob <= self.sing_sing_ratio_cum:
            data_path_1 = self.singing_train_list[idx]
            data_path_2 = random.choice(self.singing_train_list)
            return {"path_1": data_path_1, "path_2": data_path_2,'category':"sing_sing"}

        # Case 2. singing + speech
        elif self.sing_sing_ratio_cum < rand_prob <= self.sing_speech_ratio_cum:
            data_path_1 = self.singing_train_list[idx]
            data_path_2 = random.choice(self.speech_wav_paths)
            return {"path_1": data_path_1, "path_2": data_path_2,'category':"sing_speech"}

        # Case 3. same song (but with different singer)
        elif self.sing_speech_ratio_cum < rand_prob <= self.same_song_ratio_cum:
            song_dict = random.choice(
                self.same_song_list
            )  # First, randomly choose a song (song_1)
            data_root = self.same_song_dataname_path_dict[
                song_dict["dataset"]
            ]  # Return the name of song_1's dataset
            filename_1 = song_dict[
                "filename"
            ]  # filename_1 => ['filename', 'audio_length']. Return the list that contains the filename and audio_length [sec] of song_1.
            songname_1 = song_dict[
                "songname"
            ]  # Return the song name (not the path or basename of file) of song_1
            data_path_1 = (
                f"{filename_1[0]}"  # Return the data path of song_1
            )
            # if "k_multisinger" in filename_1[0]:
            #     print(f"filename_1: {filename_1}")
            
            same_song_1_list = self.same_song_dict[
                songname_1
            ].copy()  # Copy the list of song_1's other speeches.
            # For same songs, we need to store the audio lengths because we are going to sample audio segments from the exact same positions
            audio_length_1 = filename_1[1]  # Return the audio length [sec] of song_1
            same_song_1_list.remove(
                filename_1
            )  # Before randomly choose song_2, remove song_1 from the song_1's other speeches.
            filename_2 = random.choice(
                same_song_1_list
            )  # Randomly choose a speech_2 from same_song_1_list
            data_path_2 = f"{filename_2[0]}"  # Return the data path of data_path_2
            audio_length_2 = filename_2[1]  # Return the audio length [sec] of song_2
            same_song_1_list.remove(
                filename_2
            )  # Remove metadata of sampled files for future usage in multi_singing_libri_dataset.py
            return {
                "path_1": data_path_1,
                "path_2": data_path_2,
                "same_list": same_song_1_list,
                "audio_len_1": audio_length_1,
                "audio_len_2": audio_length_2,
                "data_root": data_root,'category':"same_song"
            }

        # Case 4. same singer (but with different song)
        elif self.same_song_ratio_cum < rand_prob <= self.same_singer_ratio_cum:
            singer_dict = random.choice(
                self.same_singer_list
            )  # First, randomly choose a song (song_1)
            data_root = self.same_singer_dataname_path_dict[
                singer_dict["dataset"]
            ]  # Return the name of song_1's dataset
            filename_1 = singer_dict[
                "filename"
            ]  # Return the name of the filename of song_1
            singername_1 = singer_dict[
                "singername"
            ]  # Return the name of the singer name of song_1
            data_path_1 = (
                f"{filename_1}"  # Return the data path of song_1
            )
            same_singer_1_list = self.same_singer_dict[
                singername_1
            ].copy()  # Copy the list of singer_1's other songs.
            same_singer_1_list.remove(
                filename_1
            )  # Before randomly choose another song of singer_1, remove filename_1 from the list first.
            filename_2 = random.choice(same_singer_1_list)  # Randomly choose a song_2
            data_path_2 = (
                f"{filename_2}"  # Return the data path of song_2
            )
            same_singer_1_list.remove(
                filename_2
            )  # Remove metadata of sampled files for future usage in multi_singing_libri_dataset.py
            return {
                "path_1": data_path_1,
                "path_2": data_path_2,
                "same_list": same_singer_1_list,
                "data_root": data_root,'category':"same_singer"
            }

        # Case 5. same speaker (but with different speech)
        elif self.same_singer_ratio_cum < rand_prob <= self.same_speaker_ratio_cum:
            speaker_dict = random.choice(
                self.same_speaker_list
            )  # First, randomly choose a speech (speech_1)
            data_root = self.same_speaker_dataname_path_dict[
                speaker_dict["dataset"]
            ]  # Return the name of speech_1's dataset
            filename_1 = speaker_dict[
                "filename"
            ]  # Return the name of the filename of speech_1
            speakername_1 = speaker_dict[
                "speakername"
            ]  # Return the speaker name of speech_1
            data_path_1 = (
                f"{data_root}/{filename_1}.wav"  # Return the data path of speech_1
            )
            same_speaker_1_list = self.same_speaker_dict[
                speakername_1
            ].copy()  # Copy the list of speaker_1's other speeches.
            same_speaker_1_list.remove(
                filename_1
            )  # Before randomly choose another speech of speaker_1, remove speech_1 from the list first.
            filename_2 = random.choice(
                same_speaker_1_list
            )  # Randomly choose a speech_2
            data_path_2 = (
                f"{data_root}/{filename_2}.wav"  # Return the data path of speech_2
            )
            same_speaker_1_list.remove(
                filename_2
            )  # Remove metadata of sampled files for future usage in multi_singing_libri_dataset.py
            return {
                "path_1": data_path_1,
                "path_2": data_path_2,
                "same_list": same_speaker_1_list,
                "data_root": data_root,'category':"same_speaker"
            }

        # Case 6. different speech + speech
        elif self.same_speaker_ratio_cum < rand_prob <= 1:
            data_path_1 = random.choice(self.speech_wav_paths)
            data_path_2 = random.choice(self.speech_wav_paths)
            return {"path_1": data_path_1, "path_2": data_path_2,   'category':"speech_speech"}

    def load_first_audio(self, rand_prob, paths_info_dict):
        # Case 3, same song (but with different singer)
        if self.sing_speech_ratio_cum < rand_prob <= self.same_song_ratio_cum:
            min_audio_length = min(
                paths_info_dict["audio_len_1"], paths_info_dict["audio_len_2"]
            )  # Choose smaller audio_length of two chosen data
            rand_start = random.uniform(
                0, min_audio_length - self.segment
            )  # Randomly choose the start position of audio
            source_1 = load_wav_specific_position_mono(
                paths_info_dict["path_1"], self.sample_rate, self.segment, rand_start
            )  # if rand_start is smaller than 0, starting position will be converted to zero.
            paths_info_dict["rand_start"] = rand_start
            return source_1, paths_info_dict
        # Other Cases
        else:
            source_1 = load_wav_arbitrary_position_mono(
                paths_info_dict["path_1"], self.sample_rate, self.segment
            )
            return source_1, paths_info_dict

    def load_second_audio(self, source_1, rand_prob, augment_prob, paths_info_dict):
        # unison augmentation
        # In Case 1 (diff sing+sing) or Case 6 (diff speech+speech), apply unison augmentation. data_path_2 will be neglected.
        if (augment_prob <= self.unison_prob) and (
            rand_prob <= self.sing_sing_ratio_cum
            or self.same_speaker_ratio_cum < rand_prob <= 1
        ):
            source_2 = change_pitch_and_formant_random(source_1, self.sample_rate)

        # pitch+formant augmentation only on one source
        elif (
            self.unison_prob
            < augment_prob
            <= self.unison_prob + self.pitch_formant_augment_prob
        ):
            # pitch+formant augment on Case 3, same song (but with different singer)
            if self.sing_speech_ratio_cum < rand_prob <= self.same_song_ratio_cum:
                source_2 = load_wav_specific_position_mono(
                    paths_info_dict["path_2"],
                    self.sample_rate,
                    self.segment,
                    paths_info_dict["rand_start"],
                )
            # pitch+formant augment on other Cases.
            else:
                source_2 = load_wav_arbitrary_position_mono(
                    paths_info_dict["path_2"], self.sample_rate, self.segment
                )
            which_source_prob = random.random()
            if (
                which_source_prob <= 0.333
            ):  # apply pitch+formant augmentation to source_1 or source_2
                source_2 = change_pitch_and_formant_random(source_2, self.sample_rate)
            elif 0.333 < which_source_prob <= 0.666:
                source_1 = change_pitch_and_formant_random(source_1, self.sample_rate)
            else:
                source_1 = change_pitch_and_formant_random(source_1, self.sample_rate)
                source_2 = change_pitch_and_formant_random(source_2, self.sample_rate)
        # No augmentation. Case 3, same song (but with different singer)
        elif self.sing_speech_ratio_cum < rand_prob <= self.same_song_ratio_cum:
            source_2 = load_wav_specific_position_mono(
                paths_info_dict["path_2"],
                self.sample_rate,
                self.segment,
                paths_info_dict["rand_start"],
            )
        # No augmentation. Other Cases.
        else:
            source_2 = load_wav_arbitrary_position_mono(
                paths_info_dict["path_2"], self.sample_rate, self.segment
            )

        return source_1, source_2
    def __getitem__(self, idx):
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                mix, sources,path=self.getitem(idx)
                #rms_mix = torch.sqrt(torch.mean(mix ** 2))
                rms_src1 = torch.sqrt(torch.mean(sources[0] ** 2))
                rms_src2 = torch.sqrt(torch.mean(sources[1] ** 2))
                if rms_src1<0.001 or rms_src2<0.001:
                    continue
                else:
                    return mix, sources, path
            except Exception as e:
                print(f"[Attempt {attempt+1}/{max_attempts}] Error in getitem({idx}): {e}")
        segment_len = int(self.sample_rate * self.segment)
        dummy_mix = torch.zeros((segment_len,), dtype=torch.float32)
        dummy_sources = torch.zeros((2, segment_len), dtype=torch.float32)
        dummy_path = "dummy_path"
        print(f"[Warning] Returning dummy data for idx={idx}")
        return dummy_mix, dummy_sources, dummy_path
    def getitem(self, idx):
        # Load two audio paths first.
        idx = random.randint(0, self.len_singing_train_list - 1)
        rand_prob = random.random()
        
        paths_info_dict = self.return_paths(idx, rand_prob)
        #print(f"paths_info_dict: {paths_info_dict}")
        # Load audio
        sources_list = []

        source_1, paths_info_dict = self.load_first_audio(rand_prob, paths_info_dict)

        augment_prob = random.random()
        source_1, source_2 = self.load_second_audio(
            source_1, rand_prob, augment_prob, paths_info_dict
        )

        # Apply loudness normalization and math between source_1 and source_2
        if self.augment:
            source_1, source_2 = loudness_normal_match_and_norm(
                source_1, source_2, self.meter
            )

        mixture = source_1 + source_2
        mixture, adjusted_gain = loudnorm(
            mixture, -24.0, self.meter
        )  # -24 is target_lufs. -14 is too hot.
        source_1 = source_1 * db2linear(adjusted_gain)
        source_2 = source_2 * db2linear(adjusted_gain)

        sources_list.append(source_1)
        sources_list.append(source_2)

        # Convert to torch tensor
        mixture = torch.as_tensor(mixture, dtype=torch.float32)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.as_tensor(sources, dtype=torch.float32)
        if augment_prob <= self.unison_prob and paths_info_dict["category"] == "sing_sing":
            paths_info_dict["category"] = "unison"
        return mixture, sources, paths_info_dict["category"]


class DuetSingingSpeechMixValidation(Dataset):
    """Dataset class for duet source separation. For validation dataset
    Args:
        data_dir (List) : The list of lists. Each list is made of
            [
                'root path of data for source_1',
                'root path of data for source_2',
                'path of json file that contains metadata of sources',
                'data name (ex. sing_sing_diff)'
            ]
        sample_rate (int) : The sample rate of the sources and mixtures.
        n_src (int) : The number of sources in the mixture.
        segment (int, optional) : The desired sources and mixtures length in s.
        augment (bool) : If true, the volume of two input sources are roughly matched and the loudness of mixture is normalized to -24 LUFS
    """

    dataset_name = "singing_with_speech_valid"

    def __init__(
        self,
        data_dir,
        sample_rate=24000,
        n_src=2,
        segment=4,
        augment=True,
    ):
        self.source_1_paths = []
        self.source_2_paths = []
        self.metadata_list = []
        
        for data_dir_set in data_dir:
            with open(data_dir_set[2], "r") as json_file:
                self.valid_regions_dict = json.load(json_file)
            for key, value in self.valid_regions_dict.items():
                self.source_1_paths.append(f"{data_dir_set[0]}/{key}")
                self.source_2_paths.append(
                    f'{data_dir_set[1]}/{value["corresponding_data"]}'
                )
                self.metadata_list.append(value)

        self.segment = segment
        self.sample_rate = sample_rate
        self.n_src = n_src
        self.augment = augment
        self.meter = pyln.Meter(self.sample_rate)

    def __len__(self):
        debug=False
        if debug:
            return 2
        return len(self.source_1_paths)

    def __getitem__(self, idx):
        data_path_1 = self.source_1_paths[idx]
        data_path_2 = self.source_2_paths[idx]
        metadata = self.metadata_list[idx]

        sources_list = []

        source_1 = load_wav_specific_position_mono(
            data_path_1, self.sample_rate, self.segment, 0.0
        )  # data_1 starts from 0. sec
        source_2 = load_wav_specific_position_mono(
            data_path_2, self.sample_rate, self.segment, metadata["position(sec)"]
        )

        if metadata["unison_aug"]:
            source_2 = change_pitch_and_formant(
                source_2,
                self.sample_rate,
                metadata["unison_params"][0],
                metadata["unison_params"][1],
                1,
                metadata["unison_params"][3],
            )

        if self.augment:
            source_1, source_2 = loudness_match_and_norm(source_1, source_2, self.meter)

        mixture = source_1 + source_2
        mixture, adjusted_gain = loudnorm(
            mixture, -24.0, self.meter
        )  # -24 is target_lufs. -14 is too hot.
        source_1 = source_1 * db2linear(adjusted_gain)
        source_2 = source_2 * db2linear(adjusted_gain)

        sources_list.append(source_1)
        sources_list.append(source_2)

        # Convert to torch tensor
        mixture = torch.as_tensor(mixture, dtype=torch.float32)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.as_tensor(sources, dtype=torch.float32)

        return mixture, sources,"path1"
class MyMedleyVox(torch.utils.data.Dataset):
    """Dataset class for MedleyVox source separation tasks.

    Args:
        task (str): One of ``'unison'``, ``'duet'``, ``'main_vs_rest'`` or
            ``'total'`` :
            * ``'unison'`` for unison vocal separation.
            * ``'duet'`` for duet vocal separation.
            * ``'main_vs_rest'`` for main vs. rest vocal separation (main vs rest).
            * ``'n_singing'`` for N-singing separation. We will use all of the duet, unison, and main vs. rest data.

        sample_rate (int) : The sample rate of the sources and mixtures.
        n_src (int) : The number of sources in the mixture. Actually, this is fixed to 2 for our tasks. Need to be specified for N-singing training (future work).
        segment (int, optional) : The desired sources and mixtures length in s.
    """
    def __init__(
        self,
        root_dir="duet_svs/MedleyVox_24k_chunks",
        metadata_dir=None,
        simple=False,
        task="duet",
        sample_rate=24000,
        n_src=2,
        segment=None,
        return_id=True,
    ):
        self.simple=simple
        self.root_dir = root_dir  # /path/to/data/test_medleyDB
        self.metadata_dir = metadata_dir  # ./testset/testset_config
        self.task = task.lower()
        self.return_id = return_id
        # Get the csv corresponding to the task
        if self.task == "unison":
            self.total_mix_list = sorted(glob.glob(f"{self.root_dir}/unison/*/*/mix/*.wav"))
            self.total_gt_list =sorted( glob.glob(f"{self.root_dir}/unison/*/*/gt/*.wav"))
            assert 2*len(self.total_mix_list) == len(self.total_gt_list)
        elif self.task == "duet":
            self.total_mix_list = sorted(glob.glob(f"{self.root_dir}/duet/*/*/mix/*.wav"))
            self.total_gt_list =sorted( glob.glob(f"{self.root_dir}/duet/*/*/gt/*.wav"))
            assert 2*len(self.total_mix_list) == len(self.total_gt_list)
        elif self.task == "main_vs_rest":
            self.total_segments_list = glob.glob(f"{self.root_dir}/rest/*/*")
        elif self.task == "n_singing":
            self.total_segments_list = glob.glob(f"{self.root_dir}/unison/*/*") + glob.glob(f"{self.root_dir}/duet/*/*") + glob.glob(f"{self.root_dir}/rest/*/*")
        self.segment = segment
        self.sample_rate = sample_rate
        self.n_src = n_src
        
    def get_chunked_embedding(self, input_waveform, chunk_sec=1.5, hop_sec=0.75, sample_rate=16000):
        """
        Efficiently extracts chunk-level speaker embeddings for a batch of audio.

        Args:
            input_waveform (Tensor): (batch, time)
        Returns:
            Tensor: (batch, num_chunks, emb_dim)
        """
        B, T = input_waveform.shape
        chunk_size = int(chunk_sec * sample_rate)
        hop_size = int(hop_sec * sample_rate)

        num_chunks = (T - chunk_size) // hop_size + 1
        if num_chunks <= 0:
            raise ValueError("Input too short to extract even one chunk.")

        # 모든 chunk index를 미리 뽑아내기
        indices = torch.arange(0, num_chunks * hop_size, hop_size, device=input_waveform.device)  # (num_chunks,)
        chunked = torch.stack([
            input_waveform[:, i:i+chunk_size] for i in indices
        ], dim=1)  # shape: (B, num_chunks, chunk_size)

        B, N, L = chunked.shape
        chunked = chunked.reshape(B * N, L)  # (B*N, chunk_size)

        # 전체 chunk에 대해 한 번에 encode (batchify)
        with torch.no_grad():  # inference 모드
            emb, _ = self.classifier.encode_batch(chunked)  # (B*N, 1, D)
        emb = emb.squeeze(1)  # (B*N, D)
        emb = F.normalize(emb, dim=1)

        emb = emb.view(B, N, -1)  # (B, N_chunks, D)
        return emb
    def __len__(self):
        # if debug:
        #     return 5
        return len(self.total_mix_list)

    def __getitem__(self, idx):
        mixture_path = self.total_mix_list[idx]
        song_name=mixture_path.split("/")[-4]
        segment_name=mixture_path.split("/")[-3]
        chunk_name=os.path.basename(mixture_path).split("_")[-1]
        sources_path_list_cand = glob.glob(
            os.path.dirname(mixture_path).replace("mix", "gt")+"/*.wav"
            )
        sources_path_list=[x for x in sources_path_list_cand if song_name in x and segment_name in x and chunk_name in x]
        assert len(sources_path_list) == 2, f"Expected 2 sources, found {len(sources_path_list)} for {mixture_path}"
        sources_list=[]
        ids = []
        for i, source_path in enumerate(sources_path_list):
            s, sr = torchaudio.load(source_path)
            if sr != self.sample_rate:
                s = torchaudio.functional.resample(s, sr, self.sample_rate)
            sources_list.append(s)
            ids.append(os.path.basename(source_path).replace(".wav", ""))
        # Read the mixture
        mixture, sr = torchaudio.load(mixture_path)
        if sr != self.sample_rate:
            mixture = torchaudio.functional.resample(mixture, sr, self.sample_rate)
        waveform1 = sources_list[0][0]  # mono
        waveform2 = sources_list[1][0]  # mono
        mixture= mixture[0]  # mono
        # Convert sources to tensor
        sources = torch.stack([waveform1, waveform2])
        if not self.return_id:
            return mixture, sources
        time=mixture.shape[-1]
        return mixture, sources, ids[0]

class SingingLibriDataModuleNew(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        n_src: int = 2,
        sample_rate: int = 24000,
        segment_seconds: float = 4.0,
        normalize_audio: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        sing_sing_ratio=0.2, #0.2
        sing_speech_ratio=0.2, #0.2
        same_song_ratio=0.15,
        same_singer_ratio=0.15,
        same_speaker_ratio=0.15, #0.15
        unison_prob=0.1,pitch_formant_augment_prob=0.2,
        reduced_training_data_ratio=0.03
    ) -> None:
        super().__init__()
        if train_dir == None or valid_dir == None or test_dir == None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))

        # this line allows to access init params with 'self.hparams' attribute
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.n_src = n_src
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.normalize_audio = normalize_audio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None
        self.sing_sing_ratio = sing_sing_ratio
        self.sing_speech_ratio = sing_speech_ratio
        self.same_song_ratio = same_song_ratio
        self.same_singer_ratio = same_singer_ratio
        self.same_speaker_ratio = same_speaker_ratio
        self.unison_prob=unison_prob
        self.pitch_formant_augment_prob=pitch_formant_augment_prob
        self.reduced_training_data_ratio=reduced_training_data_ratio
    def setup(self,stage=None) -> None:
        self.data_train = DuetSingingSpeechMixTraining(
            sing_sing_ratio=self.sing_sing_ratio, #0.2
            sing_speech_ratio=self.sing_speech_ratio, #0.2
            same_song_ratio=self.same_song_ratio, #0.15
            same_singer_ratio=self.same_singer_ratio, #0.15
            same_speaker_ratio= self.same_speaker_ratio, #0.15
            unison_prob=self.unison_prob, #0.1
            segment=self.segment_seconds,
            pitch_formant_augment_prob=self.pitch_formant_augment_prob, #0.2
            reduced_training_data_ratio=self.reduced_training_data_ratio, #0.03
        )
        self.data_val0= DuetSingingSpeechMixValidation(
            data_dir=[[
                "duet_svs/24k/musdb_a_test",
                "duet_svs/24k/musdb_a_test",
                "duet_svs/24k/json/valid/valid_regions_dict_singing_singing.json",
                "duet",]
            ]
        )
        self.data_val = MyMedleyVox(
            task="duet",  # or "unison" or "main_vs_rest"
        )
        self.data_test = MyMedleyVox(
            task="unison"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,#collate_fn=safe_collate
        )

    def val_dataloader(self) -> DataLoader:
        return [
            DataLoader(
                dataset=self.data_val0,
                shuffle=False,
                batch_size=1,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
                drop_last=False#,collate_fn=safe_collate
            ),
            
            DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False#,collate_fn=safe_collate
        ),DataLoader(
            dataset=self.data_test,
            shuffle=False,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )]

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            shuffle=False,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    @property
    def make_sets(self):
        return self.data_train, self.data_val, self.data_test

