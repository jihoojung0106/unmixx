import os
import math
import json
import random
import glob
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
import pyloudnorm as pyln
import librosa
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import torchaudio
from look2hear.utils import (
    load_wav_arbitrary_position_mono,
    load_wav_from_start_mono,
    load_wav_specific_position_mono,
    db2linear,
    loudness_match_and_norm,
    loudness_normal_match_and_norm,
    loudnorm,
    change_pitch_and_formant_random,
    worker_init_fn,
    change_pitch_and_formant,
)
import torchaudio
def generate_random_string(length=4):
    """Generate a random string of fixed length."""
    import random
    import string

    letters = string.ascii_letters  # a-z, A-Z
    return ''.join(random.choice(letters) for i in range(length))


def save_waveform(waveform, path, sample_rate):
    # mono 또는 stereo tensor 지원
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    torchaudio.save(path, waveform.unsqueeze(0), sample_rate)
    print(f"Saved waveform to {path}"   )
# Singing dataset + LibriSpeech dataset
class DuetSingingSpeechMixTraining(Dataset):
    
    def __init__(
        self,
        singing_data_dir: list[str]= None,
        speech_data_dir: list[str] = None,
        song_length_dict_path: str= "duet_svs/16k/json/song_length_dict_16k.json",
        same_song_dict_path: list[list[str]]= None,
        same_singer_dict_path: list[list[str]]= None,
        same_speaker_dict_path: list[list[str]]= None,
        *,
        sample_rate: int = 16000,
        n_src: int = 2,
        segment: float = 3.0,                 # ← argparse --seq_dur
        unison_prob: float = 0.01,             # ← argparse --unison_prob
        pitch_formant_augment_prob: float = 0.01,
        augment: bool = False,
        part_of_data: float= None,
        reduced_training_data_ratio: float = 1.0,
        sing_sing_ratio: float = 0.4,
        sing_speech_ratio: float = 0.0,
        same_song_ratio: float = 0.3,
        same_singer_ratio: float = 0.3,
        same_speaker_ratio: float = 0.0,
        #segment_seconds: float = 4.0,  # segment length in seconds
    ):
        # ---------- 2. None 이면 argparse 기본값으로 교체 ----------
        if singing_data_dir is None:
            singing_data_dir = [
                "duet_svs/16k/CSD",
                "duet_svs/16k/NUS",
                "duet_svs/16k/VocalSet",
                "duet_svs/16k/jsut-song_ver1",
                "duet_svs/16k/jvs_music_ver1",
                "duet_svs/16k/musdb_a_train",
                "duet_svs/16k/OpenSinger",
                "duet_svs/16k/k_multisinger",
                "duet_svs/16k/k_multitimbre",
            ]
        if speech_data_dir is None:
            speech_data_dir = [
                "/path/to/data/24k/LibriSpeech_train-clean-360",
                "/path/to/data/24k/LibriSpeech_train-clean-100",
            ]
        if song_length_dict_path is None:
            song_length_dict_path = "duet_svs/16k/json/song_length_dict_16k.json"

        if same_song_dict_path is None:
            same_song_dict_path = [
                [
                    "duet_svs/16k/k_multisinger",
                    "duet_svs/16k/json/same_song/same_song_k_multisinger_filtered.json",
                    "k_multisinger",
                ],
                
            ]
        if same_singer_dict_path is None:
            same_singer_dict_path = [
                [
                    "duet_svs/16k/OpenSinger",
                    "duet_svs/16k/json/same_singer/same_singer_OpenSinger_filtered.json",
                    "OpenSinger",
                ],
                [
                    "duet_svs/16k/k_multisinger",
                    "duet_svs/16k/json/same_singer/same_singer_k_multisinger_filtered.json",
                    "k_multisinger",
                ],
                [
                    "duet_svs/16k/CSD",
                    "duet_svs/16k/json/same_singer/same_singer_CSD_filtered.json",
                    "CSD",
                ],
                [
                    "duet_svs/16k/jsut-song_ver1",
                    "duet_svs/16k/json/same_singer/same_singer_jsut-song_ver1_filtered.json",
                    "jsut-song_ver1",
                ],
                [
                    "duet_svs/16k/jvs_music_ver1",
                    "duet_svs/16k/json/same_singer/same_singer_jvs_music_ver1_filtered.json",
                    "jvs_music_ver1",
                ],
                [
                    "duet_svs/16k/k_multitimbre",
                    "duet_svs/16k/json/same_singer/same_singer_k_multitimbre_filtered.json",
                    "k_multitimbre",
                ],
                [
                    "duet_svs/16k/musdb_a_train",
                    "duet_svs/16k/json/same_singer/same_singer_musdb_a_train_filtered.json",
                    "musdb_a_train",
                ],
                [
                    "duet_svs/16k/NUS",
                    "duet_svs/16k/json/same_singer/same_singer_NUS_filtered.json",
                    "NUS",
                ],
                [
                    "duet_svs/16k/VocalSet",
                    "duet_svs/16k/json/same_singer/same_singer_VocalSet_filtered.json",
                    "VocalSet",
                ],
            ]
        
        self.segment = segment  # segment is length of input segment
        self.sample_rate=self.sr = sample_rate
        self.segment_len = int(segment * sample_rate)
        self.n_src = n_src
        self.augment = augment
        self.unison_prob = unison_prob
        self.pitch_formant_augment_prob = pitch_formant_augment_prob
        self.meter = pyln.Meter(self.sample_rate)
        self.reduced_training_data_ratio = reduced_training_data_ratio = 0.03

        # load singing_data_list from the list of singing data dirs
        self.singing_wav_paths = []
        for path in singing_data_dir:
            self.singing_wav_paths.extend(glob.glob(f"{path}/*.wav"))

        # to check the influence of the training data size, reduce the number of training data.
        if part_of_data != None:
            print("before number of singing data  :", len(self.singing_wav_paths))
            self.singing_wav_paths = random.sample(
                self.singing_wav_paths, int(len(self.singing_wav_paths) * part_of_data)
            )
            print("after number of singing data :", len(self.singing_wav_paths))

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
        # def is_valid_wav(path):
        #     try:
        #         with sf.SoundFile(path) as f:
        #             return True
        #     except:
        #         return False

        # # 예시: 유효한 파일만 로딩
        # self.singing_wav_paths = [p for p in self.singing_wav_paths if is_valid_wav(p)]
        self.jacapella_list=glob.glob("duet_svs/jaCappella/**/*.wav", recursive=True)
        self.jacapella_list = [x for x in self.jacapella_list if "mixture" not in x and "vocal_percussion" not in x and "finger_snap" not in x]
        self.jacapella_list =[x for x in self.jacapella_list if "_24K.wav" not in x]
        self.choir_list=glob.glob("duet_svs/Choir_Dataset/**/*.wav", recursive=True)
        self.choir_list=[x for x in self.choir_list if "CANTUS" in x or "Alto" in x or "Soprano" in x or "Tenor" in x or "Bass" in x or "Cantus" in x or "Bassus" in x or "Altus" in x or "BASSUS" in x or "ALTUS" in x or "SOPRANO" in x or "TENOR" in x or "ALTO" in x or "BASS" in x] 
        self.choir_list=[x for x in self.choir_list if "_24K.wav" not in x] #16k로 바꿔야함
        self.jpop_list=glob.glob("duet_svs/datasets--imprt--idol-songs-jp/snapshots/c026e76507d574b4f79efb0f01e41fb1b421b563/vocals_24k/**/*.wav", recursive=True)
        self.jpop_list=[x for x in self.jpop_list if "serifu" not in x]
        self.jacapella_dict = defaultdict(list)
        for path in self.jacapella_list:
            self.jacapella_dict[os.path.dirname(path)].append(path)
        self.choir_dict = defaultdict(list)
        for path in self.choir_list:
            self.choir_dict[os.path.dirname(path)].append(path)
        self.jpop_dict = defaultdict(list)
        for path in self.jpop_list:
            dir_name = os.path.dirname(path)
            basename=os.path.basename(path)  # 디렉토리 이름만 추출
            key_parts = basename.split('-')[1]
            key = dir_name+"_"+key_parts#key_parts[1] if len(key_parts) > 1 else dir_name
            self.jpop_dict[key].append(path)
        self.jpop_dict = {k: v for k, v in self.jpop_dict.items() if len(v) >= 2}
        #self.musbd_list=glob.glob("duet_svs/16k/musdb_a_train/*.wav")
        
    def __len__(self):
        return int(self.len_singing_train_list * self.reduced_training_data_ratio)

    def return_paths(self, idx, rand_prob):
        # Case 1. different singing + singing
        if rand_prob <= self.sing_sing_ratio_cum:
            data_path_1 = self.singing_train_list[idx]
            data_path_2 = random.choice(self.singing_train_list)
            return {"path_1": data_path_1, "path_2": data_path_2}

        # Case 2. singing + speech
        elif self.sing_sing_ratio_cum < rand_prob <= self.sing_speech_ratio_cum:
            data_path_1 = self.singing_train_list[idx]
            data_path_2 = random.choice(self.speech_wav_paths)
            return {"path_1": data_path_1, "path_2": data_path_2}

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
                "data_root": data_root,
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
                "data_root": data_root,
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
                f"{filename_1}"  # Return the data path of speech_1
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
                f"{filename_2}"  # Return the data path of speech_2
            )
            same_speaker_1_list.remove(
                filename_2
            )  # Remove metadata of sampled files for future usage in multi_singing_libri_dataset.py
            return {
                "path_1": data_path_1,
                "path_2": data_path_2,
                "same_list": same_speaker_1_list,
                "data_root": data_root,
            }

        # Case 6. different speech + speech
        elif self.same_speaker_ratio_cum < rand_prob <= 1: #이거 harmony로 하자. 
            data_path_1 = random.choice(self.speech_wav_paths)
            data_path_2 = random.choice(self.speech_wav_paths)
            return {"path_1": data_path_1, "path_2": data_path_2}

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
        # Load two audio paths first.
        try:
            rand_prob = random.random()
            if self.same_speaker_ratio_cum < rand_prob <= 1:
                if random.random()<0.5:
                    dir_ = random.choice(list(self.jacapella_dict.keys()))
                    path1, path2 = random.sample(self.jacapella_dict.get(dir_, []), 2)
                else: #choirdataset
                    dir_ = random.choice(list(self.choir_dict.keys()))
                    path1, path2 = random.sample(self.choir_dict.get(dir_, []), 2)
                source_1,source_2=self.load_harmody(path1, path2)
            else:
                idx = random.randint(0, self.len_singing_train_list - 1)
                paths_info_dict = self.return_paths(idx, rand_prob)
                # Load audio
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
            # mixture, adjusted_gain = loudnorm(
            #     mixture, -24.0, self.meter
            # )  # -24 is target_lufs. -14 is too hot.
            # source_1 = source_1 * db2linear(adjusted_gain)
            # source_2 = source_2 * db2linear(adjusted_gain)
            visualize=False
            # if torch.distributed.is_initialized():
            #     rank = torch.distributed.get_rank()
            #     print(f"[Rank {rank}] __getitem__ called: idx={paths_info_dict['path_1']}")
            # else:
            #     print(f"[Single-process] __getitem__ called: idx={paths_info_dict['path_1']}")

            if visualize:
                if rand_prob <= self.sing_sing_ratio_cum:
                    case= "sing_sing"
                elif self.sing_sing_ratio_cum < rand_prob <= self.sing_speech_ratio_cum:
                    case= "sing_speech"
                elif self.sing_speech_ratio_cum < rand_prob <= self.same_song_ratio_cum:
                    case= "same_song"
                elif self.same_song_ratio_cum < rand_prob <= self.same_singer_ratio_cum:
                    case= "same_singer"
                elif self.same_singer_ratio_cum < rand_prob <= self.same_speaker_ratio_cum:
                    case= "same_speaker"
                else:
                    case= "harmony"
                save_folder = f"result/{case}/{generate_random_string(4)}"
                os.makedirs(save_folder, exist_ok=True)
                save_waveform(source_1, f"{save_folder}/source_1.wav", self.sample_rate)
                save_waveform(source_2, f"{save_folder}/source_2.wav", self.sample_rate)
                save_waveform(mixture, f"{save_folder}/mixture.wav", self.sample_rate)
                
            sources_list = []
            sources_list.append(source_1)
            sources_list.append(source_2)

            # Convert to torch tensor
            mixture = torch.as_tensor(mixture, dtype=torch.float32)
            # Stack sources
            sources = np.vstack(sources_list)
            # Convert sources to tensor
            sources = torch.as_tensor(sources, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            dummy_mixture = torch.zeros(self.segment_len)
            dummy_sources = torch.zeros(2, self.segment_len)
            dummy_path1 = "dummy.wav"
            return dummy_mixture, dummy_sources, dummy_path1
        return mixture, sources, "path1"
    def load_json_intervals(self,path):
        with open(path, "r") as f:
            data = json.load(f)
        return data[0]["value"]  # assuming structure is like above
    def find_overlapping_intervals(self,intervals1, intervals2):
        overlaps = []
        for s1, e1 in intervals1:
            for s2, e2 in intervals2:
                if not (e1 < s2 or e2 < s1):  # 겹치는 조건
                    overlaps.append({
                        "interval1": [s1, e1],
                        "interval2": [s2, e2]
                    })
        return overlaps

    def load_harmody(self,path1,path2):
        path1_json = path1.replace(".wav", ".json")
        path2_json = path2.replace(".wav", ".json")
        interval1=self.load_json_intervals(path1_json)
        interval2=self.load_json_intervals(path2_json)
        overlaps = self.find_overlapping_intervals(interval1, interval2)
        if overlaps:
            selected = random.choice(overlaps)
            selected_interval1= selected["interval1"]
            selected_interval2= selected["interval2"]
            duration1=selected_interval1[1]-selected_interval1[0]
            duration2=selected_interval2[1]-selected_interval2[0]
            if duration1<duration2: #더 작은 Interval을 기준으로 선택
                start, end = selected_interval1
            else:
                start, end = selected_interval2
            
        else:
            if len(interval1)!=0:
                start, end = random.choice(interval1)
            elif len(interval2)!=0:
                start, end = random.choice(interval2)
            else:
                start = random.randint(0, 10000)
                end = random.randint(30000, 60000)
        seg_len_ms     = int(self.segment * 1000)                 # self.segment : 초
        duration_ms    = end - start
        max_offset_ms  = max(0, duration_ms - seg_len_ms)
        offset_ms      = random.randint(0, max_offset_ms) if max_offset_ms else 0

        # ----- (4) ms → sec 로 변환해 wav 로더에 전달 -----
        start_sec = (start + offset_ms) / 1000.0                  # float sec

        segment1 = load_wav_specific_position_mono(
            path1,
            self.sample_rate,          # 보통 16_000
            self.segment,              # 초 단위 길이
            start_position=start_sec
        )
        segment2 = load_wav_specific_position_mono(
            path2,
            self.sample_rate,
            self.segment,
            start_position=start_sec
        )
        return segment1, segment2

    def load_harmony_audio(self, path, start_ms, end_ms, offset=None):
        audio, sr = torchaudio.load(path)  # (1, T) or (2, T)
        
        if sr != self.sr:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sr)
        audio = audio[0]  # mono

        start_sample = int(start_ms * self.sr / 1000)
        segment_len = int(self.segment_seconds * self.sr)
        duration_samples = int((end_ms - start_ms) * self.sr / 1000)
        total_len = audio.size(0)

        if duration_samples >= segment_len:
            max_offset = duration_samples - segment_len
            if offset is None:
                offset = random.randint(0, max_offset)
            segment = audio[start_sample + offset : start_sample + offset + segment_len]
        else:
            segment_end = start_sample + segment_len
            if segment_end <= total_len:
                segment = audio[start_sample : segment_end]
            else:
                segment = audio[start_sample:]
                pad_len = segment_len - segment.size(0)
                segment = torch.nn.functional.pad(segment, (0, pad_len))  # (T,)

        return segment

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
        root_dir="duet_svs/MedleyVox_4s_16k",
        metadata_dir=None,
        simple=False,
        task="duet",
        sample_rate=16000,
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
            s = load_wav_from_start_mono(source_path)
            sources_list.append(s)
            ids.append(os.path.basename(source_path).replace(".wav", ""))
        # Read the mixture
        mixture = load_wav_from_start_mono(mixture_path)
        mixture = torch.as_tensor(mixture, dtype=torch.float32)
        
        sources = np.vstack(sources_list)
        sources = torch.as_tensor(sources, dtype=torch.float32)   
        return mixture, sources,ids[0]


class MedleyVox(Dataset):
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

    dataset_name = "MedleyVox"

    def __init__(
        self,
        root_dir="duet_svs/MedleyVox",
        metadata_dir=None,
        task="duet",
        sample_rate=16000,
        n_src=2,
        segment=None,
        return_id=True,
    ):
        self.root_dir = root_dir  # /path/to/data/test_medleyDB
        self.metadata_dir = metadata_dir  # ./testset/testset_config
        self.task = task.lower()
        self.return_id = return_id
        # Get the csv corresponding to the task
        if self.task == "unison":
            self.total_segments_list = glob.glob(f"{self.root_dir}/unison/*/*")
        elif self.task == "duet":
            self.total_segments_list = glob.glob(f"{self.root_dir}/duet/*/*")
        elif self.task == "main_vs_rest":
            self.total_segments_list = glob.glob(f"{self.root_dir}/rest/*/*")
        elif self.task == "n_singing":
            self.total_segments_list = glob.glob(f"{self.root_dir}/unison/*/*") + glob.glob(f"{self.root_dir}/duet/*/*") + glob.glob(f"{self.root_dir}/rest/*/*")
        self.segment = segment
        self.sample_rate = sample_rate
        self.n_src = n_src

    def __len__(self):
        return len(self.total_segments_list)

    def __getitem__(self, idx):
        song_name = self.total_segments_list[idx].split("/")[-2]
        segment_name = self.total_segments_list[idx].split("/")[-1]
        mixture_path = (
            f"{self.total_segments_list[idx]}/mix/{song_name} - {segment_name}.wav"
        )
        self.mixture_path = mixture_path
        sources_path_list = glob.glob(f"{self.total_segments_list[idx]}/gt/*.wav")

        if self.task == "main_vs_rest" or self.task == "n_singing":
            if os.path.exists(
                f"{self.metadata_dir}/V1_rest_vocals_only_config/{song_name}.json"
            ):
                metadata_json_path = (
                    f"{self.metadata_dir}/V1_rest_vocals_only_config/{song_name}.json"
                )
            elif os.path.exists(
                f"{self.metadata_dir}/V2_vocals_only_config/{song_name}.json"
            ):
                metadata_json_path = (
                    f"{self.metadata_dir}/V2_vocals_only_config/{song_name}.json"
                )
            else:
                print("main vs. rest metadata not found.")
                raise AttributeError
            with open(metadata_json_path, "r") as json_file:
                metadata_json = json.load(json_file)

        # Read sources
        sources_list = []
        ids = []
        if self.task == "main_vs_rest" or self.task == "n_singing":
            gt_main_name = metadata_json[segment_name]["main_vocal"]
            gt_source, sr = librosa.load(
                f"{self.total_segments_list[idx]}/gt/{gt_main_name} - {segment_name}.wav",
                sr=self.sample_rate,
            )
            gt_rest_list = metadata_json[segment_name]["other_vocals"]
            ids.append(f"{gt_main_name} - {segment_name}")

            rest_sources_list = []
            for other_vocal_name in gt_rest_list:
                s, sr = librosa.load(
                    f"{self.total_segments_list[idx]}/gt/{other_vocal_name} - {segment_name}.wav",
                    sr=self.sample_rate,
                )
                rest_sources_list.append(s)
                ids.append(f"{other_vocal_name} - {segment_name}")
            rest_sources_list = np.stack(rest_sources_list, axis=0)
            gt_rest = rest_sources_list.sum(0)

            sources_list.append(gt_source)
            sources_list.append(gt_rest)
        else: # self.task == 'unison' or self.task == 'duet'
            for i, source_path in enumerate(sources_path_list):
                s, sr = librosa.load(source_path, sr=self.sample_rate)
                sources_list.append(s)
                ids.append(os.path.basename(source_path).replace(".wav", ""))
        # Read the mixture
        mixture, sr = librosa.load(mixture_path, sr=self.sample_rate)
        # Convert to torch tensor
        mixture = torch.as_tensor(mixture, dtype=torch.float32)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.as_tensor(sources, dtype=torch.float32)
        if not self.return_id:
            return mixture, sources
        # 5400-34479-0005_4973-24515-0007.wav
        # id1, id2 = mixture_path.split("/")[-1].split(".")[0].split("_")

        return mixture, sources, ids
def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch)
class OptimDataModule16k(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        n_src: int = 2,
        sample_rate: int = 16000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        segment_seconds: float = 4.0,
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
        self.segment = segment
        self.normalize_audio = normalize_audio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self,stage=None) -> None:
        self.data_train = DuetSingingSpeechMixTraining(
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
            drop_last=True,collate_fn=safe_collate
        )

    def val_dataloader(self) -> DataLoader:
        return[DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,collate_fn=safe_collate
        ), DataLoader(
            dataset=self.data_test,
            shuffle=False,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,collate_fn=safe_collate
        )]

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            shuffle=False,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,collate_fn=safe_collate
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    @property
    def make_sets(self):
        return self.data_train, self.data_val, self.data_test
