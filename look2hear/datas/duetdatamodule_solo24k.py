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
def generate_random_string(length=5):
    import random
    import string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def save_waveform(waveform, sr, save_path):
    waveform = waveform.detach().cpu().numpy()
    sf.write(f"{save_path}", waveform, sr)
    print(f"{save_path}")
def plot_and_save_spectrogram(waveform, sr, n_fft, hop_length, win_length, save_path, title="Spectrogram"):
    spec = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=torch.hann_window(win_length),
                      return_complex=True)
    mag = spec.abs().clamp(min=1e-5)
    log_mag = torch.log(mag).cpu().numpy()

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mag, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_dataloaders(args, dataset_config, loader_config):    
    # create dataset object for each partition
    partitions = ["test"] if "test" in args.engine_mode  else ["train", "valid", "test"]
    dataloaders = {}
    for partition in partitions:
        dataset = MyDataset(
            partition = partition,
            )
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = 1 if partition == 'test' else loader_config["batch_size"],
            shuffle = True, # only train: (partition == 'train') / all: True
            pin_memory = loader_config["pin_memory"],
            num_workers = loader_config["num_workers"],
            drop_last = loader_config["drop_last"],
            collate_fn = safe_collate)
        dataloaders[partition] = dataloader
    return dataloaders

def safe_collate(batch):
    filtered_batch = [
        item for item in batch
        if item is not None and all(sub_item is not None for sub_item in item[:6])
    ]
    return torch.utils.data.dataloader.default_collate(filtered_batch) if filtered_batch else None

def _collate(egs):
    """
        Transform utterance index into a minbatch

        Arguments:
            index: a list type [{},{},{}]

        Returns:
            input_sizes: a tensor correspond to utterance length
            input_feats: packed sequence to feed networks
            source_attr/target_attr: dictionary contains spectrogram/phase needed in loss computation
    """
    def __prepare_target_rir(dict_lsit, index):
        return torch.nn.utils.rnn.pad_sequence([torch.tensor(d["src"][index], dtype=torch.float32)  for d in dict_lsit], batch_first=True)
    if type(egs) is not list: raise ValueError("Unsupported index type({})".format(type(egs)))
    num_spks = 2 # you need to set this paramater by yourself
    dict_list = sorted([eg for eg in egs], key=lambda x: x['num_sample'], reverse=True)
    mixture = torch.nn.utils.rnn.pad_sequence([torch.tensor(d['mix'], dtype=torch.float32) for d in dict_list], batch_first=True)
    src = [__prepare_target_rir(dict_list, index) for index in range(num_spks)]
    input_sizes = torch.tensor([d['num_sample'] for d in dict_list], dtype=torch.float32)
    key = [d['key'] for d in dict_list]
    return input_sizes, mixture, src, key


def append_file_list(path):
    file_list_ = glob.glob(path)
    file_list=[]
    for file_path in file_list_:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        group = []
        for line in lines:
            line = line.strip()
            if line == "":
                continue
            group.append(line)  # 여기에 pitch_npy 경로가 들어 있음
        
        file_list.append(group)
    return file_list


class MyDataset(Dataset):
    def __init__(self, isfinetune=False,simple=False,sample_rate=24000,
                 n_fft=1024, hop_length=160, win_length=1024,
                 segment_seconds=4.0,segment=None):
        #self.file_list =  glob.glob("duet_svs/**/*.wav", recursive=True)
        filelist_txt_path = "duet_svs/24k/24k_all_filelist.txt"
        with open(filelist_txt_path, "r") as f:
            all_wav_files = [line.strip() for line in f if line.strip()]
            print(f"기존 filelist_txt_path.txt에서 {len(all_wav_files)}개의 파일을 불러왔습니다.")

        self.file_list = all_wav_files
        self.simple=simple
        self.sr = sample_rate
        self.segment_seconds=segment_seconds#=2.7
        print(f"[INFO] Segment seconds: {self.segment_seconds}")
        self.segment_len = int(segment_seconds * sample_rate)
        self.n_fft = n_fft
        self.hop = hop_length
        self.isfinetune = isfinetune
        self.win = win_length
        self.window = torch.hann_window(win_length)
        self.frame_len=int(self.segment_len // self.hop)
        self.jacapella_list=glob.glob("duet_svs/jaCappella/**/*.wav", recursive=True)
        self.jacapella_list = [x for x in self.jacapella_list if "mixture" not in x and "vocal_percussion" not in x and "finger_snap" not in x]
        self.jacapella_list =[x for x in self.jacapella_list if "_16k.wav" not in x]
        self.jacapella_list =[x for x in self.jacapella_list if "_24K.wav" not in x] #serifu가 있는 파일은 제외
        self.choir_list=glob.glob("duet_svs/Choir_Dataset/**/*.wav", recursive=True)
        self.choir_list=[x for x in self.choir_list if "CANTUS" in x or "Alto" in x or "Soprano" in x or "Tenor" in x or "Bass" in x or "Cantus" in x or "Bassus" in x or "Altus" in x or "BASSUS" in x or "ALTUS" in x or "SOPRANO" in x or "TENOR" in x or "ALTO" in x or "BASS" in x] 
        self.choir_list=[x for x in self.choir_list if "_16k.wav" not in x] #16k로 바꿔야함
        self.choir_list=[x for x in self.choir_list if "_24K.wav" not in x] #serifu가 있는 파일은 제외
        self.jpop_list=glob.glob("duet_svs/datasets--imprt--idol-songs-jp/snapshots/c026e76507d574b4f79efb0f01e41fb1b421b563/vocals_24k/**/*.wav", recursive=True)
        self.jpop_list=[x for x in self.jpop_list if "serifu" not in x and "call" not in x]
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
        same_singer_json_path = "duet_svs/24k/json/same_singer.json"
        with open(same_singer_json_path, "r") as f:
            self.same_singer_dict = json.load(f)
        same_song_json_path = "duet_svs/24k/json/same_song.json"
        with open(same_song_json_path, "r") as f:
            self.same_song_dict = json.load(f)
        self.unison_prob=0.08
        if not self.isfinetune: #Best pop , diff, HARMONY, SAME SINGER, SAME SONG, lead_align
            self.rand_order=[0.0,0.6,0.6,0.85,0.1]
        else:
            print("[INFO] Finetune mode activated. Using different random order for data selection.")
            self.rand_order=[0.05,0.1,0.8,0.85,0.95] #finetune에서는 HARMONY를 더 많이 사용
        #self.lead_align_path=glob.glob("duet_svs/best_pop_song/lead_align/*/*.csv")
        song_length_dict_path="duet_svs/24k/json/song_length_dict_24k_mine.json"
        with open(song_length_dict_path, "r") as json_file:
            song_length_dict = json.load(json_file)
        # sort dict by descending order
        song_length_dict = dict(
            sorted(song_length_dict.items(), key=lambda x: x[1], reverse=True)
        )
        song_names = []
        song_lengths = []
        for key, value in song_length_dict.items():
            song_names.append(key)
            song_lengths.append(value)
        # Determine how many times to load one audio file during one epoch
        train_list_number = np.array(song_lengths) / (self.segment_seconds * self.sr)
        self.singing_train_list = []
        song_name_path_dict = {}
        for data_path in self.file_list:
            song_name_path_dict[os.path.basename(data_path)] = data_path
        for i, num_seg in enumerate(list(train_list_number)):
            try:
                self.singing_train_list.extend(
                    [song_name_path_dict[song_names[i]]] * math.ceil(num_seg)
                )
            except KeyError:  # some songs might not be in the self.song_name_path_dict
                pass
        self.len_singing_train_list = len(self.singing_train_list)
        self.reduced_training_data_ratio=0.03
    def __len__(self):
        debug=False
        if debug:
            return 5
        return  int(self.len_singing_train_list * self.reduced_training_data_ratio)
    def load_and_segment_together(self, path1, path2, start1: float = None, end1: float = None, start2: float = None, end2: float = None):
        waveform1, sr1 = torchaudio.load(path1)
        waveform2, sr2 = torchaudio.load(path2)

        if sr1 != self.sr:
            waveform1 = torchaudio.functional.resample(waveform1, sr1, self.sr)
        if sr2 != self.sr:
            waveform2 = torchaudio.functional.resample(waveform2, sr2, self.sr)

        waveform1 = waveform1[0]  # mono
        waveform2 = waveform2[0]  # mono

        start_sample1, start_sample2 = 0, 0  # default는 전체 waveform

        if start1 is not None and end1 is not None:
            start_sample1 = int(start1 * self.sr)
            end_sample1 = int(end1 * self.sr)
            end_sample1 = min(end_sample1, waveform1.size(0))
            if end_sample1 <= start_sample1:
                raise ValueError(f"Invalid start1 and end1: {start1}, {end1}")
            waveform1 = waveform1[start_sample1:end_sample1]

        if start2 is not None and end2 is not None:
            start_sample2 = int(start2 * self.sr)
            end_sample2 = int(end2 * self.sr)
            end_sample2 = min(end_sample2, waveform2.size(0))
            if end_sample2 <= start_sample2:
                raise ValueError(f"Invalid start2 and end2: {start2}, {end2}")
            waveform2 = waveform2[start_sample2:end_sample2]

        len1, len2 = waveform1.size(0), waveform2.size(0)
        min_len = min(len1, len2)

        if min_len < self.segment_len:
            if len1 < self.segment_len:
                waveform1 = torch.nn.functional.pad(waveform1, (0, self.segment_len - len1))
            if len2 < self.segment_len:
                waveform2 = torch.nn.functional.pad(waveform2, (0, self.segment_len - len2))
            relative_offset = 0
        else:
            relative_offset = random.randint(0, min_len - self.segment_len)

        # 전체 waveform 기준의 절대 offset 계산
        offset1 = start_sample1 + relative_offset
        offset2 = start_sample2 + relative_offset

        segment1 = waveform1[relative_offset:relative_offset + self.segment_len]
        segment2 = waveform2[relative_offset:relative_offset + self.segment_len]

        return segment1, segment2, offset1, offset2
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

    def load_and_segment(self, path, max_attempts=5, silence_threshold=1e-4):
        waveform, sr = torchaudio.load(path)
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        waveform = waveform[0]  # mono

        if waveform.size(0) < self.segment_len:
            pad_len = self.segment_len - waveform.size(0)
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            offset = 0
            segment = waveform[:self.segment_len]
            return segment, offset

        for _ in range(max_attempts):
            offset = random.randint(0, waveform.size(0) - self.segment_len)
            segment = waveform[offset:offset + self.segment_len]
            rms = torch.sqrt(torch.mean(segment ** 2))
            if rms > silence_threshold:
                return segment, offset

        # fallback: 그냥 마지막 시도한 segment 반환
        return segment, offset
    def load_source(self,key_path,save_audio=False):
        max_attempt=10
        for att in range(max_attempt):
            try:
                align_path=key_path.replace("vocal_sep", "align")
                align_files=glob.glob(os.path.join(align_path, "*.csv"))
                align_file_path=random.choice(align_files)
                source1_basename = os.path.basename(align_file_path).split("_vs_")[0]
                source2_basename = os.path.basename(align_file_path).split("_vs_")[1].split(".csv")[0]
                source1_wav_path=os.path.join(key_path, source1_basename+".wav")
                source2_wav_path=os.path.join(key_path, source2_basename+".wav")
                df = pd.read_csv(align_file_path)
                row = df.sample(n=1)

                source1_start = float(row['Cover_1_Timestamp'].values[0].split(",")[0].split("(")[1])
                source1_end = float(row['Cover_1_Timestamp'].values[0].split(",")[1].split(")")[0])
                source2_start = float(row['Cover_2_Timestamp'].values[0].split(",")[0].split("(")[1])
                source2_end = float(row['Cover_2_Timestamp'].values[0].split(",")[1].split(")")[0])
                break
            except Exception as e:
                key_num=random.randint(0, 697)
                key_path=os.path.join("duet_svs/best_pop_song/vocal_sep",str(key_num))
                if att == max_attempt-1:
                    raise e
        return source1_wav_path, source2_wav_path, source1_start, source1_end, source2_start, source2_end
    def shift_left(self,segment, offset, shift_frames=10):
        shift_samples = shift_frames * self.hop   # = 1600
        seg_len = segment.size(0)
        
        if shift_samples >= seg_len:
            # 너무 많이 밀어서 다 사라지면 0으로 채우기
            shifted = torch.zeros_like(segment)
        else:
            # 왼쪽으로 밀고, 오른쪽에 0 padding
            shifted = torch.cat([segment[shift_samples:], torch.zeros(shift_samples, dtype=segment.dtype)])

        # offset도 앞으로 당긴 만큼 뒤로 보정
        new_offset = offset + shift_samples
        return shifted, new_offset
    def select_existing_piano_roll(self,path):
        """
        path: .wav 파일 경로
        리턴: 선택된 piano_roll npy 파일의 로드 결과 (np.ndarray), 또는 None
        """
        pitch_path = path.replace(".wav", "_piano_roll.npy")
        rosvot_path = path.replace(".wav", "_rosvot_piano_roll_rosvot.npy")

        pitch_exists = os.path.exists(pitch_path)
        rosvot_exists = os.path.exists(rosvot_path)

        if pitch_exists and rosvot_exists:
            chosen_path = pitch_path if random.random() < 0.8 else rosvot_path
        elif pitch_exists:
            chosen_path = pitch_path
        elif rosvot_exists:
            chosen_path = rosvot_path
        else:
            return None  # or raise FileNotFoundError(path)

        return np.load(chosen_path)
    def test_getitem(self, idx):
        song_name = self.total_segments_list[idx].split("/")[-2]
        segment_name = self.total_segments_list[idx].split("/")[-1]
        mixture_path = (
            f"{self.total_segments_list[idx]}/mix/{song_name} - {segment_name}.wav"
        )
        sources_path_list = glob.glob(f"{self.total_segments_list[idx]}/gt/*.wav")
        # Read sources
        sources_list = []
        ids = []
        for i, source_path in enumerate(sources_path_list):
            s, sr = torchaudio.load(source_path)
            if sr != self.sr:
                s = torchaudio.functional.resample(s, sr, self.sr)
            sources_list.append(s)
            ids.append(os.path.basename(source_path).replace(".wav", ""))
        # Read the mixture
        mixture, sr = torchaudio.load(mixture_path)
        if sr != self.sr:
            mixture = torchaudio.functional.resample(mixture, sr, self.sr)
        waveform1 = sources_list[0][0]  # mono
        waveform2 = sources_list[1][0]  # mono
        mixture= mixture[0]  # mono
        # Convert sources to tensor
        sources = torch.stack([waveform1, waveform2])
        
        return mixture, sources, ids[0]
    def __getitem__(self, idx):
        max_attempts = 5
        for attempt in range(max_attempts):
            data = self.getitem(idx if attempt == 0 else random.randint(0, len(self)-1))
            if data is not None:
                #mixture, sources, emb1,emb2,path1= data
                return data
            # 모든 시도 실패 → dummy 반환
        print(f"[DUMMY] All {max_attempts} attempts failed at index {idx}")
        dummy_mixture = torch.zeros(self.segment_len)
        dummy_sources = torch.zeros(2, self.segment_len)
        dummy_path1 = "dummy.wav"
        emb1 = torch.zeros(192)  # 예시로 192차원 임베딩
        if self.simple:
            return dummy_mixture, dummy_sources, dummy_path1
        return dummy_mixture, dummy_sources, emb1,emb1,dummy_path1
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
        duration_samples = int((end - start) * self.sr / 1000)
        
        if duration_samples >= self.segment_len:
            max_offset = duration_samples - self.segment_len
            offset = random.randint(0, max_offset)
        else:
            offset = None  # fallback for padding mode

        segment1 = self.load_harmony_audio(path1, start, end, offset=offset)
        segment2 = self.load_harmony_audio(path2, start, end, offset=offset)
        return segment1, segment2
    def load_and_segment_with_offset(self, path, offset=None):
        waveform, sr = torchaudio.load(path)
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        waveform = waveform[0]  # mono
        
        total_len = waveform.size(0)

        if offset is None:
            offset = 0

        end = offset + self.segment_len

        if end > total_len:
            # 부족한 길이만큼 padding 추가
            pad_len = end - total_len
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        segment = waveform[offset:offset + self.segment_len]
        return segment, offset
    def _crop_or_pad(self, audio):
        audio_len = audio.size(0)
        if audio_len >= self.segment_len:
            max_offset = audio_len - self.segment_len
            offset = random.randint(0, max_offset)
            return audio[offset:offset + self.segment_len]
        else:
            pad_len = self.segment_len - audio_len
            return torch.nn.functional.pad(audio, (0, pad_len))
    def load_align(self):
        max_attempts=5
        for attempt in range(max_attempts):
            csv_path = random.choice(self.lead_align_path)
            df = pd.read_csv(csv_path)
            if not df.empty:
                break
        else:
            dummy = torch.zeros(self.segment_len)
            return dummy, dummy

        row = df.sample(n=1).iloc[0]
        folder = os.path.basename(os.path.dirname(csv_path))
        basename = os.path.basename(csv_path).replace(".csv", "")

        # 오디오 경로
        wav_path_1 = f"duet_svs/best_pop_song/lead_sep/{folder}/{basename}.wav"
        # 시작/끝 시간
        start1, end1 = row["start_1"], row["end_1"]
        start2, end2 = row["start_2"], row["end_2"]

        # wave 로딩
        wav1, sr = torchaudio.load(wav_path_1)
        #wav2, _ = torchaudio.load(wav_path_1)
        if sr != self.sr:
            waveform1 = torchaudio.functional.resample(wav1, sr, self.sr)
        # if sr != self.sr:
        #     waveform2 = torchaudio.functional.resample(wav2, sr, self.sr)

        waveform1 = waveform1[0]  # mono
        #waveform2 = waveform2[0]  # mono

        # 초 → 샘플 인덱스
        s1 = int(start1 * self.sr)
        e1 = int(end1 * self.sr)
        s2 = int(start2 * self.sr)
        e2 = int(end2 * self.sr)

        seg1 = self._crop_or_pad(waveform1[s1:e1])
        seg2 = self._crop_or_pad(waveform1[s2:e2])

        return seg1, seg2,wav_path_1
    def getitem(self, idx):
        path1, path2 = None, None  # 예외 처리용 초기화
        save_=False
        
        #model_output, midi_data, note_events = predict(<input-audio-path>)
        try:
            rand_num1=random.random()#이거는 best_pop_song에서 가져옴.
            rand_num2=random.random()#이거는 같은 사람 목소리 조금위치달리해서 가져옴
            if rand_num1<0.2:
                category="diff_singer"
                if not self.isfinetune:
                    if rand_num2<0.7:
                        path1 = random.choice(self.file_list)#.replace("_piano_roll.npy", ".wav")
                        path2 = random.choice(self.file_list)#.replace("_piano_roll.npy", ".wav")
                        w1, offset1 = self.load_and_segment(path1) #(128000)
                        w2, offset2 = self.load_and_segment(path2)
                    else:
                        path1 = random.choice(self.file_list)#.replace("_piano_roll.npy", ".wav")
                        w1, offset1 = self.load_and_segment(path1) #(128000)
                        w2 = torch.from_numpy(change_pitch_and_formant_random(w1,self.sr)).type(w1.dtype)
                else:
                    path1 = random.choice(self.file_list)#.replace("_piano_roll.npy", ".wav")
                    w1, offset1 = self.load_and_segment(path1) #(128000)
                    w2 = torch.from_numpy(change_pitch_and_formant_random(w1,self.sr)).type(w1.dtype)
            elif rand_num1<0.4:
                category="same_singer"
                dir_= random.choice(list(self.same_singer_dict.keys()))
                path1, path2 = random.sample(self.same_singer_dict.get(dir_, []), 2)
                w1, offset1 = self.load_and_segment(path1) #(128000)
                w2, offset2 = self.load_and_segment(path2)
            # elif rand_num1<0.8:
            #     category="harmony"
            #     if rand_num2<0.2: #jacapella
            #         dir_ = random.choice(list(self.jacapella_dict.keys()))
            #         path1, path2 = random.sample(self.jacapella_dict.get(dir_, []), 2)
            #     elif rand_num2<0.4: #choirdataset
            #         dir_ = random.choice(list(self.choir_dict.keys()))
            #         path1, path2 = random.sample(self.choir_dict.get(dir_, []), 2)
            #     else: #jpop
            #         dir_ = random.choice(list(self.jpop_dict.keys()))
            #         path1, path2 = random.sample(self.jpop_dict.get(dir_, []), 2)
            #     w1,w2=self.load_harmody(path1, path2)

            else:
                category="same_song"
                dir_= random.choice(list(self.same_song_dict.keys()))
                path1, path2 = random.sample(self.same_song_dict.get(dir_, []), 2)
                w1, offset1 = self.load_and_segment(path1) #(128000)
                w2, offset2 = self.load_and_segment_with_offset(path2,offset1)
            
            #segment_len = self.segment_len  # 예: segment=4이면 64000
            time=w1.shape[-1]
            
            gain1 = pow(10,-random.uniform(-1.5,1.5)/20)
            gain2 = pow(10,-random.uniform(-1.5,1.5)/20)
            w1 = w1 * gain1
            #normalize w1
            w2 = w2 * gain2
            if not self.isfinetune:
                if random.random() < self.unison_prob:
                    w2 = torch.from_numpy(change_pitch_and_formant_random(w1,self.sr)).type(w1.dtype)
            mixture = w1 + w2
            sources = torch.cat([w1.unsqueeze(0), w2.unsqueeze(0)], dim=0)
            # if torch.distributed.is_initialized():
            #     rank = torch.distributed.get_rank()
            #     print(f"[Rank {rank}] __getitem__ called: idx={path1}")
            # else:
            #     print(f"[Single-process] __getitem__ called: idx={path1}")
            if save_:
                base = os.path.splitext(os.path.basename(path1))[0]
                folder=f"result/yeah/wav/{category}_{generate_random_string(3)}"
                os.makedirs(folder, exist_ok=True)
                save_waveform(w1, self.sr, f"{folder}/{base}_1.wav")
                save_waveform(w2, self.sr, f"{folder}/{base}_2.wav")
                save_waveform(mixture, self.sr, f"{folder}/{base}.wav")
            
            
            if self.simple:
                return mixture, sources, path1#, path2
            with torch.no_grad():
                spk_emb1=self.get_chunked_embedding(w1.unsqueeze(0)).squeeze(0)  # (num_chunks, emb_dim)
                spk_emb2=self.get_chunked_embedding(w2.unsqueeze(0)).squeeze(0)  # (num_chunks, emb_dim)
                cos_sim = F.cosine_similarity(spk_emb1, spk_emb2, dim=-1).mean()
            
            
            return mixture,sources,spk_emb1,spk_emb2,path1#,path2
        except Exception as e:
            print(f"Error processing index {path1,path2}: {e}")
            return None
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
def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    # if len(batch) == 0:
    #     batch=
    return torch.utils.data.dataloader.default_collate(batch)
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
          # Set to evaluation mode
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

class DuetDataModuleTime(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        n_src: int = 2,
        sample_rate: int = 16000,
        segment_seconds: float = 4.0,
        normalize_audio: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
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

    def setup(self,stage=None) -> None:
        self.data_train = MyDataset(
            segment_seconds=self.segment_seconds
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
        return [DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,collate_fn=safe_collate
        ),DataLoader(
            dataset=self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,collate_fn=safe_collate
        )]

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
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

def seed_worker(worker_id):
    # Lightning 내부에서 torch.initial_seed() 에
    #   global_seed + rank + epoch 가 이미 섞여 있습니다.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
class DuetDataModuleTimeSolo24k(LightningDataModule):
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
        segment=None
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

    def setup(self,stage=None) -> None:
        self.data_train = MyDataset(
            simple=True,
            segment_seconds=self.segment_seconds
        )
        self.data_val = MyMedleyVox(
            task="duet",  # or "unison" or "main_vs_rest"
            simple=True,
        )
        self.data_test = MyMedleyVox(
            task="unison",simple=True,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            #worker_init_fn=seed_worker,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,collate_fn=safe_collate
        )

    def val_dataloader(self) -> DataLoader:
        return [DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,collate_fn=safe_collate
        ),DataLoader(
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

class DuetDataModuleTimeSimpleFinetune(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        n_src: int = 2,
        sample_rate: int = 16000,
        segment_seconds: float = 4.0,
        normalize_audio: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
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

    def setup(self,stage=None) -> None:
        self.data_train = MyDataset(
            simple=True,
            isfinetune=True,
            segment_seconds=self.segment_seconds
        )
        self.data_val = MyMedleyVox(
            task="duet",  # or "unison" or "main_vs_rest"
            simple=True,
        )
        self.data_test = MyMedleyVox(
            task="unison",simple=True,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            #worker_init_fn=seed_worker,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,collate_fn=safe_collate
        )

    def val_dataloader(self) -> DataLoader:
        return [DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,collate_fn=safe_collate
        ),DataLoader(
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



class DuetDataModuleTimeFinetune(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        n_src: int = 2,
        sample_rate: int = 16000,
        segment_seconds: float = 4.0,
        normalize_audio: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,task="unison",
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
        self.task=task
    def setup(self) -> None:
        self.data_train = MyDataset(
            isfinetune=True,  # finetune용 데이터셋
            segment_seconds=self.segment_seconds
        )
        self.data_val = MyMedleyVox(
            task="duet"
        )
        self.data_test = MyMedleyVox(
            task=self.task
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=safe_collate
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,collate_fn=safe_collate
        )

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


            
if __name__=="__main__":
    dataset = MyDataset()
    #dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataset:
        print(batch[-1])
        #break