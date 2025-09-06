import os
import random
import json
from tqdm import tqdm
from pprint import pprint
import look2hear.models
import torchaudio
from torch.utils.data import Dataset
import glob
import look2hear.datas
import os
import torch
import random
import librosa as audio_lib
import numpy as np
import torchaudio
from look2hear.utils import change_pitch_and_formant_random
from speechbrain.inference.speaker import EncoderClassifier
import torch.nn.functional as F 
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
from look2hear.metrics import MetricsTracker
from look2hear.utils import tensors_to_device, RichProgressBarTheme, MyMetricsTextColumn, BatchesProcessedColumn#,save_state_dict_keys
#from functions import load_ola_func_with_args#, load_w2v_func_with_args, load_w2v_chunk_func_with_args, load_spectral_features_chunk_func_with_args
import soundfile as sf
import librosa
import numpy as np
import torch
import argparse
import pandas as pd
import pyloudnorm as pyln
from asteroid.metrics import get_metrics
#from asteroid.data.librimix_dataset import LibriMix
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.utils import tensors_to_device

#from data import MyMedleyVox
#from models import load_model_with_args
#from functions import load_ola_func_with_args
from look2hear.utils import str2bool, loudnorm, db2linear

import yaml
def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
#COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi"]
COMPUTE_METRICS = ["si_sdr", "sdr"]
def resample_to_16k(waveform, orig_sr, target_sr=16000):
    if waveform.ndim == 1:
        return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    elif waveform.ndim == 2:
        return np.stack([
            librosa.resample(ch, orig_sr=orig_sr, target_sr=target_sr)
            for ch in waveform
        ])
    else:
        raise ValueError("Unsupported waveform shape")

class MyJpopDataset(Dataset):
    def __init__(self, isfinetune=False,simple=True,sample_rate=16000,
                 n_fft=1024, hop_length=160, win_length=1024,
                 segment_seconds=30.0,data="jpop"):
        #self.file_list =  glob.glob("duet_svs/**/*.wav", recursive=True)
        filelist_txt_path = "all_filelist.txt"
        self.isfinetune = isfinetune
        if os.path.exists(filelist_txt_path):
            # 이미 txt가 있다면 그것을 불러오기
            with open(filelist_txt_path, "r") as f:
                all_wav_files = [line.strip() for line in f if line.strip()]
            print(f"기존 all_filelist.txt에서 {len(all_wav_files)}개의 파일을 불러왔습니다.")
        else:
            dirs = [
                "/home/jungji/real_sep/tiger_ver3/duet_svs/CSD",
                "/home/jungji/real_sep/tiger_ver3/duet_svs/GTSinger_unzipped",
                "/home/jungji/real_sep/tiger_ver3/duet_svs/jaCappella",
                "/home/jungji/real_sep/tiger_ver3/duet_svs/m4singer",
                "/home/jungji/real_sep/tiger_ver3/duet_svs/opencpop/wavs",
                "/home/jungji/real_sep/tiger_ver3/duet_svs/OpenSinger",
                #"/home/jungji/real_sep/tiger_ver3/duet_svs/VocalSet/FULL"
            ]
            existing_dirs = [d for d in dirs if os.path.isdir(d)]

            all_wav_files = []
            for d in existing_dirs:
                wavs = glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)
                all_wav_files.extend(wavs)

            print(f"총 {len(all_wav_files)}개의 wav 파일을 찾았습니다. 파일 저장 중...")

            with open(filelist_txt_path, "w") as f:
                for path in all_wav_files:
                    f.write(path + "\n")

            print(f"{filelist_txt_path}에 저장 완료.")
        self.file_list = all_wav_files
        self.simple=simple
        self.sr = sample_rate
        self.segment_seconds=segment_seconds#=2.7
        print(f"[INFO] Segment seconds: {self.segment_seconds}")
        self.segment_len = int(segment_seconds * sample_rate)
        self.n_fft = n_fft
        self.hop = hop_length
        
        self.win = win_length
        self.window = torch.hann_window(win_length)
        self.frame_len=int(self.segment_len // self.hop)
        if not self.simple:
            self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")#.eval()
            self.classifier.eval()  # Set to evaluation mode
        self.jacapella_list=glob.glob("duet_svs/jaCappella/**/*.wav", recursive=True)
        self.jacapella_list = [x for x in self.jacapella_list if "mixture" not in x and "vocal_percussion" not in x and "finger_snap" not in x]
        self.jacapella_list =[x for x in self.jacapella_list if "_16k.wav" not in x]
        self.jacapella_list =[x for x in self.jacapella_list if "_24K.wav" not in x] #serifu가 있는 파일은 제외
        self.choir_list=glob.glob("duet_svs/Choir_Dataset/**/*.wav", recursive=True)
        self.choir_list=[x for x in self.choir_list if "CANTUS" in x or "Alto" in x or "Soprano" in x or "Tenor" in x or "Bass" in x or "Cantus" in x or "Bassus" in x or "Altus" in x or "BASSUS" in x or "ALTUS" in x or "SOPRANO" in x or "TENOR" in x or "ALTO" in x or "BASS" in x] 
        self.choir_list=[x for x in self.choir_list if "_16k.wav" not in x] #16k로 바꿔야함
        self.choir_list=[x for x in self.choir_list if "_24K.wav" not in x] #serifu가 있는 파일은 제외
        self.jpop_list=glob.glob("duet_svs/datasets--imprt--idol-songs-jp/snapshots/c026e76507d574b4f79efb0f01e41fb1b421b563/vocals_16k/**/*.wav", recursive=True)
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
        self.data=data
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

        
    def __len__(self):
        debug=False
        if debug:
            return 5
        if self.data=="jacapella":
            return self.jacapella_dict.keys().__len__()#*10
        return self.jpop_dict.keys().__len__()#*10 
    
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
        if self.data=="jpop":
            dir_ = list(self.jpop_dict.keys())[idx]
            paths = self.jpop_dict.get(dir_, [])
            path1, path2 = paths[0], paths[1]
        elif self.data=="jacapella":
            dir_ = list(self.jacapella_dict.keys())[idx]
            paths = self.jacapella_dict.get(dir_, [])
            path1, path2 = paths[0], paths[1]
            #path1, path2 = random.sample(self.jacapella_dict.get(dir_, []), 2)
        w1,w2=self.load_harmody(path1, path2)
        time=w1.shape[-1]
        gain1 = pow(10,-random.uniform(-1.5,1.5)/20)
        gain2 = pow(10,-random.uniform(-1.5,1.5)/20)
        w1 = w1 * gain1
        w2 = w2 * gain2
        mixture = w1 + w2
        
        sources = torch.cat([w1.unsqueeze(0), w2.unsqueeze(0)], dim=0)
        return mixture, sources, path1+"_"+path2#, path2
            


def main(args):
    compute_metrics = COMPUTE_METRICS

    # Handle device placement
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    )#inference_speech_my.py
    model_path="Experiments/checkpoint/combine7_finetune/20250704_07/epoch=36.ckpt" #Experiments/checkpoint/ver1/20240621_07/config.yaml
    exp_dir = os.path.dirname(model_path) #Experiments/checkpoint/melt6/epoch=21.ckpt
    conf_dir = os.path.join(exp_dir,"conf.yml") #Experiments/checkpoint/ver1/epoch=109.ckpt
    #Experiments/checkpoint/combine2/20250621_10/epoch=120.ckpt
    if args.use_overlapadd:
        eval_save_dir = (
                f"{exp_dir}/result_{args.use_overlapadd}"
            )
    else:
        eval_save_dir = (
            f"{exp_dir}/result_{os.path.basename(model_path).split('.')[0]}"
        )
    with open(conf_dir, "rb") as f:
        train_conf = yaml.safe_load(f)
    config={}
    config["train_conf"] = train_conf
    
    model_class = getattr(look2hear.models, config["train_conf"]["audionet"]["audionet_name"])
    model = model_class(
        sample_rate=config["train_conf"]["datamodule"]["data_config"]["sample_rate"],
        **config["train_conf"]["audionet"]["audionet_config"],
    )
    try:
        state_dict = torch.load(model_path, map_location='cpu')["state_dict"]  # or 'cuda'
        model.load_state_dict(state_dict)
        print("한 번에 로드 성공")  
    except :
        state_dict = torch.load(model_path, map_location='cpu')["state_dict"]
        # audio_model. prefix 제거
        converted_state_dict = {}
        prefix = "audio_model."
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]  # prefix 제거
                converted_state_dict[new_key] = v
        # 모델에 로드 (strict=False는 일부 키 mismatch 허용)
        model.load_state_dict(converted_state_dict, strict=True)
        print(set(converted_state_dict.keys()).difference(set(model.state_dict().keys())))
        print("한 번에 로드 실패, prefix 제거 후 로드 성공")    
    model.eval()
    
    args.sample_rate=16000
   # args.seq_dur=4
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    data_choice="jpop" #jpop, jacapella, choir
    test_set = MyJpopDataset(
            sample_rate=16000,
            data=data_choice
            
        )  
    
    meter = pyln.Meter(args.sample_rate)
    if args.use_overlapadd:  # Default to ola if not specified
        continuous_nnet = load_ola_func_with_args(args, model, device, meter).to(device)
    else:
        model.to(device)
    
    # Define overlap add functions
    

    ex_save_dir = f"{eval_save_dir}/examples_{args.singing_task}_{data_choice}len_24k"
    os.makedirs(ex_save_dir, exist_ok=True)
    if args.n_save_ex == -1:
        args.n_save_ex = len(test_set)

    # Randomly choose the indexes of sentences to save.
    save_idx = list(range(len(test_set)))

    series_list = []

    with torch.no_grad():
        for idx in tqdm(range(len(test_set))):
            # Forward the network on the mixture.
            mix, sources, ids = test_set[idx]

            # Apply loudness normalization
            #mix, adjusted_gain = loudnorm(mix.numpy(), -24.0, meter, eps=0.0)
            #mix = torch.as_tensor(mix, dtype=torch.float32)
            #sources = sources.numpy() * db2linear(adjusted_gain, eps=0.0)
            #sources = torch.as_tensor(sources, dtype=torch.float32)
            mix, sources = tensors_to_device([mix, sources], device=device)
            try:
                if args.use_overlapadd:
                    if mix.shape[-1]<16000*4: #
                        est_sources = model(mix.unsqueeze(0),istest=True)
                    else:
                        est_sources = continuous_nnet(mix.unsqueeze(0).unsqueeze(0))
                else:
                    est_sources = model(mix.unsqueeze(0),istest=True)
                if isinstance(est_sources, dict):
                    est_sources = est_sources["output_final"]#["second_est_speech"]
                elif isinstance(est_sources, tuple):
                    if "split2" in conf_dir or "time" in conf_dir or "combine1" in conf_dir:
                        est_sources = est_sources[1]
                    elif len(est_sources) == 2:
                        est_sources = est_sources[1]
                    elif len(est_sources) == 3 or len(est_sources) == 5:
                        est_sources = est_sources[0]
                #est_sources=resampler(est_sources)
                
                loss, reordered_sources = loss_func(
                    est_sources, sources[None], return_est=True
                )
                mix_np = mix.cpu().data.numpy() #* db2linear(-adjusted_gain, eps=0.0)
                sources_np = sources.cpu().data.numpy() #* db2linear(-adjusted_gain, eps=0.0)
                est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy() #* db2linear(-adjusted_gain, eps=0.0)
                
                utt_metrics = get_metrics(
                    mix_np,
                    sources_np,
                    est_sources_np,
                    sample_rate=args.sample_rate,
                    metrics_list=COMPUTE_METRICS,
                )
                utt_metrics["mix_path"] = ids

                series_list.append(pd.Series(utt_metrics))

                # Save some examples in a folder. Wav files and metrics as text.
                if idx in save_idx:
                    local_save_dir = f"{ex_save_dir}/ex_{idx}/"
                    os.makedirs(local_save_dir, exist_ok=True)
                    sf.write(local_save_dir + "mixture.wav", mix_np, args.sample_rate)
                    # Loop over the sources and estimates
                    for src_idx, src in enumerate(sources_np):
                        sf.write(f"{local_save_dir}/s{src_idx}.wav", src, args.sample_rate)
                    for src_idx, est_src in enumerate(est_sources_np):
                        sf.write(
                            f"{local_save_dir}/s{src_idx}_estimate.wav",
                            est_src,
                            args.sample_rate,
                        )
                    # Write local metrics to the example folder.
                    with open(local_save_dir + "metrics.json", "w") as f:
                        json.dump(utt_metrics, f, indent=0)
            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                continue
    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(f"{eval_save_dir}/all_metrics_{args.singing_task}.csv")

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()

    print("Overall metrics :")
    print(f"{args.exp_name}{args.suffix_name}")
    pprint(final_results)

    with open(f"{eval_save_dir}/final_metrics.json", "w") as f:
        json.dump(final_results, f, indent=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model test.py")
    # Added arguments
    parser.add_argument("--target", type=str, default="vocals")
    parser.add_argument(
        "--test_target",
        type=str,
        default="singing",
        choices=["speech", "singing"],
        help="choose",
    )
    parser.add_argument(
        "--singing_task",
        type=str,
        default="duet",
        help="only valid when test_target=='singing'. 'unison' or 'duet' or 'main_vs_rest', or 'n_singing'",
    )
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument(
        "--suffix_name",
        type=str,
        default="",
        help="additional folder name you want to attach on the last folder name of 'exp_name'. for example, '_online'",
    )
    parser.add_argument(
        "--model_dir", type=str, default="/path/to/results/singing_sep"
    )
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument(
        "--use_overlapadd",
        type=str,
        default=None,
        choices=[None, "ola", "ola_norm", "w2v", "w2v_chunk", "sf_chunk"],
        help="use overlapadd functions, ola, ola_norm, w2v will work with ola_window_len, ola_hop_len argugments. w2v_chunk and sf_chunk is chunk-wise processing based on VAD, so you have to specify the vad_method args. If you use sf_chunk (spectral_featrues_chunk), you also need to specify spectral_features.",
    )
    parser.add_argument(
        "--vad_method",
        type=str,
        default="spec",
        choices=["spec", "webrtc"],
        help="what method do you want to use for 'voice activity detection (vad) -- split chunks -- processing. Only valid when 'w2v_chunk' or 'sf_chunk' for args.use_overlapadd.",
    )
    parser.add_argument(
        "--spectral_features",
        type=str,
        default="mfcc",
        choices=["mfcc", "spectral_centroid"],
        help="what spectral feature do you want to use in correlation calc in speaker assignment (only valid when using sf_chunk)",
    )
    parser.add_argument(
        "--w2v_ckpt_dir",
        type=str,
        default="ckpt",
        help="only valid when use_overlapadd is 'w2v or 'w2v_chunk'.",
    )
    parser.add_argument(
        "--w2v_nth_layer_output",
        nargs="+",
        type=int,
        default=[0],
        help="wav2vec nth layer output",
    )
    parser.add_argument(
        "--ola_window_len",
        type=float,
        default=None,
        help="ola window size in [sec]",
    )
    parser.add_argument(
        "--nfft",
        default=512,
        type=int),
    
    parser.add_argument(
        "--ola_hop_len",
        type=float,
        default=None,
        help="ola hop size in [sec]",
    )
    parser.add_argument(
        "--reorder_chunks",
        type=str2bool,
        default=True,
        help="ola reorder chunks",
    )
    parser.add_argument(
        "--use_ema_model",
        type=str2bool,
        default=True,
        help="use ema model or online model? only vaind when args.ema it True (model trained with ema)",
    )

    # Original parameters of test code
    parser.add_argument(
        "--test_dir",
        type=str,
        # required=True,
        # default="/path/to/dataLibri2Mix/wav16k/max/metadata",
        # default="/path/to/dataLibri2Mix/wav24k/max/metadata",
        default="duet_svs/MedleyVox",
        help="Test directory including the csv files",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default="./testset/testset_config",
        help="Metadata for testset, only for 'main vs. rest' separation",
    )
    parser.add_argument(
        "--speech_task",
        type=str,
        # required=True,
        default="sep_clean",
        help="One of `enh_single`, `enh_both`, " "`sep_clean` or `sep_noisy`",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        # required=True,
        default="eval_results",
        help="Directory in exp_dir where the eval results" " will be stored",
    )
    parser.add_argument(
        "--exp_dir",
        default="/path/to/results/singing_sep",
        help="Experiment root. Evaluation results will saved in '(args.exp_dir)/(args.out_dir)/(args.exp_name'",
    )
    parser.add_argument(
        "--n_save_ex",
        type=int,
        default=10,
        help="Number of audio examples to save, -1 means all",
    )

    parser.add_argument(
        "--save_and_load_eval",
        type=str2bool,
        default=False,
        help="To check the output scale exploding, save and load outputs for eval.",
    )
    parser.add_argument(
        "--save_smaller_output",
        type=str2bool,
        default=False,
        help="To check the output scale exploding, save and load outputs for eval.",
    )

    # Original arguments

    args, _ = parser.parse_known_args()

    main(args)
