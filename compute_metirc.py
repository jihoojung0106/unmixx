
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
from look2hear.metrics import MetricsTracker
from look2hear.utils import tensors_to_device, RichProgressBarTheme, MyMetricsTextColumn, BatchesProcessedColumn#,save_state_dict_keys
#@from functions import load_ola_func_with_args#, load_w2v_func_with_args, load_w2v_chunk_func_with_args, load_spectral_features_chunk_func_with_args
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

from functions import load_ola_func_with_args
from look2hear.utils import str2bool, loudnorm, db2linear
from mir_eval.separation import _bss_decomp_mtifilt, _bss_source_crit
from look2hear.losses.ssnr_loss import compute_ssnr,compute_whole_ssnr,compute_vad_ssnr
from look2hear.utils.vad_combiner import conservative_vad_union_intervals
import yaml
#COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi"]
COMPUTE_METRICS = ["si_sdr","sdr"]
mix_np=librosa.load("sample_music/ex_102/mixture.wav", sr=24000)[0]
sources_np=[librosa.load(f"sample_music/ex_102/s{i}.wav", sr=24000)[0] for i in range(2)]
est_sources_np=[librosa.load(f"sample_music/ex_102/s{i}_rev.wav", sr=24000)[0] for i in range(2)]
ssnr=compute_ssnr(
                    sources_np[0], est_sources_np[0],
                    sources_np[1], est_sources_np[1],
                )
ssnr_whole = compute_whole_ssnr(
    sources_np[0], est_sources_np[0],
    sources_np[1], est_sources_np[1])
sources_np = np.stack([
    librosa.load(f"sample_music/ex_102/s{i}.wav", sr=24000)[0]
    for i in range(2)
])

# 복원된 소스 (2개 파일)
est_sources_np = np.stack([
    librosa.load(f"sample_music/ex_102/s{i}_rev.wav", sr=24000)[0]
    for i in range(2)
])
utt_metrics = get_metrics(
                    mix_np,
                    sources_np,
                    est_sources_np,
                    sample_rate=24000,
                    metrics_list=COMPUTE_METRICS,
                    
                )
print(utt_metrics)
print("ssnr",ssnr.mean())
print("ssnr_whole",ssnr_whole[1].mean())