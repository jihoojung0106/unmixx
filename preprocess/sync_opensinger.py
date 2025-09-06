import os
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import soundfile as sf
import shutil

#from libfmp.b.b_plot import plot_signal, plot_chromagram
#from libfmp.c3.c3s2_dtw_plot import plot_matrix_with_points
from synctoolbox.dtw.core import compute_warping_path
from synctoolbox.dtw.cost import cosine_distance
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from tqdm import tqdm  # <-- 추가

Fs = 24000
N = 2048
H = 1024
feature_rate = int(Fs / H)

def warp(short_path, long_path, save_folder=""):
    audio_1, _ = librosa.load(long_path, sr=Fs)
    audio_2, _ = librosa.load(short_path, sr=Fs)

    chroma_1 = librosa.feature.chroma_stft(y=audio_1, sr=Fs, n_fft=N, hop_length=H, norm=2.0)
    chroma_2 = librosa.feature.chroma_stft(y=audio_2, sr=Fs, n_fft=N, hop_length=H, norm=2.0)
    C = cosine_distance(chroma_1, chroma_2)
    _, _, wp_full = compute_warping_path(C=C)

    wp_full = wp_full.T.astype(int)
    spec2 = librosa.stft(audio_2, n_fft=N, hop_length=H)
    warped_spec2 = spec2[:, wp_full[:, 1]]
    warped_audio2 = librosa.istft(warped_spec2, hop_length=H, length=len(audio_1))

    os.makedirs(save_folder, exist_ok=True)

    # Save original long audio
    long_basename = os.path.basename(long_path)
    long_save_path = os.path.join(save_folder, long_basename.replace('.wav', '_long.wav'))
    shutil.copy(long_path, long_save_path)

    short_basename = os.path.basename(short_path)
    short_save_path = os.path.join(save_folder, short_basename)
    shutil.copy(short_path, short_save_path)
    
    # Save warped short audio
    short_basename = os.path.splitext(os.path.basename(short_path))[0] + "_warped.wav"
    short_save_path = os.path.join(save_folder, short_basename)
    sf.write(short_save_path, warped_audio2, Fs)
   # print(f"Audio 2 warped to match Audio 1 saved in {short_save_path}")
# Main loop with tqdm
file_path = "duet_svs/24k/json/same_song/opensinger_combination.txt"
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(tqdm(lines, desc="Warping Progress")):
    key, wav1, len1, wav2, len2 = line.strip().split("|")
    save_folder = f"duet_svs/24k/OpenSinger_same_song/{key}_{i}"
    warp(wav1, wav2, save_folder)
