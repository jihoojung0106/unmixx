# Loading some modules and defining some constants used later
import time
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
from libfmp.b.b_plot import plot_signal, plot_chromagram
from libfmp.c3.c3s2_dtw_plot import plot_matrix_with_points

from synctoolbox.dtw.core import compute_warping_path
from synctoolbox.dtw.cost import cosine_distance
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
import soundfile as sf

Fs = 24000
N = 2048
H = 1024
feature_rate = int(24000 / H)

def warp(short_path,long_path,save_folder=""):
    audio_1, _ = librosa.load(long_path, sr=Fs)
    audio_2, sr = librosa.load(short_path, sr=Fs)

    chroma_1 = librosa.feature.chroma_stft(y=audio_1, sr=Fs, n_fft=N, hop_length=H, norm=2.0)
    chroma_2 = librosa.feature.chroma_stft(y=audio_2, sr=Fs, n_fft=N, hop_length=H, norm=2.0)
    C = cosine_distance(chroma_1, chroma_2)
    _, _, wp_full = compute_warping_path(C=C)
    
    n_fft = N
    hop_length = H
    wp_full=wp_full.T
    wp_full = wp_full.astype(int)  # <- 이 줄 추가
    spec2 = librosa.stft(audio_2, n_fft=n_fft, hop_length=hop_length)
    warped_spec2 = spec2[:, wp_full[:, 1]]
    warped_audio2 = librosa.istft(warped_spec2, hop_length=hop_length, length=len(audio_1))
    sf.write("audio1_warped_to_audio1.wav", warped_audio2, 24000)
    print(f"Audio 2 warped to match Audio 1 saved as 'audio1_warped_to_audio1.wav' with length {len(warped_audio2)} samples.")
#warp("WomanRaw_8_123木头人_0.wav", "ManRaw_25_123木头人_0.wav")