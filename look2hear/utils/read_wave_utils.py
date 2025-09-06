import random

import numpy as np
import librosa
import torchaudio

def load_wav_from_start_mono(
    filename: str,
    target_sr: int = 16_000,   # 무조건 16 kHz 로 리샘플
):
    """
    Always read `seq_duration`-second mono audio from the *beginning* of `filename`,
    resampling to 16 kHz.  If the file is shorter than `seq_duration`, pad with zeros
    (left/right split at a random point, as in the original).

    Args
    ----
    filename     : path to .wav (or any format that librosa can read)
    seq_duration : desired length in **seconds**
    target_sr    : (fixed) target sampling rate, **always 16 000**

    Returns
    -------
    1-D NumPy array, shape = (`seq_duration` × 16 000,)
    """
    # 원하는 샘플 수 계산
    # = int(seq_duration * target_sr)

    # 0 초에서 시작해 원하는 길이만큼 읽어오고, 16 kHz 로 리샘플
    audio, _ = librosa.load(
        filename,
        sr=target_sr,        # ← 16 kHz 고정
        mono=True,
        offset=0.0,
        #duration=seq_duration
    )
    
    return audio
def load_wav_arbitrary_position_mono(filename, sample_rate, seq_duration):
    # mono
    # seq_duration[second]
    length = torchaudio.info(filename).num_frames

    read_length = librosa.time_to_samples(seq_duration, sr=sample_rate)
    if length > read_length:
        random_start = random.randint(0, int(length - read_length - 1)) / sample_rate #sec
        X, sr = librosa.load(
            filename, sr=None, offset=random_start, duration=seq_duration
        )
    else:
        random_start = 0
        total_pad_length = read_length - length
        X, sr = librosa.load(filename, sr=None, offset=0, duration=seq_duration)
        pad_left = random.randint(0, total_pad_length)
        X = np.pad(X, (pad_left, total_pad_length - pad_left))

    return X

import torchaudio
import librosa
import numpy as np
import pandas as pd
import random

def load_wav_downbeat_position_mono(filename, sample_rate, seq_duration):
    length = torchaudio.info(filename).num_frames
    read_length = librosa.time_to_samples(seq_duration, sr=sample_rate)
    down_beat_filename = filename.replace(".wav", ".beats")

    try:
        # beats 파일 읽기
        df = pd.read_csv(down_beat_filename, sep=r"\s+", header=None)
        filtered = df[df[1] == 1]
        if not filtered.empty:
            selected_time = float(filtered.sample(n=1).iloc[0, 0])
        else:
            selected_time = float(df.sample(n=1).iloc[0, 0])

        # 시점이 너무 뒤일 경우 보정
        max_start_time = max(0, (length - read_length) / sample_rate)
        selected_time = min(selected_time, max_start_time)
    except Exception as e:
        print(f"[⚠️] beats 파일 처리 실패: {e}")
        selected_time = random.randint(0, int(length - read_length - 1)) / sample_rate if length > read_length else 0

    # 오디오 로딩
    if length > read_length:
        X, sr = librosa.load(filename, sr=None, offset=selected_time, duration=seq_duration)
    else:
        total_pad_length = read_length - length
        X, sr = librosa.load(filename, sr=None, offset=0, duration=seq_duration)
        pad_left = random.randint(0, total_pad_length)
        X = np.pad(X, (pad_left, total_pad_length - pad_left))

    return X



def load_wav_specific_position_mono(
    filename, sample_rate, seq_duration, start_position
):
    # mono
    # seq_duration[second]
    # start_position[second]
    length = torchaudio.info(filename).num_frames
    read_length = librosa.time_to_samples(seq_duration, sr=sample_rate)

    start_pos_sec = max(
        start_position, 0
    )  # if start_position is minus, then start from 0.
    start_pos_sample = librosa.time_to_samples(start_pos_sec, sr=sample_rate)

    if (
        length <= start_pos_sample
    ):  # if start position exceeds audio length, then start from 0.
        start_pos_sec = 0
        start_pos_sample = 0
    X, sr = librosa.load(filename, sr=None, offset=start_pos_sec, duration=seq_duration)

    if length < start_pos_sample + read_length:
        X = np.pad(X, (0, (start_pos_sample + read_length) - length))

    return X

