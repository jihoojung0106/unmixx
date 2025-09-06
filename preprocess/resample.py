import os
import torchaudio
from torchaudio.transforms import Resample
import glob

# 원본 경로와 저장 경로
src_dir = "/home/jungji/real_sep/tiger_ver4/duet_svs/datasets--imprt--idol-songs-jp/snapshots/c026e76507d574b4f79efb0f01e41fb1b421b563/vocals_48k32b"
dst_dir = "/home/jungji/real_sep/tiger_ver4/duet_svs/datasets--imprt--idol-songs-jp/snapshots/c026e76507d574b4f79efb0f01e41fb1b421b563/vocals_24k"

# 저장 폴더 생성
os.makedirs(dst_dir, exist_ok=True)

# wav 파일 찾기
wav_paths = glob.glob(os.path.join(src_dir, "**/*.wav"), recursive=True)

# 변환 시작
for wav_path in wav_paths:
    # 오디오 로드
    waveform, sample_rate = torchaudio.load(wav_path)

    # 이미 16kHz면 스킵
    if sample_rate == 24000:
        print(f"Already 24kHz: {wav_path}")
        continue

    # 리샘플링
    resampler = Resample(orig_freq=sample_rate, new_freq=24000)
    waveform_16k = resampler(waveform)

    # 상대 경로에 맞춰 저장
    relative_path = os.path.relpath(wav_path, src_dir)
    save_path = os.path.join(dst_dir, relative_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 저장 (기본은 float32로 저장)
    torchaudio.save(save_path, waveform_16k, 24000)
    print(f"Saved: {save_path}")
