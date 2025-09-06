#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Recursively resample every WAV in duet_svs/jaCappella to 24 kHz and save as *_24K.wav
"""

from pathlib import Path
import torchaudio
from torchaudio.transforms import Resample
from tqdm.auto import tqdm

SRC_ROOT   = Path("duet_svs/Choir_Dataset")   # 원본 루트
TARGET_SR  = 24_000                        # 24 kHz
SUFFIX_TAG = "_24K"                        # 파일명에 붙일 태그

def resample_file(wav_path: Path):
    """하나의 WAV 파일을 24 kHz로 리샘플하여 <stem>_24K.wav 로 저장한다."""
    tgt_path = wav_path.with_stem(wav_path.stem + SUFFIX_TAG)
    if tgt_path.exists():
        return  # 이미 변환된 경우 스킵

    # 로드 (채널: [C, T])
    waveform, src_sr = torchaudio.load(wav_path)

    # 필요하면 리샘플
    if src_sr != TARGET_SR:
        resampler = Resample(src_sr, TARGET_SR)
        waveform = resampler(waveform)

    # 저장 (torchaudio는 부모 폴더가 없으면 자동 생성)
    torchaudio.save(tgt_path, waveform, TARGET_SR)

def main():
    wav_files = list(SRC_ROOT.rglob("*.wav"))
    wav_files = [f for f in wav_files if "_16k.wav" not in f.name]  # 이미 변환된 파일 제외
    if not wav_files:
        print(f"⚠️  {SRC_ROOT} 아래에 WAV 파일을 찾을 수 없습니다.")
        return

    for wav_path in tqdm(wav_files, desc="Resampling to 24 kHz"):
        try:
            resample_file(wav_path)
        except Exception as e:
            print(f"❌ {wav_path}: {e}")

if __name__ == "__main__":
    main()
