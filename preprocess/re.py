#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
16 kHz → 24 kHz 리샘플링 스크립트
duet_svs/16k/musdb_a_train  →  duet_svs/24k/musdb_a_train
"""

from pathlib import Path
import torchaudio
from torchaudio.transforms import Resample
from tqdm.auto import tqdm

SRC_ROOT   = Path("duet_svs/16k/musdb_a_test")   # 원본 16 kHz 루트
DST_ROOT   = Path("duet_svs/24k/musdb_a_test")   # 저장할 24 kHz 루트
TARGET_SR  = 24_000

# 한 번만 생성해 두고 재사용 (성능 ↑)
_resampler_cache = {}

def get_resampler(orig_sr):
    """필요한 Resample 객체를 캐싱해서 반환"""
    if orig_sr not in _resampler_cache:
        _resampler_cache[orig_sr] = Resample(orig_sr, TARGET_SR)
    return _resampler_cache[orig_sr]

def resample_and_save(wav_path: Path):
    """단일 파일 리샘플링 후 저장"""
    # 대응하는 출력 경로 계산
    rel_path = wav_path.relative_to(SRC_ROOT)
    out_path = DST_ROOT / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 이미 존재하면 스킵 (선택사항)
    if out_path.exists():
        return

    # 로드
    waveform, sr = torchaudio.load(wav_path)

    # 리샘플
    if sr != TARGET_SR:
        waveform = get_resampler(sr)(waveform)

    # 저장
    torchaudio.save(out_path, waveform, TARGET_SR)

def main():
    wav_files = list(SRC_ROOT.rglob("*.wav"))
    if not wav_files:
        print(f"⚠️  {SRC_ROOT} 아래에 WAV 파일이 없습니다.")
        return

    for wav in tqdm(wav_files, desc="Resampling to 24 kHz"):
        try:
            resample_and_save(wav)
        except Exception as e:
            print(f"❌ {wav}: {e}")

    print("✅ 모든 파일 변환 완료!")

if __name__ == "__main__":
    main()
