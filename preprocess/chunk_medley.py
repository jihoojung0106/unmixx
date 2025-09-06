import torchaudio
from torchaudio.transforms import Resample
from pathlib import Path
import os

SRC_DIR = Path("duet_svs/MedleyVox")             # 원본 경로
DST_DIR = Path("duet_svs/MedleyVox_24k_2sec_chunks")  # 저장 경로
TARGET_SR = 24000
CHUNK_SEC = 2
CHUNK_SAMPLES = TARGET_SR * CHUNK_SEC

resamplers = {}  # 샘플레이트 별 resampler 캐시

def get_resampler(orig_sr):
    if orig_sr not in resamplers:
        resamplers[orig_sr] = Resample(orig_sr, TARGET_SR)
    return resamplers[orig_sr]

def split_and_save(wav_path: Path):
    waveform, sr = torchaudio.load(wav_path)

    # 리샘플
    if sr != TARGET_SR:
        waveform = get_resampler(sr)(waveform)

    total_samples = waveform.shape[1]
    n_chunks = (total_samples + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES

    # 출력 경로
    rel_path = wav_path.relative_to(SRC_DIR)
    out_dir = DST_DIR / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_chunks):
        start = i * CHUNK_SAMPLES
        end = min((i + 1) * CHUNK_SAMPLES, total_samples)
        chunk = waveform[:, start:end]

        out_path = out_dir / f"{wav_path.stem}_chunk{i:03d}.wav"
        torchaudio.save(str(out_path), chunk, TARGET_SR)

def main():
    wav_files = list(SRC_DIR.rglob("*.wav"))
    print(f"🔍 총 {len(wav_files)}개 파일 처리 예정")

    for path in wav_files:
        try:
            split_and_save(path)
        except Exception as e:
            print(f"❌ {path}: {e}")

    print("✅ 완료: 24kHz 리샘플링 및 4초 단위 저장")

if __name__ == "__main__":
    main()
