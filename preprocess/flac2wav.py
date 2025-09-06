from pathlib import Path
import soundfile as sf
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def convert_flac_to_wav(flac_path):
    flac_path = Path(flac_path)
    wav_path = flac_path.with_suffix(".wav")
    if wav_path.exists():
        return  # 이미 존재하면 스킵
    try:
        data, sr = sf.read(flac_path)
        sf.write(wav_path, data, sr)
    except Exception as e:
        print(f"❌ Error with {flac_path}: {e}")

def batch_convert_flac_to_wav(directory):
    flac_files = list(Path(directory).rglob("*.flac"))
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(convert_flac_to_wav, flac_files), total=len(flac_files), desc="Converting FLAC → WAV"))

# 실행
batch_convert_flac_to_wav("duet_svs/24k/LibriSpeech_train-clean-360")
