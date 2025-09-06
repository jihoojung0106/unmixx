import argparse
from pydub import AudioSegment
import random
import os
import shutil
import glob
import csv

def swap_random_segment(audio_path1, audio_path2, cross_dir, random_seed=None, min_ratio=0.2, max_ratio=0.5):
    """
    동일 길이의 두 오디오에서 전체 길이의 [min_ratio, max_ratio] 사이 구간을 랜덤으로 선택해
    해당 구간을 서로 교환하여 두 개의 새 오디오를 생성/저장한다.

    Parameters:
        audio_path1 (str): 첫 번째 오디오 파일 경로
        audio_path2 (str): 두 번째 오디오 파일 경로
        cross_dir (str): 결과를 저장할 base 폴더 이름 (예: 'duet_svs/MedleyVox_cross2')
    """
    if random_seed is not None:
        random.seed(random_seed)

    a1 = AudioSegment.from_file(audio_path1)
    a2 = AudioSegment.from_file(audio_path2)

    dur = len(a1)
    if len(a2) != dur:
        raise ValueError("두 오디오의 길이가 같아야 합니다.")

    seg_len_ms = int(dur * random.uniform(min_ratio, max_ratio))
    start_ms = random.randint(0, max(0, dur - seg_len_ms))
    end_ms = start_ms + seg_len_ms

    a1_left, a1_mid, a1_right = a1[:start_ms], a1[start_ms:end_ms], a1[end_ms:]
    a2_left, a2_mid, a2_right = a2[:start_ms], a2[start_ms:end_ms], a2[end_ms:]

    new_a1 = a1_left + a2_mid + a1_right
    new_a2 = a2_left + a1_mid + a2_right

    # 경로 교체 (MedleyVox → cross_dir)
    out1 = audio_path1.replace(".wav", "_cross.wav").replace("duet_svs/MedleyVox", cross_dir)
    out2 = audio_path2.replace(".wav", "_cross.wav").replace("duet_svs/MedleyVox", cross_dir)
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    os.makedirs(os.path.dirname(out2), exist_ok=True)

    new_a1.export(out1, format="wav")
    new_a2.export(out2, format="wav")

    return {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "len_ms": seg_len_ms,
        "ratio": seg_len_ms / dur if dur > 0 else 0.0
    }

def copy_original_audios(audio_path1, audio_path2, cross_dir):
    """
    원본 오디오를 cross_dir 폴더로 복사
    """
    orig1 = audio_path1.replace("duet_svs/MedleyVox", cross_dir)
    orig2 = audio_path2.replace("duet_svs/MedleyVox", cross_dir)
    os.makedirs(os.path.dirname(orig1), exist_ok=True)
    os.makedirs(os.path.dirname(orig2), exist_ok=True)
    shutil.copy2(audio_path1, orig1)
    shutil.copy2(audio_path2, orig2)
    print(f"Original audios copied to:\n  {orig1}\n  {orig2}")

def process_all_gt_wavs(base_path, cross_dir,
                        csv_path=None, random_seed=None, min_ratio=0.2, max_ratio=0.5, copy_original=True):
    """
    base_path 하위 'gt' 폴더에서 wav 2개를 찾아 랜덤 구간을 교환,
    결과를 cross_dir 폴더에 저장하고 CSV 기록.
    """
    if csv_path is None:
        csv_path = os.path.join(cross_dir, "unison_swap_segments.csv")

    if random_seed is not None:
        random.seed(random_seed)

    rows = []
    for root, dirs, files in os.walk(base_path):
        if os.path.basename(root) == "gt":
            wavs = sorted(glob.glob(os.path.join(root, "*.wav")))
            if len(wavs) == 2:
                a1, a2 = wavs
                if copy_original:
                    copy_original_audios(a1, a2, cross_dir)
                info = swap_random_segment(a1, a2, cross_dir,
                                           random_seed=random_seed,
                                           min_ratio=min_ratio,
                                           max_ratio=max_ratio)
                rows.append({
                    "audio1": a1,
                    "audio2": a2,
                    "segment_start_ms": info["start_ms"],
                    "segment_end_ms": info["end_ms"],
                    "segment_len_ms": info["len_ms"],
                    "segment_ratio": round(info["ratio"], 4)
                })
            else:
                print(f"{root}: wav 파일이 2개가 아닙니다. (found={len(wavs)}) → 건너뜀")

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio1","audio2","segment_start_ms","segment_end_ms","segment_len_ms","segment_ratio"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved at {csv_path} (rows={len(rows)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random segment swapping for duet/unison wavs.")
    parser.add_argument("--base_path", type=str, default="duet_svs/MedleyVox/unison", help="기존 MedleyVox 데이터셋 루트 경로")
    parser.add_argument("--cross_dir", type=str, default="duet_svs/MedleyVox_cross_10", help="출력 교차 데이터셋 루트 경로 (예: duet_svs/MedleyVox_cross2)")
    parser.add_argument("--seed", type=int, default=None, help="랜덤 시드")
    parser.add_argument("--min_ratio", type=float, default=0.1, help="교환 구간 최소 비율")
    parser.add_argument("--max_ratio", type=float, default=0.1, help="교환 구간 최대 비율")
    args = parser.parse_args()

    process_all_gt_wavs(args.base_path, args.cross_dir,
                        random_seed=args.seed,
                        min_ratio=args.min_ratio,
                        max_ratio=args.max_ratio,
                        copy_original=True)
