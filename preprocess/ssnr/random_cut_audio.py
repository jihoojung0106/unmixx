from pydub import AudioSegment
import random
import os
import shutil
import glob
import csv
def cross_split_audio(audio_path1, audio_path2, random_seed=None):
    """
    동일한 길이의 두 오디오에서 앞 10%, 뒤 10%를 제외한 구간 중 한 포인트에서 분할,
    교차하여 두 개의 새로운 오디오 파일로 저장하는 함수.

    Parameters:
        audio_path1 (str): 첫 번째 오디오 파일 경로
        audio_path2 (str): 두 번째 오디오 파일 경로
        output_path1 (str): 저장할 결과 오디오1 경로
        output_path2 (str): 저장할 결과 오디오2 경로
        random_seed (int, optional): 랜덤 시드(재현 목적)
    """
    if random_seed is not None:
        random.seed(random_seed)
        
    audio1 = AudioSegment.from_file(audio_path1)
    audio2 = AudioSegment.from_file(audio_path2)
    
    duration_ms = len(audio1)
    if len(audio2) != duration_ms:
        raise ValueError("두 오디오의 길이가 같아야 합니다.")
    
    ten_percent = int(duration_ms * 0.1)
    start_range = ten_percent
    end_range = duration_ms - ten_percent
    
    split_point = random.randint(start_range, end_range)
    
    audio1_1 = audio1[:split_point]
    audio1_2 = audio1[split_point:]
    audio2_1 = audio2[:split_point]
    audio2_2 = audio2[split_point:]
    
    new_audio1 = audio1_1 + audio2_2
    new_audio2 = audio2_1 + audio1_2
    output_path1=audio_path1.replace(".wav", "_cross.wav").replace("duet_svs/MedleyVox", "duet_svs/MedleyVox_cross")
    output_path2=audio_path2.replace(".wav", "_cross.wav").replace("duet_svs/MedleyVox", "duet_svs/MedleyVox_cross")
    os.makedirs(os.path.dirname(output_path1), exist_ok=True)
    os.makedirs(os.path.dirname(output_path2), exist_ok=True)
    new_audio1.export(output_path1, format="wav")
    new_audio2.export(output_path2, format="wav")
    return split_point   # 어디서 잘랐는지 반환하면 추적에 편리
def copy_original_audios(audio_path1, audio_path2):
    """
    원본 오디오를 별도 폴더로 복사 저장 (duet_svs/MedleyVox_original)
    """
    orig_path1 = audio_path1.replace("duet_svs/MedleyVox", "duet_svs/MedleyVox_cross")
    orig_path2 = audio_path2.replace("duet_svs/MedleyVox", "duet_svs/MedleyVox_cross")
    
    os.makedirs(os.path.dirname(orig_path1), exist_ok=True)
    os.makedirs(os.path.dirname(orig_path2), exist_ok=True)
    
    shutil.copy2(audio_path1, orig_path1)
    shutil.copy2(audio_path2, orig_path2)
    print(f"Original audios copied to:\n  {orig_path1}\n  {orig_path2}")

# 예시 사용
# cross_split_audio("audio1.wav", "audio2.wav", "new_audio1.wav", "new_audio2.wav")
def process_all_gt_wavs(base_path,csv_path='duet_svs/MedleyVox_cross/unison_split_points.csv'):
    csv_rows = []
    # base_path 예: "duet_svs/MedleyVox/unison/"
    for root, dirs, files in os.walk(base_path):
        if os.path.basename(root) == "gt":
            wav_files = glob.glob(os.path.join(root, "*.wav"))
            if len(wav_files) == 2:
                audio_path1, audio_path2 = wav_files
                #copy_original_audios(audio_path1, audio_path2)
                split_pos = cross_split_audio(audio_path1, audio_path2)
                split_sec = round(split_pos / 1000, 3)
                csv_rows.append({
                    "audio1": audio_path1,
                    "audio2": audio_path2,
                    "split_point_ms": split_pos,
                    "split_point_sec": split_sec
                })
            else:
                print(f"{root} 폴더에 wav 파일 2개가 아닙니다. 무시합니다.")
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["audio1", "audio2", "split_point_ms", "split_point_sec"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV file saved at {csv_path}")
# 사용 예시
base_dir = "duet_svs/MedleyVox/unison/"
process_all_gt_wavs(base_dir)