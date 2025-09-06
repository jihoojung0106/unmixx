import os
import glob
import csv
from speechbrain.inference.speaker import SpeakerRecognition

def verify_speaker_similarity(audio_path1, audio_path2, verification):
    """
    두 오디오 파일에 대해 verification score 계산
    """
    score, prediction = verification.verify_files(audio_path1, audio_path2)
    return score, prediction

def process_all_gt_wavs_with_verification(base_path, csv_path):
    # SpeakerRecognition 모델 로드 (한 번만)
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",run_opts={"device":"cuda"} 
    )
    
    csv_rows = []
    
    for root, dirs, files in os.walk(base_path):
        if os.path.basename(root) == "gt":
            wav_files = glob.glob(os.path.join(root, "*.wav"))
            if len(wav_files) == 2:
                audio_path1, audio_path2 = wav_files
                print(f"Verifying speaker similarity:\n  {audio_path1}\n  {audio_path2}")
                
                score, prediction = verify_speaker_similarity(audio_path1, audio_path2, verification)
                score=score.item()
                prediction=prediction.item()
                csv_rows.append({
                    "audio1": audio_path1,
                    "audio2": audio_path2,
                    "score": score,
                    "prediction": prediction
                })
                
                print(f"Score={score:.4f}, Prediction={prediction}\n")
            else:
                print(f"{root} 폴더에 wav 파일 2개가 아닙니다. 무시합니다.")
    
    # CSV 파일 저장
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["audio1", "audio2", "score", "prediction"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Speaker verification results saved to {csv_path}")


# 사용 예시
base_dir = "duet_svs/MedleyVox/unison/"
csv_output = "duet_svs/MedleyVox_cross/speaker_verification_results.csv"
process_all_gt_wavs_with_verification(base_dir, csv_output)