import os
import json
import librosa
import soundfile as sf
import torchaudio
import glob
def load_audio(audio_path, sr=24000):
    wav, _ = librosa.load(audio_path, sr=sr, mono=True)
    return wav
def generate_random_string(length=4):
    import random
    import string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
def save_audio(wav, path, sr=24000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, wav, sr)

def extract_segment(wav, start_sec, end_sec, sr=24000):
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    return wav[start_sample:end_sample]

def mix_wavs(wav1, wav2):
    min_len = min(len(wav1), len(wav2))
    return 0.5 * (wav1[:min_len] + wav2[:min_len])

# ----------- 메인 루프 -----------

json_files = glob.glob("duet_svs/nus/nus-smc-corpus_48/**/*_unison.json", recursive=True)

cnt=0
for json_path in json_files:
    json_path="SINGER_16_10TO29_CLEAR_FEMALE_BALLAD_C0632_unison.json"
    #json_path="duet_svs/177.k_multitimbre_guide_vocal/01.data/1.Training/original_data/FEMALE/UNDER10/CHILDREN/NORMAL/SINGER_26/SINGER_26_UNDER10_NORMAL_FEMALE_CHILDREN_C1083_unison.json"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 오디오 경로 추정
        audio_path=json_path.replace("_unison.json", ".wav")
        #audio_path = 'duet_svs/CSD/english/wav/en001a.wav'
        if not os.path.exists(audio_path):
            print(f"❌ 오디오 없음: {audio_path}")
            continue

        wav = load_audio(audio_path)
        sr = 24000  # sample rate
        out_mix_folder="result/unison_my/multitimbre/"+generate_random_string(4)
        os.makedirs(out_mix_folder, exist_ok=True)    
            
        segment_dict = data.get("segment", {})
        for group_id, segments in segment_dict.items():
            for i, seg in enumerate(segments):
                seg_wav = extract_segment(wav, seg["start_time_sec"], seg["end_time_sec"], sr)
                out_path = os.path.join(out_mix_folder, f"{group_id}_{i}_seg.wav")
                save_audio(seg_wav, out_path, sr)
                if i==0:
                    seg_wav1= seg_wav
                elif i==1:
                    seg_wav2= seg_wav    
                # mix 두 개
            out_mix_path = os.path.join(out_mix_folder, f"{group_id}_mix.wav")
            mixed = mix_wavs(seg_wav1, seg_wav2)
            save_audio(mixed, out_mix_path, sr)
            print(f"✅ Mix 저장 완료: {out_mix_path}")
        #print(f"✅ Segment 저장 완료: {json_path}")
       #break
    except Exception as e:
        print(f"❌ 오류 발생: {json_path} - {e}")