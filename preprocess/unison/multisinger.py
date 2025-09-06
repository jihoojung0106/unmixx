import pandas as pd
import json
from collections import defaultdict
import os
import glob
import json
import re
import glob
import os
from collections import defaultdict
import os
import json
import librosa
import soundfile as sf
import torchaudio,torch
from tqdm import tqdm
def compute_mag_phase_mse(wav1, wav2, sr=24000, n_fft=960, hop_length=240):
    min_len = min(len(wav1), len(wav2))
    wav1 = wav1[:min_len]
    wav2 = wav2[:min_len]
    def stft(wav):
        wav_tensor = torch.tensor(wav).float()
        spec = torch.stft(wav_tensor, n_fft=n_fft, hop_length=hop_length,
                          window=torch.hann_window(n_fft), return_complex=True)
        return spec

    spec1 = stft(wav1)
    spec2 = stft(wav2)
    # Magnitude and Phase
    mag1 = spec1.abs()
    mag2 = spec2.abs()
    # phase1 = torch.angle(spec1)
    # phase2 = torch.angle(spec2)

    # MSE
    mag_mse = torch.mean((mag1 - mag2) ** 2).item()
    return mag_mse
def load_audio(audio_path, sr=24000):
    wav, _ = librosa.load(audio_path, sr=sr, mono=True)
    return wav

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

txt_paths= glob.glob("duet_svs/004.k_multisinger/01.data/1.Training/original_data/**/*.json",recursive=True)
txt_paths=[x for x in txt_paths if "unison.json" not in x]

def compute_mag_phase_mse(wav1, wav2, sr=24000, n_fft=960, hop_length=240):
    min_len = min(len(wav1), len(wav2))
    wav1 = wav1[:min_len]
    wav2 = wav2[:min_len]
    def stft(wav):
        wav_tensor = torch.tensor(wav).float()
        spec = torch.stft(wav_tensor, n_fft=n_fft, hop_length=hop_length,
                          window=torch.hann_window(n_fft), return_complex=True)
        return spec

    spec1 = stft(wav1)
    spec2 = stft(wav2)
    # Magnitude and Phase
    mag1 = spec1.abs()
    mag2 = spec2.abs()
    # phase1 = torch.angle(spec1)
    # phase2 = torch.angle(spec2)

    # MSE
    mag_mse = torch.mean((mag1 - mag2) ** 2).item()
    return mag_mse
def load_audio(audio_path, sr=24000):
    wav, _ = librosa.load(audio_path, sr=sr, mono=True)
    return wav

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

def extract_syllables_and_timings(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    word_segments = data.get("word_segments", [])
    data_list = []
    for seg in word_segments:
        word = seg.get("word", "").strip().lower()
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        if word not in ["sil", "sp"]:  # silence 제거
            data_list.append([float(start), float(end), word])

    # DataFrame 생성
    df = pd.DataFrame(data_list, columns=["start_time_sec", "end_time_sec", "lyric"])

    # 리스트 추출
    syllables = df["lyric"].tolist()
    timings = list(zip(df["start_time_sec"], df["end_time_sec"]))

    return df, syllables, timings


for txt_path in tqdm(txt_paths):
    try:
        #txt_path="ba_05688_-4_a_s02_m_02.json"
        # ----------- 2. CSV 로딩 및 전처리 -----------
        df, syllables, timings=extract_syllables_and_timings(txt_path)
        import pandas as pd
        # ----------- 3. 반복 구간 탐지 -----------
        def find_all_repeated_sequences(tokens, timings, min_len=3):
            seen = defaultdict(list)
            max_len = len(tokens)
            all_repeats = []
            used_ranges = []  # (start, end) index 범위 저장

            for n in reversed(range(min_len, max_len)):
                seen.clear()
                for i in range(max_len - n + 1):
                    ngram = tuple(tokens[i:i + n])
                    seen[ngram].append(i)
                for ngram, positions in seen.items():
                    if len(positions) > 1:
                        for pos in positions:
                            pos_range = (pos, pos + len(ngram) - 1)

                            # 겹치는지 확인
                            if any(not (pos_range[1] < used[0] or pos_range[0] > used[1]) for used in used_ranges):
                                continue  # 겹치면 무시

                            # 유효하다면 추가
                            all_repeats.append((ngram, pos))
                            used_ranges.append(pos_range)
            return all_repeats

        # ----------- 4. longest JSON 생성 -----------
        def build_repeat_group_json(results):
            group_dict = defaultdict(list)
            group_id_map = {}
            group_id = 0

            for item in results:
                key = item["lyric"]
                if key not in group_id_map:
                    group_id_map[key] = str(group_id)
                    group_id += 1
                group_key = group_id_map[key]
                group_dict[group_key].append(item)

            # 길이가 2개 이상인 그룹만 유지하고, 길이 3초 미만인 항목 제거
            filtered_group_dict = {}
            for group_key, group_items in group_dict.items():
                group_items = [item for item in group_items if item["length"] >= 3.0]
                if len(group_items) >= 2:
                    filtered_group_dict[group_key] = group_items

            return filtered_group_dict

        # ----------- 5. segment 생성 -----------
        def segment_lyrics(grouped_data):
            segments = {}
            segment_index = 0  # 인덱스 기반 key

            for group_id, items in grouped_data.items():
                lyrics = [item["lyric"] for item in items]
                if len(set(lyrics)) != 1:
                    continue
                base_group = min(items, key=lambda x: x["start_time_sec"])
                base_lyric = base_group["lyric"].split()
                base_start = base_group["start_time_sec"]
                base_end = base_group["end_time_sec"]
                base_duration = base_end - base_start
                per_unit_duration = base_duration / len(base_lyric)

                for i in range(len(base_lyric)):
                    for j in range(i + 1, len(base_lyric) + 1):
                        segment_text = " ".join(base_lyric[i:j])
                        duration = per_unit_duration * (j - i)
                        if duration < 4.0:
                            continue
                        seg_start = base_start + per_unit_duration * i
                        seg_end = base_start + per_unit_duration * j

                        instance_list = []
                        for item in items:
                            offset = item["start_time_sec"]
                            duration_full = item["end_time_sec"] - offset
                            duration_unit = duration_full / len(item["lyric"].split())
                            s = offset + duration_unit * i
                            e = offset + duration_unit * j
                            if e - s < 0.1:
                                continue
                            instance_list.append({
                                "lyric": segment_text,
                                "start_time_sec": round(s, 4),
                                "end_time_sec": round(e, 4),
                                "length": round(e - s, 4)
                            })

                        if len(instance_list) >= 2:
                            segments[str(segment_index)] = instance_list
                            segment_index += 1
                        break  # 첫 번째 유효한 segment만 저장
            return segments
        # ----------- 6. 처리 및 저장 -----------
        repeated = find_all_repeated_sequences(syllables, timings, min_len=4)

        results = []
        if repeated:
            #ngram_len = len(repeated[0][0])
            for ngram, idx in repeated:
                ngram_len = len(ngram) 
                start_time = timings[idx][0]
                end_time = timings[idx + ngram_len - 1][1]
                results.append({
                    "lyric": " ".join(ngram),
                    "start_time_sec": round(start_time, 4),
                    "end_time_sec": round(end_time, 4),
                    "length": round(end_time - start_time, 4)
                })

        longest = build_repeat_group_json(results)
        segment = segment_lyrics(longest)
        output_json_path=txt_path.replace(".json", "_unison.json")
        audio = load_audio(txt_path.replace(".json", ".wav"))
        filtered_segments = {}
        segment_index = 0
        for group_id, segments_in_group_list in segment.items():
            if len(segments_in_group_list) < 2:
                continue
            seg_infos = [seg_info for seg_info in segments_in_group_list if "lyric" in seg_info]
            start_times = [seg_info["start_time_sec"] for seg_info in seg_infos]
            end_times = [seg_info["end_time_sec"] for seg_info in seg_infos]
            segs=[extract_segment(audio, start, end) for start, end in zip(start_times, end_times)]
            segs=[seg for seg in segs if len(seg) >= 2400]  # 최소 길이 2400 샘플
            mag_mse = compute_mag_phase_mse(segs[0], segs[1])
            #print(mag_mse)
            if mag_mse >= 0.35 and mag_mse<=5.0:
                for seg_info in seg_infos:
                    seg_info["mag_mse"]=round(mag_mse,4)
                filtered_segments[str(segment_index)] = seg_infos
                segment_index += 1

        if len(filtered_segments)==0:
            if os.path.exists(output_json_path):
                os.remove(output_json_path)
                print(f"🗑️ 기존 파일 삭제됨: {output_json_path}")
            else:
                print(f"⚠️ segment 없음, 저장하지 않음: {output_json_path}")
            continue
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump({"longest": longest, "segment": filtered_segments}, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON 저장 완료: {output_json_path}")
    except Exception as e:
        print(f"❌ 오류 발생: {txt_path} - {e}")    
        continue   