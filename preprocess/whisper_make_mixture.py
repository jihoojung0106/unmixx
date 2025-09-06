import os
import glob
import json
import pandas as pd
import torchaudio
from difflib import SequenceMatcher
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 설정
json_files = sorted(glob.glob("duet_svs/best_pop_song/lead_whisper/*/*.json"), reverse=True)
window_size = 8
stride = 1
threshold = 0.9

def process_json(json_path):
    try:
        parts = json_path.split(os.sep)
        folder = parts[-2]
        basename = os.path.basename(json_path).replace(".json", "")
        wav_path = f"duet_svs/best_pop_song/lead_sep/{folder}/{basename}.wav"
        output_dir = f"duet_svs/best_pop_song/lead_align/{folder}"
        output_csv = os.path.join(output_dir, f"{basename}.csv")

        # 이미 결과가 있다면 스킵
        if os.path.exists(output_csv):
            return f"⏩ Skipped (exists): {json_path}"

        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(wav_path):
            return f"⚠️ Missing WAV: {json_path}"

        waveform, sr = torchaudio.load(wav_path)

        with open(json_path, "r") as f:
            data = json.load(f)

        word_segments = data.get("word_segments", [])
        if len(word_segments) < window_size:
            return f"⚠️ Too few words: {json_path}"

        # 슬라이딩 윈도우
        windows = []
        for i in range(0, len(word_segments) - window_size + 1, stride):
            window_words = word_segments[i:i + window_size]
            text = " ".join([w["word"] for w in window_words])
            start = window_words[0]["start"]
            end = window_words[-1]["end"]
            windows.append({"start": start, "end": end, "text": text})

        # 유사 문장 찾기
        metadata = []
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                dur1 = windows[i]["end"] - windows[i]["start"]
                dur2 = windows[j]["end"] - windows[j]["start"]
                if min(dur1, dur2) < 2.0:
                    continue

                sim = SequenceMatcher(None, windows[i]["text"], windows[j]["text"]).ratio()
                if sim > threshold:
                    metadata.append({
                        "pair_folder": f"mix_pair_{len(metadata):03d}",
                        "start_1": windows[i]["start"],
                        "end_1": windows[i]["end"],
                        "text_1": windows[i]["text"],
                        "start_2": windows[j]["start"],
                        "end_2": windows[j]["end"],
                        "text_2": windows[j]["text"],
                        "similarity": round(sim, 4),
                    })

        if metadata:
            df = pd.DataFrame(metadata)
            df.to_csv(output_csv, index=False)
            return f"✅ Done: {json_path}"
        else:
            return f"⚠️ No pairs: {json_path}"

    except Exception as e:
        return f"❌ Error: {json_path} - {e}"

if __name__ == "__main__":
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_json, json_files), total=len(json_files)))

    for r in results:
        print(r)
