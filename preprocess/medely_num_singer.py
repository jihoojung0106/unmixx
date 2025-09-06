import json
import glob
import pandas as pd
from pathlib import Path

def build_csv_from_json(json_dir, output_csv,sel_type="duet"):
    rows = []
    for json_file in glob.glob(f"{json_dir}/**/*.json",recursive=True):
        with open(json_file, "r") as f:
            data = json.load(f)

        for seg_key, seg_info in data.items():
            song_name = seg_info["song_name"]
            seg_type = seg_info["type"]
            if "main" in seg_type:
                continue
            if sel_type not in seg_type:
                continue
            num_singers = seg_info["num_of_singers"]
            num_voices = seg_info["num_of_voices"]

            # wav_path 생성
            wav_path = f"duet_svs/MedleyVox/{seg_type}/{song_name}/{seg_key}/mix/{song_name} - {seg_key}.wav"

            rows.append({
                "wav_path": wav_path,
                "type": seg_type,
                "num_of_singers": num_singers,
                "num_of_voices": num_voices
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv+f"/{sel_type}_num_singer.csv", index=False)
    print(output_csv+f"/{sel_type}_num_singer.csv")

# 사용법 예시
build_csv_from_json("preprocess/meldey", "duet_svs/MedleyVox",sel_type="duet")
