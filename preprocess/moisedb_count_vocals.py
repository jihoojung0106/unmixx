import os
import glob
import json
import shutil
from tqdm import tqdm
# 원본 JSON 경로들
json_paths = glob.glob("duet_svs/moisesdb/moisesdb/moisesdb_v0.1/*/data.json")

# 대상 복사 루트
target_root = "duet_svs/moisesdb_filtered"

for json_path in tqdm(json_paths):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # vocals stem의 track 개수 확인
        vocal_track_count = 0
        for stem in data.get("stems", []):
            if stem.get("stemName", "").lower() == "vocals":
                vocal_track_count = len(stem.get("tracks", []))
                break

        # 조건 만족 시 복사
        if vocal_track_count >= 2:
            parent_dir = os.path.dirname(json_path)                      # 예: .../moisesdb_v0.1/00001
            folder_name = os.path.basename(parent_dir)                  # 예: 00001

            # vocals 복사
            vocals_src = os.path.join(parent_dir, "vocals")
            vocals_dst = os.path.join(target_root, folder_name, "vocals")
            if os.path.exists(vocals_src):
                os.makedirs(os.path.dirname(vocals_dst), exist_ok=True)
                shutil.copytree(vocals_src, vocals_dst, dirs_exist_ok=True)
                print(f"Copied: {vocals_src} -> {vocals_dst}")
            else:
                print(f"Skip (no vocals folder): {vocals_src}")

            # data.json 복사
            json_dst = os.path.join(target_root, folder_name, "data.json")
            shutil.copy2(json_path, json_dst)
            print(f"Copied: {json_path} -> {json_dst}")

    except Exception as e:
        print(f"Error processing {json_path}: {e}")
