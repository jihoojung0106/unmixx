import os
import glob
import json

# 모든 data.json 경로 수집
json_paths = glob.glob("duet_svs/moisesdb_filtered/*/data.json")

# 결과 누적 리스트
vocals_info_list = []

for json_path in json_paths:
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # vocals stem 찾기
        for stem in data.get("stems", []):
            if stem.get("stemName", "").lower() == "vocals":
                # track에서 id와 trackType만 추출
                simplified_tracks = [
                    {
                        "id": track.get("id"),
                        "trackType": track.get("trackType")
                    }
                    for track in stem.get("tracks", [])
                ]
                vocals_info_list.append({
                    "path": json_path,
                    "tracks": simplified_tracks
                })
                break
    except Exception as e:
        print(f"Error reading {json_path}: {e}")

# 결과 저장
output_path = "duet_svs/moisesdb_filtered/vocals_tracks_info.json"
with open(output_path, "w") as f:
    json.dump(vocals_info_list, f, indent=2)

print(f"✅ 저장 완료: {output_path}")
