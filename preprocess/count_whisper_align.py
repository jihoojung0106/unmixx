import glob

# 모든 JSON 파일 개수
json_files = glob.glob("duet_svs/best_pop_song/lead_whisper/*/*.json")
num_json = len(json_files)

# 모든 CSV 파일 개수
csv_files = glob.glob("duet_svs/best_pop_song/lead_align/*/*.csv")
num_csv = len(csv_files)

# 출력
print(f"JSON 개수: {num_json}")
print(f"CSV 개수: {num_csv}")
print(f"차이: {num_json - num_csv}")
