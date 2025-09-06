import os
import shutil
import glob

src_root = "duet_svs/nus/nus-smc-corpus_48"
dst_root = "duet_svs/24k/NUS"

os.makedirs(dst_root, exist_ok=True)

# 모든 json 파일을 찾음 (재귀적으로)
json_files = glob.glob(os.path.join(src_root, "**", "*.json"), recursive=True)
json_files=[x for x in json_files if "sing" in x]
for src_path in json_files:
    filename = src_path.split("/")[-3]+"_"+src_path.split("/")[-1]  # 중복 방지를 원하면 이 부분을 수정
    dst_path = os.path.join(dst_root, filename)
    #dst_path=dst_path.replace("_u")
    # 파일명이 겹칠 경우 고유한 이름으로 저장
    if os.path.exists(dst_path):
        base, ext = os.path.splitext(filename)
        i = 1
        while os.path.exists(os.path.join(dst_root, f"{base}_{i}{ext}")):
            i += 1
        dst_path = os.path.join(dst_root, f"{base}_{i}{ext}")
    
    shutil.copy2(src_path, dst_path)
    print(f"✅ {src_path} → {dst_path}")
