import os
import shutil
import glob

src_root = "duet_svs/004.k_multisinger/01.data/1.Training/original_data"
dst_root = "duet_svs/24k/k_multisinger"

os.makedirs(dst_root, exist_ok=True)

# 모든 json 파일을 찾음 (재귀적으로)
json_files = glob.glob(os.path.join(src_root, "**", "*_unison.json"), recursive=True)

for src_path in json_files:
    filename = os.path.basename(src_path)  # 중복 방지를 원하면 이 부분을 수정
    dst_path = os.path.join(dst_root, filename)
    
    # 파일명이 겹칠 경우 고유한 이름으로 저장
    if os.path.exists(dst_path):
        base, ext = os.path.splitext(filename)
        i = 1
        while os.path.exists(os.path.join(dst_root, f"{base}_{i}{ext}")):
            i += 1
        dst_path = os.path.join(dst_root, f"{base}_{i}{ext}")
    
    shutil.copy2(src_path, dst_path)
    print(f"✅ {src_path} → {dst_path}")
