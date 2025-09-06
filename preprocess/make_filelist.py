import os
import glob

# 목록을 추가할 대상 텍스트 파일 경로
txt_path = "all_filelist.txt"  # ← 여기를 수정하세요!

# 파일 목록을 추가할 디렉토리 경로들
dirs_to_add = [
    "duet_svs/datasets--imprt--idol-songs-jp/snapshots/c026e76507d574b4f79efb0f01e41fb1b421b563/vocals_16k"
    # "duet_svs/Choir_Dataset",
    # "duet_svs/16k/jsut-song_ver1",
    # "duet_svs/16k/k_multisinger",
    # "duet_svs/16k/k_multitimbre",
    # "duet_svs/16k/jvs_music_ver1",
    # "duet_svs/16k/MIR-1K","duet_svs/16k/musdb_a_train",
    # "duet_svs/16k/NUS","duet_svs/VocalSet",
    
]

# 수집된 파일 경로 저장용 리스트
all_files = []

# 각 디렉토리 순회하며 파일 경로 수집
for d in dirs_to_add:
    file_list = glob.glob(os.path.join(d, "**/*.wav"), recursive=True)
    file_list = [os.path.normpath(f) for f in file_list]  # 경로 정리 (윈도우 호환)
    all_files.extend(file_list)

# 중복 방지 (선택적)
all_files = sorted(set(all_files))

# .txt 파일에 경로 추가
with open(txt_path, "a") as f:
    for path in all_files:
        f.write(path + "\n")

print(f"{len(all_files)}개 경로가 '{txt_path}'에 추가되었습니다.")
