# # Load and process the file to filter out lines where the difference between durations is 2 seconds or more
# filtered_lines = []
# input_path = "duet_svs/24k/json/same_song/opensinger_combination.txt"
# output_path = "duet_svs/24k/json/same_song/opensinger_combination_filtered.txt"

# with open(input_path, "r", encoding="utf-8") as f:
#     for line in f:
#         parts = line.strip().split("|")
#         if len(parts) != 5:
#             continue
#         try:
#             len1 = float(parts[2])
#             len2 = float(parts[4])
#             if abs(len1 - len2) < 2.0:
#                 filtered_lines.append(line.strip())
#         except ValueError:
#             continue

# # Save the filtered lines to a new file
# with open(output_path, "w", encoding="utf-8") as f:
#     for line in filtered_lines:
#         f.write(line + "\n")

# output_path
import os
import glob
import soundfile as sf
import pandas as pd
from tqdm import tqdm
base_dir = "/home/jungji/real_sep/tiger_ver5/duet_svs/24k/OpenSinger_same_song"
results = []

folders = sorted(glob.glob(os.path.join(base_dir, "*")))
cnt=0
for folder in tqdm(folders):
    wav_files = glob.glob(os.path.join(folder, "*.wav"))
    if len(wav_files) != 3:
        print(f"Skipping folder {folder} due to unexpected number of files: {len(wav_files)}")
        continue

    long_file = next((f for f in wav_files if "long" in os.path.basename(f)), None)
    warped_file = next((f for f in wav_files if "warped" in os.path.basename(f)), None)
    other_files = [f for f in wav_files if f != long_file and f != warped_file]

    if long_file and warped_file and len(other_files) == 1:
        other_file = other_files[0]
        try:
            long_info = sf.info(long_file)
            other_info = sf.info(other_file)
            time_diff = abs(long_info.duration - other_info.duration)
            if time_diff < 2.0 and time_diff > 1.0:
                results.append((long_file, warped_file))
            else:
                #print(f"Time difference too large for {folder}: {time_diff:.2f} seconds")
                cnt += 1
        except:
            continue
output_txt_path = "duet_svs/24k/json/same_song/opensinger_long_warped_pairs.txt"
print(f"Total folders skipped due to time difference: {cnt}")
# long_file|warped_file 형식으로 저장
with open(output_txt_path, "w", encoding="utf-8") as f:
    for long_file, warped_file in results:
        f.write(f"{long_file}|{warped_file}\n")

print(f"Filtered results saved to {output_txt_path}")
