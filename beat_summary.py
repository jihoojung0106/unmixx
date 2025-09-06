import os
import glob
import numpy as np
import csv
from tqdm import tqdm
# 탐색할 폴더
root_dir = "duet_svs/24k"

# 저장할 CSV 경로
output_csv = "duet_svs/24k/beat_info_summary.csv"

# 결과 저장용 리스트
results = []

# .beats 파일 찾기
beat_files = glob.glob(os.path.join(root_dir, "**", "*.beats"), recursive=True)
for beat_file in tqdm (beat_files):
    try:
        data = np.loadtxt(beat_file)
        
        if data.ndim == 1:  # 한 줄만 있는 경우 대비
            data = np.expand_dims(data, axis=0)
        
        # 첫 번째 컬럼 차이로 median 계산
        time_col = data[:, 0]
        if len(time_col) < 2:
            median_diff = None
        else:
            median_diff = np.median(np.diff(time_col))
        # 두 번째 컬럼이 1인 첫 번째 컬럼 값 추출
        beat_ones = time_col[data[:, 1] == 1].tolist()
        results.append([beat_file, median_diff, beat_ones])
    except Exception as e:
        print(f"⚠️ {beat_file} 처리 중 오류 발생: {e}")
        

# CSV로 저장
with open(output_csv, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["path", "median_diff", "onset_1_times"])
    for row in results:
        writer.writerow(row)

print(f"✅ 저장 완료: {output_csv}")
