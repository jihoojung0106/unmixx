import pandas as pd
import numpy as np

# -------------------------------
# 1. CSV 불러오기
# -------------------------------
metrics_df = pd.read_csv("Experiments/checkpoint/fin_all/20250813_13/best/all_metrics_duet.csv")
num_singers_df = pd.read_csv("duet_svs/MedleyVox/duet_num_singer.csv")

# -------------------------------
# 2. mix_path 와 wav_path 기준으로 merge
# -------------------------------
merged_df = metrics_df.merge(
    num_singers_df[["wav_path", "num_of_singers"]],
    left_on="mix_path",
    right_on="wav_path",
    how="inner"
)
print(f"Merge 후 행의 개수: {len(merged_df)}")

# -------------------------------
# 3. 그룹별 평균 계산
# -------------------------------
grouped = merged_df.groupby("num_of_singers").agg(
    ssnr_mean=("ssnr", "mean"),
    ssnr_whole_mean=("ssnr_whole", "mean")
).reset_index()

# -------------------------------
# 4. 결과 출력
# -------------------------------
for _, row in grouped.iterrows():
    n = row["num_of_singers"]
    print(f"num_of_singers={n}: ssnr_whole 평균={row['ssnr_whole_mean']:.4f} ssnr 평균={row['ssnr_mean']:.4f}")
