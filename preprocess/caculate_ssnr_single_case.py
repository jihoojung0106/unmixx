import pandas as pd

# -------------------------------
# 1. CSV 불러오기
# -------------------------------
file_path = "Experiments/checkpoint/fin_van1/20250716_23/result_best_model/all_metrics_unison.csv"
df = pd.read_csv(file_path)
print(f"Total rows: {len(df)}")

# -------------------------------
# 2. Diff song list 정의
# -------------------------------
DIFF_SONG = [
    'HeladoNegro_MitadDelMundo'
]

# -------------------------------
# 3. ssnr_selected 생성
# -------------------------------
def select_ssnr(row):
    # DIFF_SONG에 포함 → num_of_singers=2 취급 → ssnr_whole 사용
    for song in DIFF_SONG:
        if song in str(row['mix_path']):
            return row['ssnr_whole']
    # 나머지 → num_of_singers=1 취급 → ssnr 사용
    return row['ssnr']

df['ssnr_selected'] = df.apply(select_ssnr, axis=1)

# -------------------------------
# 4. 그룹별 평균 계산 (1 vs 2 취급)
# -------------------------------
df['num_of_singers'] = df['mix_path'].apply(
    lambda x: 2 if any(song in str(x) for song in DIFF_SONG) else 1
)

grouped = df.groupby("num_of_singers").agg(
    ssnr_mean=("ssnr", "mean"),
    ssnr_whole_mean=("ssnr_whole", "mean"),
    ssnr_selected_mean=("ssnr_selected", "mean")
).reset_index()

print(grouped)
