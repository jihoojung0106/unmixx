import pandas as pd

def compute_metrics_average(csv_path: str):
    """
    주어진 CSV에서 HeladoNegro_MitadDelMundo가 포함된 행을 제외하고
    (sdr - input_sdr), (si_sdr - input_si_sdr), ssnr_whole, ssnr
    의 평균을 계산한다.

    Parameters:
        csv_path (str): CSV 파일 경로

    Returns:
        dict: 각 metric의 평균값 (소수점 둘째 자리까지)
    """
    # CSV 로드
    df = pd.read_csv(csv_path)

    # 제외할 행 필터링 (copy()로 경고 방지)
    filtered_df = df[~df['mix_path'].str.contains("HeladoNegro_MitadDelMundo", na=False)].copy()
    print(f"Total rows: {len(df)} | Filtered rows: {len(filtered_df)}")

    # 새로운 컬럼 계산
    filtered_df["sdr-input_sdr"] = filtered_df["sdr"] - filtered_df["input_sdr"]
    filtered_df["si_sdr-input_si_sdr"] = filtered_df["si_sdr"] - filtered_df["input_si_sdr"]

    # 평균 계산할 컬럼 (순서 고정)
    columns_to_average = ["sdr-input_sdr", "si_sdr-input_si_sdr", "ssnr_whole", "ssnr"]

    # 평균 계산
    averages = filtered_df[columns_to_average].mean()

    # dict로 반환 (소수점 둘째 자리까지 포맷)
    return {col: f"{averages[col]:.2f}" for col in columns_to_average}


# 사용 예시
if __name__ == "__main__":
    file_path = "duet_svs/MedleyVox_cross_50/all_metrics_unison.csv"
    result = compute_metrics_average(file_path)
    print("평균값:")
    for k, v in result.items():
        print(f"  {k}: {v}")
