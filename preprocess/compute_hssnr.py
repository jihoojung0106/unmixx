import argparse
import pandas as pd
import numpy as np

# 특정 곡에 대해서는 ssnr_whole 사용
DIFF_SONG = [
    'HeladoNegro_MitadDelMundo'
]

def compute_unison_ssnr_mean(csv_path: str) -> float:
    """
    Unison metric CSV를 입력받아,
    DIFF_SONG 목록에 포함된 곡은 ssnr_whole을,
    나머지는 ssnr을 선택해 평균 SSNR을 계산한다.
    """
    df = pd.read_csv(csv_path)
    #print(f"[Unison] Total rows: {len(df)}")

    def select_ssnr(row):
        for song in DIFF_SONG:
            if song in str(row['mix_path']):
                return row['ssnr_whole']
        return row['ssnr']

    df['ssnr_selected'] = df.apply(select_ssnr, axis=1)
    ssnr_mean = df['ssnr_selected'].mean()

    #print(f"[Unison] Final Mean SSNR: {ssnr_mean:.4f}")
    return ssnr_mean


def compute_ssnr_mean(metric_csv_path: str) -> float:
    """
    Duet metrics CSV를 입력받아,
    num_of_singers 기준으로 ssnr / ssnr_whole을 선택해 평균 SSNR을 계산한다.
    """
    metrics_df = pd.read_csv(metric_csv_path)
    num_singers_df = pd.read_csv("duet_svs/MedleyVox/duet_num_singer.csv")

    merged_df = metrics_df.merge(
        num_singers_df[["wav_path", "num_of_singers"]],
        left_on="mix_path",
        right_on="wav_path",
        how="inner"
    )
    #print(f"[Duet] Merge 후 행의 개수: {len(merged_df)}")

    merged_df["ssnr_selected"] = np.where(
        merged_df["num_of_singers"] == 1,
        merged_df["ssnr"],
        merged_df["ssnr_whole"]
    )
    ssnr_mean = merged_df["ssnr_selected"].mean()
    #print(f"[Duet] 조건별로 선택한 SSNR 평균: {ssnr_mean:.4f}")
    return ssnr_mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SSNR metrics")
    parser.add_argument("--p", type=str, 
                        default="Experiments/checkpoint/att3/20250831_11/result_best_model/all_metrics_duet.csv",
                        help="Path to the metrics CSV file")
    args = parser.parse_args()
    try:
        unison_mean=compute_unison_ssnr_mean(args.p.replace("duet", "unison"))
    except Exception as e:
        print(f"Error computing unison SSNR mean: {e}")
    try:    
        duet_mean=compute_ssnr_mean(args.p.replace("unison", "duet"))
    except Exception as e:
        print(f"Error computing duet SSNR mean: {e}")
    print(f"Duet SSNR Mean: {duet_mean:.2f}")
    print(f"Unison SSNR Mean: {unison_mean:.2f}")
