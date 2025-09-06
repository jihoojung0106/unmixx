#!/usr/bin/env python
"""
duet_svs/best_pop_song/align 이하 CSV 들을 읽어
  └ 파일명 '<CoverA>_vs_<CoverB>.csv' 에서 CoverA / CoverB 를 Key 로 삼고
  └ 각 Key 별 Cover_?_Timestamp 구간을 중복 없이 병합해 총 길이(sec)를 구함
결과: cover_timestamp_lengths.json | cover_timestamp_lengths.csv 저장

필요 패키지: pip install pandas tqdm
"""

import json, csv, glob
from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# ── 설정 ────────────────────────────────────────────────────────────────────
ROOT_DIR  = Path("/home/jungji/real_sep/tiger_ver4/duet_svs/best_pop_song/align")
OUT_JSON  = "cover_timestamp_lengths.json"
OUT_CSV   = "cover_timestamp_lengths.csv"
# ───────────────────────────────────────────────────────────────────────────

def parse_interval(s: str):
    """'(start,end)' 또는 '[start, end]' → tuple(float, float)"""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s and s[0] in "([{":
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        return None
    try:
        a, b = map(float, parts)
        return (min(a, b), max(a, b))
    except ValueError:
        return None

def merge_intervals(intervals):
    """겹치는 구간 병합"""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged

def total_length(intervals):
    return sum(e - s for s, e in intervals)

def main():
    csv_files = glob.glob(str(ROOT_DIR / "**/*.csv"), recursive=True)
    if not csv_files:
        print(f"⚠️  {ROOT_DIR} 안에 CSV 가 없습니다.")
        return

    # Key → interval 목록
    intervals_dict = defaultdict(list)
    for fp in tqdm(csv_files, desc="CSV 읽는 중"):
        stem = Path(fp).stem              # 예: cover__A_vs_cover__B
        if "_vs_" not in stem:
            tqdm.write(f"❌ 파일명 형식 오류: {fp}")
            continue
        key1, key2 = stem.split("_vs_", 1)

        df = pd.read_csv(fp)
        if "Cover_1_Timestamp" in df.columns:
            intervals_dict[key1].extend(
                filter(None, (parse_interval(x) for x in df["Cover_1_Timestamp"].dropna()))
            )
        if "Cover_2_Timestamp" in df.columns:
            intervals_dict[key2].extend(
                filter(None, (parse_interval(x) for x in df["Cover_2_Timestamp"].dropna()))
            )

    # Key별 병합 및 길이 계산
    result_rows = []
    summary = {}
    for key, ivals in intervals_dict.items():
        merged = merge_intervals(ivals)
        length_sec = total_length(merged)
        summary[key] = length_sec
        result_rows.append({"cover": key, "length_sec": length_sec})

    # JSON 저장
    with open(OUT_JSON, "w") as jf:
        json.dump(summary, jf, ensure_ascii=False, indent=2)

    # CSV 저장
    with open(OUT_CSV, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["cover", "length_sec"])
        writer.writeheader()
        writer.writerows(result_rows)

    # 콘솔 요약
    print(f"✅ {len(result_rows)}개 커버의 길이를 계산했습니다.")
    for row in sorted(result_rows, key=lambda x: -x["length_sec"])[:10]:
        print(f"  • {row['cover']}: {row['length_sec']:.2f}s ≈ {row['length_sec']/3600:.2f}h")
    print(f"→ {OUT_JSON}, {OUT_CSV} 저장 완료")

if __name__ == "__main__":
    main()
