import pandas as pd
import json
from collections import defaultdict
import os
import glob
txt_paths= glob.glob("duet_svs/nus/nus-smc-corpus_48/**/*.txt",recursive=True)
#txt_paths=['09.txt']
for txt_path in txt_paths:
    # ----------- 2. CSV 로딩 및 전처리 -----------
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            start, end, phoneme = parts
            if phoneme not in ["sil", "sp"]:  # silence 제거
                data.append([float(start), float(end), phoneme])
    df = pd.DataFrame(data, columns=["start_time_sec", "end_time_sec", "lyric"])
    syllables = df["lyric"].tolist()
    timings = list(zip(df["start_time_sec"], df["end_time_sec"]))
    import pandas as pd
    # ----------- 3. 반복 구간 탐지 -----------
    def find_all_repeated_sequences(tokens, timings, min_len=3):
        seen = defaultdict(list)
        max_len = len(tokens)
        all_repeats = []
        used_ranges = []  # (start, end) index 범위 저장

        for n in reversed(range(min_len, max_len)):
            seen.clear()
            for i in range(max_len - n + 1):
                ngram = tuple(tokens[i:i + n])
                seen[ngram].append(i)
            for ngram, positions in seen.items():
                if len(positions) > 1:
                    for pos in positions:
                        pos_range = (pos, pos + len(ngram) - 1)

                        # 겹치는지 확인
                        if any(not (pos_range[1] < used[0] or pos_range[0] > used[1]) for used in used_ranges):
                            continue  # 겹치면 무시

                        # 유효하다면 추가
                        all_repeats.append((ngram, pos))
                        used_ranges.append(pos_range)
        return all_repeats

    # ----------- 4. longest JSON 생성 -----------
    def build_repeat_group_json(results):
        group_dict = defaultdict(list)
        group_id_map = {}
        group_id = 0

        for item in results:
            key = item["lyric"]
            if key not in group_id_map:
                group_id_map[key] = str(group_id)
                group_id += 1
            group_key = group_id_map[key]
            group_dict[group_key].append(item)

        # 길이가 2개 이상인 그룹만 유지하고, 길이 3초 미만인 항목 제거
        filtered_group_dict = {}
        for group_key, group_items in group_dict.items():
            group_items = [item for item in group_items if item["length"] >= 3.0]
            if len(group_items) >= 2:
                filtered_group_dict[group_key] = group_items

        return filtered_group_dict

    # ----------- 5. segment 생성 -----------
    def segment_lyrics(grouped_data):
        segments = {}
        segment_index = 0  # 인덱스 기반 key

        for group_id, items in grouped_data.items():
            lyrics = [item["lyric"] for item in items]
            if len(set(lyrics)) != 1:
                continue
            base_group = min(items, key=lambda x: x["start_time_sec"])
            base_lyric = base_group["lyric"].split()
            base_start = base_group["start_time_sec"]
            base_end = base_group["end_time_sec"]
            base_duration = base_end - base_start
            per_unit_duration = base_duration / len(base_lyric)

            for i in range(len(base_lyric)):
                for j in range(i + 1, len(base_lyric) + 1):
                    segment_text = " ".join(base_lyric[i:j])
                    duration = per_unit_duration * (j - i)
                    if duration < 4.0:
                        continue
                    seg_start = base_start + per_unit_duration * i
                    seg_end = base_start + per_unit_duration * j

                    instance_list = []
                    for item in items:
                        offset = item["start_time_sec"]
                        duration_full = item["end_time_sec"] - offset
                        duration_unit = duration_full / len(item["lyric"].split())
                        s = offset + duration_unit * i
                        e = offset + duration_unit * j
                        if e - s < 0.1:
                            continue
                        instance_list.append({
                            "lyric": segment_text,
                            "start_time_sec": round(s, 4),
                            "end_time_sec": round(e, 4),
                            "length": round(e - s, 4)
                        })

                    if len(instance_list) >= 2:
                        segments[str(segment_index)] = instance_list
                        segment_index += 1
                    break  # 첫 번째 유효한 segment만 저장
        return segments
    # ----------- 6. 처리 및 저장 -----------
    repeated = find_all_repeated_sequences(syllables, timings, min_len=4)

    results = []
    if repeated:
        #ngram_len = len(repeated[0][0])
        for ngram, idx in repeated:
            ngram_len = len(ngram) 
            start_time = timings[idx][0]
            end_time = timings[idx + ngram_len - 1][1]
            results.append({
                "lyric": " ".join(ngram),
                "start_time_sec": round(start_time, 4),
                "end_time_sec": round(end_time, 4),
                "length": round(end_time - start_time, 4)
            })

    longest = build_repeat_group_json(results)
    segment = segment_lyrics(longest)
    output_json_path=txt_path.replace(".txt", "_unison.json")
    if len(segment)==0:
        if os.path.exists(output_json_path):
            os.remove(output_json_path)
            print(f"🗑️ 기존 파일 삭제됨: {output_json_path}")
        else:
            print(f"⚠️ segment 없음, 저장하지 않음: {output_json_path}")
        continue
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({"longest": longest, "segment": segment}, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON 저장 완료: {output_json_path}")
    