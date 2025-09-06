import json
from typing import List, Tuple

def load_vad_intervals_from_json(json_path: str) -> List[Tuple[int, int]]:
    """
    JSON 파일에서 VAD 구간을 불러오는 함수
    :param json_path: JSON 파일 경로
    :param key_contains: 사용할 key에 포함된 문자열 (예: 'seg_53')
    :return: 구간 리스트 [(start, end), ...]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    intervals = []
    for entry in data:
        for start_end in entry.get('value', []):
            if len(start_end) == 2:
                intervals.append((start_end[0], start_end[1]))
    return intervals


def conservative_vad_union_intervals(
    file1_path,
    file2_path,
    min_duration_ms: int = 200
) -> List[Tuple[int, int]]:
    """
    두 VAD 구간 리스트(vad1, vad2)를 받아서
    - 합집합 방식으로 최대한 구간을 많이 쪼개고,
    - 각 구간이 최소 duration(ms) 이상인 것만 반환하는 함수

    :param vad1: [(start, end), ...] 첫 번째 VAD 구간 리스트 (ms 단위)
    :param vad2: [(start, end), ...] 두 번째 VAD 구간 리스트 (ms 단위)
    :param min_duration_ms: 최소 구간 길이 (밀리초), 기본 200ms
    :return: 조건에 맞는 구간 리스트 [(start, end), ...]
    """
    vad1 = load_vad_intervals_from_json(file1_path)
    vad2 = load_vad_intervals_from_json(file2_path)
    # 1. 모든 구간의 시작, 끝점을 추출해 정렬한 뒤 중복 제거
    boundaries = set()
    for start, end in vad1 + vad2:
        boundaries.add(start)
        boundaries.add(end)
    sorted_boundaries = sorted(boundaries)

    # 2. 각 작은 구간별 포함 여부 체크 함수
    def is_covered_by_any_interval(start: int, end: int, intervals: List[Tuple[int, int]]) -> bool:
        for s, e in intervals:
            if s <= start and end <= e:
                return True
        return False

    # 3. 작은 구간 단위로 나누고 포함 여부 체크 (vad1 또는 vad2 중 하나라도 포함 시)
    merged_intervals = []
    for i in range(len(sorted_boundaries) - 1):
        start = sorted_boundaries[i]
        end = sorted_boundaries[i + 1]

        # 4. 최소 길이 조건 검사
        if (end - start) < min_duration_ms:
            continue

        covered = is_covered_by_any_interval(start, end, vad1) or is_covered_by_any_interval(start, end, vad2)

        if covered:
            merged_intervals.append((start, end))

    return merged_intervals


# 함수 사용 예시
if __name__ == '__main__':
    # JSON 파일 경로와 키 필터 문자열
    file1_path = 'duet_svs/MedleyVox/unison/FamilyBand_Again/seg_4/gt/FamilyBand_Again_RAW_02_01 - seg_4.json'
    file2_path = 'duet_svs/MedleyVox/unison/FamilyBand_Again/seg_4/gt/FamilyBand_Again_RAW_02_02 - seg_4.json'

    

    union_intervals = conservative_vad_union_intervals(file1_path, file2_path)
    print("최종 합집합 구간:", union_intervals)
