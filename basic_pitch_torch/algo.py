import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.ndimage import label, find_objects
FILLE_NUM=0
def fine_continous(salience_map):
    # salience_map: (freq_bin, time_frame)
    threshold = 0.2  # 임계값 (조정 가능)
    binary_map = (salience_map > threshold).astype(np.int32)
    labeled_map, num_features = label(binary_map)
    objects = find_objects(labeled_map)
    return objects


def angle_diff_rad(a1, a2):
    diff = abs(a1 - a2)
    return min(diff, np.pi - diff)  # 최소 각도차 (0 ~ pi/2)
def keep_top2_bboxes_by_area(bboxes):
    if len(bboxes) <= 2:
        return bboxes  # 이미 2개 이하면 그냥 반환
    
    # 넓이 계산
    bbox_areas = []
    for bbox in bboxes:
        y_slice, x_slice = bbox
        height = y_slice.stop - y_slice.start
        width = x_slice.stop - x_slice.start
        area = width * height
        bbox_areas.append((area, bbox))
    
    # 넓이 기준 내림차순 정렬
    bbox_areas.sort(reverse=True, key=lambda x: x[0])
    
    # 상위 2개만 반환
    top2 = [bbox_areas[0][1], bbox_areas[1][1]]
    return top2
def keep_top2_bboxes_by_salience(bboxes, salience_map):
    """
    bboxes: list of (y_slice, x_slice)
    salience_map: 2D numpy array (freq, time)
    """
    if len(bboxes) <= 2:
        return bboxes  # 이미 2개 이하면 그대로 반환

    bbox_scores = []
    for bbox in bboxes:
        #b = bbox
        y_slice, x_slice = bbox
        # salience map indexing → transpose 고려
        salience_sum = np.sum(salience_map[y_slice, x_slice])  # (x,y) 순서
        bbox_scores.append((salience_sum, bbox))
    # salience sum 기준 내림차순 정렬
    bbox_scores.sort(reverse=True, key=lambda x: x[0])
    # 상위 2개 bbox 반환
    top2 = [bbox_scores[0][1], bbox_scores[1][1]]
    return top2

def filter_objects_by_height(objects, min_height=2,min_area=8):
    filtered = []
    for obj in objects:
        if obj is None or len(obj) != 2:
            continue  # 무효 객체 무시
        y_slice, x_slice = obj
        height = y_slice.stop - y_slice.start
        width = x_slice.stop - x_slice.start
        area = width * height
        if height > min_height and area > min_area:
            filtered.append(obj)
    return filtered
def check_overlap_time_histogram(objects, min_overlap_frames=20, total_frames=300):
    """
    objects: list of (y_slice, x_slice) tuples
    total_frames: salience map의 총 frame 개수 (x-axis size)
    """
    frame_count = np.zeros(total_frames, dtype=int)
    
    # 각 bbox의 x range에 count 증가
    for obj in objects:
        if obj is None or len(obj) != 2:
            continue
        x_slice, y_slice = obj
        x_start, x_end = x_slice.start, x_slice.stop
        frame_count[x_start:x_end] += 1
    
    # frame_count >= 2 → overlap된 frame
    overlap_mask = (frame_count >= 2).astype(int)
    
    # overlap region detect
    overlaps = []
    in_overlap = False
    start_idx = None
    for i, val in enumerate(overlap_mask):
        if val == 1 and not in_overlap:
            # overlap 시작
            in_overlap = True
            start_idx = i
        elif val == 0 and in_overlap:
            # overlap 끝
            end_idx = i
            overlap_length = end_idx - start_idx
            if overlap_length >= min_overlap_frames:
                overlaps.append((start_idx, end_idx, overlap_length))
            in_overlap = False
    # 마지막 구간 처리
    if in_overlap:
        end_idx = len(overlap_mask)
        overlap_length = end_idx - start_idx
        if overlap_length >= min_overlap_frames:
            overlaps.append((start_idx, end_idx, overlap_length))
    
    return overlaps


def visualize_salience_map(contour,path,objects=None, overlaps=None):
    
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    im = ax.imshow(contour.T, origin='lower', aspect='auto', cmap='magma')

    if objects is not None:
        for idx, obj in enumerate(objects):
            if obj is None or len(obj) != 2:
                continue  # 무시하거나 오류 메시지 출력
            (y_slice, x_slice) = obj
            y_start, y_stop = y_slice.start, y_slice.stop
            x_start, x_stop = x_slice.start, x_slice.stop
            # rectangle 기본: cyan
            edgecolor = 'cyan'
            linewidth = 1.5

            rect = plt.Rectangle(
                (y_start, x_start),  # (x축 좌표, y축 좌표) → swap
                y_stop - y_start,    # width (time 방향)
                x_stop - x_start,    # height (freq 방향)
                edgecolor=edgecolor,
                facecolor='none',
                linewidth=linewidth,
            )

            ax.add_patch(rect)
    if overlaps is not None:
        for overlap in overlaps:
            start_frame, end_frame, length = overlap
            # y축 전체 커버 → 0~contour.shape[0]
            ax.axvspan(
                start_frame, end_frame,
                ymin=0, ymax=1,
                facecolor='red',
                alpha=0.2
            )
            # 라벨 optional
            ax.text((start_frame + end_frame) / 2, contour.shape[0]-10, f"{length}", 
                    color='red', ha='center', va='top', fontsize=8)

    plt.colorbar(im,label='Salience')
    plt.xlabel('Frame')
    plt.ylabel('Frequency Bin')
    plt.title('Pitch Contour (Salience Map)')
    plt.savefig(path)
    print(f"Saved salience map to {path}")
    
def visualize_salience_map_with_upper_lower(contour, path, objects=None, overlaps=None, timestamp_upper_lower=None):
    """
    timestamp_upper_lower: dict {timestamp: {'upper': bbox, 'lower': bbox}}
    """
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    im = ax.imshow(contour.T, origin='lower', aspect='auto', cmap='magma')

    # upper/lower bbox unique 추출
    upper_bboxes = []
    lower_bboxes = []
    if timestamp_upper_lower is not None:
        for v in timestamp_upper_lower.values():
            if v['upper'] is not None and v['upper'] not in upper_bboxes:
                upper_bboxes.append(v['upper'])
            if v['lower'] is not None and v['lower'] not in lower_bboxes:
                lower_bboxes.append(v['lower'])

    # 전체 bbox (기본 cyan)
    if objects is not None:
        for obj in objects:
            if obj is None or len(obj) != 2:
                continue
            (y_slice, x_slice) = obj
            y_start, y_stop = y_slice.start, y_slice.stop
            x_start, x_stop = x_slice.start, x_slice.stop
            
            # color priority
            if obj in upper_bboxes:
                edgecolor = 'lime'
            elif obj in lower_bboxes:
                edgecolor = 'red'
            else:
                edgecolor = 'cyan'
            
            rect = plt.Rectangle(
                (y_start, x_start),  # (x축 좌표, y축 좌표) → swap
                y_stop - y_start,
                x_stop - x_start,
                edgecolor=edgecolor,
                facecolor='none',
                linewidth=1.5
            )
            ax.add_patch(rect)

    # overlap 영역 highlight
    if overlaps is not None:
        for overlap in overlaps:
            start_frame, end_frame, length = overlap
            ax.axvspan(
                start_frame, end_frame,
                ymin=0, ymax=1,
                facecolor='orange',
                alpha=0.2
            )
            ax.text((start_frame + end_frame) / 2, contour.shape[0]-10, f"{length}",
                    color='orange', ha='center', va='top', fontsize=8)

    plt.colorbar(im, label='Salience')
    plt.xlabel('Frame')
    plt.ylabel('Frequency Bin')
    plt.title('Pitch Contour (Salience Map)')
    plt.savefig(path)
    #print(f"Saved salience map to {path}")

def joint_process(source1_salience, source2_salience, objects1, objects2, overlaps1, overlaps2):
    overlap_between_sources = find_overlap_between_sources(overlaps1, overlaps2)

    for overlap_start, overlap_end in overlap_between_sources:
        # 겹치는 구간 내 포함된 bbox 추출
        bboxes1 = get_bboxes_in_overlap(objects1, overlap_start, overlap_end)
        bboxes2 = get_bboxes_in_overlap(objects2, overlap_start, overlap_end)
        if len(bboxes1)>2:
            bboxes1=keep_top2_bboxes_by_area(bboxes1)
        if len(bboxes2)>2:
            bboxes2=keep_top2_bboxes_by_area(bboxes2)
        if len(bboxes1) == 2 and len(bboxes2) == 2:
            #print(f"✅ Valid overlap at frame {overlap_start}-{overlap_end}: 2 bbox in both sources")
            # source1 → 아래쪽 bbox zero
            y1_0 = bboxes1[0][1].start
            y1_1 = bboxes1[1][1].start
            lower_idx1 = 0 if y1_0 < y1_1 else 1  # 더 작은 쪽(더 낮은 쪽)

            y_slice1, x_slice1 = bboxes1[lower_idx1]
            rm_start=max(overlap_start,y_slice1.start)
            rm_end=min(overlap_end,y_slice1.stop)
            new_slice1 = slice(rm_start, rm_end)
            source1_salience[new_slice1, x_slice1] = FILLE_NUM  # source1_salience : (249=time,120=freq)

            # source2 → 위쪽 bbox zero
            y2_0 = bboxes2[0][1].start
            y2_1 = bboxes2[1][1].start
            upper_idx2 = 0 if y2_0 > y2_1 else 1  # 더 위쪽(y가 작음)

            y_slice2, x_slice2 = bboxes2[upper_idx2] #bbox 앞이 시간, 뒤가 freq
            rm_start=max(overlap_start,y_slice2.start)
            rm_end=min(overlap_end,y_slice2.stop)
            new_slice2 = slice(rm_start, rm_end)
            source2_salience[new_slice2, x_slice2] = FILLE_NUM  # (y,x) → salience는 freq,time axis

        else:
            pass
            #print(f"❌ Skipping overlap at {overlap_start}-{overlap_end}: {len(bboxes1)} bbox in source1, {len(bboxes2)} in source2")
    return source1_salience, source2_salience
def find_overlap_between_sources(overlaps1, overlaps2):
    overlap_between_sources = []
    for start1, end1, _ in overlaps1:
        for start2, end2, _ in overlaps2:
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            if overlap_start < overlap_end:
                overlap_between_sources.append((overlap_start, overlap_end))
    return overlap_between_sources


def get_bboxes_in_overlap(objects, overlap_start, overlap_end):
    included = []
    for obj in objects:
        if obj is None or len(obj) != 2:
            continue
        x_slice, y_slice = obj
        x_start = x_slice.start
        x_end = x_slice.stop
        if x_end > overlap_start and x_start < overlap_end:
            included.append(obj)
    return included

def is_bbox_overlap(bbox1, bbox2, threshold_ratio=0.4):
    y1, x1 = bbox1
    y2, x2 = bbox2
    
    # overlap 좌표 계산
    x_overlap_start = max(x1.start, x2.start)
    x_overlap_end = min(x1.stop, x2.stop)
    y_overlap_start = max(y1.start, y2.start)
    y_overlap_end = min(y1.stop, y2.stop)
    
    overlap_width = max(0, x_overlap_end - x_overlap_start)
    overlap_height = max(0, y_overlap_end - y_overlap_start)
    overlap_area = overlap_width * overlap_height
    
    # bbox1, bbox2 넓이
    area1 = (x1.stop - x1.start) * (y1.stop - y1.start)
    area2 = (x2.stop - x2.start) * (y2.stop - y2.start)
    
    # 각 기준 비율 계산
    ratio1 = overlap_area / area1 if area1 > 0 else 0
    ratio2 = overlap_area / area2 if area2 > 0 else 0
    
    # overlap 비율이 threshold 이상일 때 True
    return ratio1 >= threshold_ratio or ratio2 >= threshold_ratio




def find_nonoverlapping_regions(overlapsA, overlapsB):
    nonoverlapA = []
    for startA, endA, *_ in overlapsA:
        overlap_found = False
        for startB, endB, *_ in overlapsB:
            if startA < endB and endA > startB:  # overlap 존재
                overlap_found = True
                break
        if not overlap_found:
            nonoverlapA.append((startA, endA))
    return nonoverlapA


def single_process(source1_salience, source2_salience, objects1, objects2, overlaps1, overlaps2, FILL_NUM=0):
    nonoverlaps1 = find_nonoverlapping_regions(overlaps1, overlaps2)
    nonoverlaps2 = find_nonoverlapping_regions(overlaps2, overlaps1)
    _source1_salience = source1_salience.copy()
    _source2_salience = source2_salience.copy()
    is_source1 = False
    is_source2 = False
    
    
    source2_mask = np.ones_like(source2_salience, dtype=np.float32)
    for obj in objects2:
        if obj is None or len(obj) != 2:
            continue
        y_slice, x_slice = obj
        source2_mask[y_slice, x_slice] = 0
    #visualize_salience_map(source2_mask, path=f"source2.png")
    for overlap_start, overlap_end in nonoverlaps1:
        #print(f"[source1] processing nonoverlap region: {overlap_start}-{overlap_end}")
        temp_mask = np.ones_like(source2_mask, dtype=np.float32)
        temp_mask[overlap_start:overlap_end, :] = source2_mask[overlap_start:overlap_end,:]
        _source1_salience = _source1_salience * temp_mask
        is_source1 = True
    # 2️⃣ source2 → nonoverlap 구간
    source1_mask = np.ones_like(source1_salience, dtype=np.float32)
    for obj in objects1:
        if obj is None or len(obj) != 2:
            continue
        y_slice, x_slice = obj
        source1_mask[y_slice, x_slice] = 0
    for overlap_start, overlap_end in nonoverlaps2:
        #print(f"[source2] processing nonoverlap region: {overlap_start}-{overlap_end}")
        temp_mask = np.ones_like(source1_mask, dtype=np.float32)
        temp_mask[overlap_start:overlap_end, :] = source1_mask[overlap_start:overlap_end,:]
        _source2_salience=_source2_salience*temp_mask
        is_source2 = True
    return _source1_salience, _source2_salience
    


def calculate_total_range_length(ranges):
    """
    ranges: list of (start, end, length) tuple
    반환: union된 total length
    """
    if not ranges:
        return 0
    # (start, end)만 추출
    ranges = [(r[0], r[1]) for r in ranges]
    
    # start 기준 sorting
    ranges = sorted(ranges, key=lambda x: x[0])
    merged = []
    current_start, current_end = ranges[0]
    for start, end in ranges[1:]:
        if start <= current_end:  # 겹침
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    
    # total length 계산
    total_length = sum(end - start for start, end in merged)
    return total_length

def is_overlap_80_percent(overlaps, objects,threshold=0.6):
    # overlaps → (start, end) tuple 리스트
    overlap_length = calculate_total_range_length(overlaps)

    # objects → x_slice로부터 (start, end) 추출
    object_ranges = []
    for obj in objects:
        if obj is None or len(obj) != 2:
            continue
        x_slice, _ = obj
        object_ranges.append((x_slice.start, x_slice.stop))
    object_length = calculate_total_range_length(object_ranges)

    #print(f"Overlap length: {overlap_length}, Object length: {object_length}")
    
    if object_length == 0:
        return False  # divide by zero 방지
    ratio = overlap_length / object_length
    #print(f"Overlap ratio: {ratio:.3f}")
    return ratio >= threshold#,overlap_length, object_length

def get_upper_lower_bboxes_per_timestamp(objects, salience_map):
    """
    각 timestamp별로 포함되는 bbox 중 → 위/아래 bbox를 구분
    objects: list of (y_slice, x_slice)
    total_frames: 전체 frame 개수 (x-axis size)
    
    반환:
        dict {timestamp: {'upper': bbox, 'lower': bbox}}
    """
    total_frames = salience_map.shape[0]
    result = {}
    for t in range(total_frames):
        included_bboxes = []
        for obj in objects:
            if obj is None or len(obj) != 2:
                continue
            x_slice, y_slice = obj
            if x_slice.start <= t < x_slice.stop:
                #y_center = (y_slice.start + y_slice.stop) / 2
                included_bboxes.append(obj)
        #obj_list = [obj for _, obj in included_bboxes]

        if len(included_bboxes) >= 2:
            top2 = keep_top2_bboxes_by_salience(included_bboxes, salience_map)
            assert len(top2) == 2, f"Expected 2 bboxes, got {len(top2)}"
            is_sorted=(top2[0][1].start+top2[0][1].stop)>(top2[1][1].start+top2[1][1].stop)
            if is_sorted:
                result[t] = {
                    'upper': top2[0],  # y_center 가장 큰 → pitch 높은
                    'lower': top2[1]    # y_center 가장 작은 → pitch 낮은
                }
            else:
                result[t] = {
                    'upper': top2[1],  # y_center 가장 큰 → pitch 높은
                    'lower': top2[0]    # y_center 가장 작은 → pitch 낮은
                }
    
    return result

def apply_upper_lower_to_salience(est1, est2,mix, timestamp_upper_lower):
    """
    est1, est2: salience maps (freq, time)
    timestamp_upper_lower: dict {t: {'upper': bbox, 'lower': bbox}}
    
    반환: est1_new, est2_new
    """
    est1_new = est1.copy()
    est2_new = est2.copy()
    
    for t, v in timestamp_upper_lower.items():
        # 해당 timestamp column 전체 0으로
        est1_new[t, :] = 0
        est2_new[t, :] = 0
        
        if v['upper'] is not None:
            y_slice, x_slice = v['upper']
            # y_slice: freq range, x_slice: time range
            # upper bbox의 freq 영역에 est1 채우기
            est1_new[t, x_slice] = mix[t, x_slice]
        
        if v['lower'] is not None:
            y_slice, x_slice = v['lower']
            # lower bbox의 freq 영역에 est2 채우기
            est2_new[t, x_slice] = mix[t, x_slice]
    
    return est1_new, est2_new
