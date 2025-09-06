import librosa
import numpy as np
import mir_eval

# --- 두 오디오 파일 로드 및 비트 추출 ---

path1 = '/home/jungji/real_sep/tiger_ver6/duet_svs/MedleyVox_4s_16k/unison/CassandraJenkins_PerfectDay/seg_5/gt/CassandraJenkins_PerfectDay_RAW_07_01 - seg_5_chunk000.wav'
y, sr = librosa.load(path1)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beats, sr=sr)  # 초 단위 비트 시퀀스
print("Beat times (ref):", beat_times)

path2 = '/home/jungji/real_sep/tiger_ver6/duet_svs/MedleyVox_4s_16k/unison/FacesOnFilm_WaitingForGa/seg_6/gt/FacesOnFilm_WaitingForGa_RAW_10_04 - seg_6_chunk000.wav'
y1, sr1 = librosa.load(path2)
tempo1, beats1 = librosa.beat.beat_track(y=y1, sr=sr1)
beat_times1 = librosa.frames_to_time(beats1, sr=sr1)
print("Beat times1 (est):", beat_times1)
# --- 변수명 명확히 ---
ref_beats = beat_times       # 기준 비트 시퀀스 (짧거나 기준이 될 쪽)
est_beats = beat_times1      # 비교 대상 비트 시퀀스
if len(ref_beats) == 0 or len(est_beats) == 0:
    print("한쪽 비트 배열에 비트가 없습니다. 다시 확인하세요.")
    exit()

# --- 1) est_beats를 기준 비트 첫 위치에 맞춰 시간 이동(shift) ---

idx = np.argmin(np.abs(est_beats - ref_beats[0]))  # est_beats에서 ref_beats 첫 비트에 가장 가까운 인덱스
time_shift = ref_beats[0] - est_beats[idx]
est_shifted = est_beats + time_shift

# --- 2) ref_beats 시간 구간 안에 est_shifted 포함되는 부분만 추출 ---

start, end = ref_beats[0], ref_beats[-1]
est_matched = est_shifted[(est_shifted >= start) & (est_shifted <= end)]

# --- 3) mir_eval.beat.evaluate() 호출 전, 비트 개수 확인 ---

print("ref_beats length:", len(ref_beats))
print("est_matched length:", len(est_matched))
scores = mir_eval.onset.f_measure(beat_times, beat_times1)
print(scores)
