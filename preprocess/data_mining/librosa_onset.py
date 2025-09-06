import librosa
import numpy as np

def rhythmic_alignment_score(wav1, wav2, sr=22050):
    onset_env1 = librosa.onset.onset_strength(y=wav1, sr=sr)
    onset_env2 = librosa.onset.onset_strength(y=wav2, sr=sr)

    # 정규화
    onset_env1 = onset_env1 / (np.linalg.norm(onset_env1) + 1e-8)
    onset_env2 = onset_env2 / (np.linalg.norm(onset_env2) + 1e-8)

    # Cross-correlation
    corr = np.correlate(onset_env1, onset_env2, mode='full')
    return np.max(corr)
from librosa.sequence import dtw

def rhythmic_dtw_score(wav1, wav2, sr=22050):
    onset1 = librosa.onset.onset_strength(y=wav1, sr=sr)
    onset2 = librosa.onset.onset_strength(y=wav2, sr=sr)

    D, wp = dtw(onset1[:, np.newaxis], onset2[:, np.newaxis], metric='euclidean')
    cost = D[-1, -1] / len(wp)
    return 1 / (1 + cost)  # 낮은 cost → 높은 score
def onset_match_score(wav1, wav2, sr=22050, max_dev=0.05):
    on1 = librosa.onset.onset_detect(y=wav1, sr=sr, backtrack=True, units='time')
    on2 = librosa.onset.onset_detect(y=wav2, sr=sr, backtrack=True, units='time')
    N = min(len(on1), len(on2))
    if N == 0:
        return 0.0
    deltas = np.abs(on1[:N] - on2[:N])
    return 1 - np.mean(deltas) / max_dev
if __name__=="__main__":
    path1 = '/home/jungji/real_sep/tiger_ver6/duet_svs/MedleyVox_4s_16k/duet/CelestialShore_DieForUs/seg_45/gt/CelestialShore_DieForUs_RAW_01_02 - seg_45_chunk000.wav'
    path2 = '/home/jungji/real_sep/tiger_ver6/duet_svs/MedleyVox_4s_16k/duet/CelestialShore_DieForUs/seg_45/gt/CelestialShore_DieForUs_RAW_02_01 - seg_45_chunk000.wav'
    wav1, sr1 = librosa.load(path1, sr=None)
    wav2, sr2 = librosa.load(path2, sr=None)

    ras = rhythmic_alignment_score(wav1, wav2, sr=sr1)
    dtws = rhythmic_dtw_score(wav1, wav2, sr=sr1)
    oms = onset_match_score(wav1, wav2, sr=sr1)

    print(f"RAS: {ras}, DTWS: {dtws}, OMS: {oms}")
    
