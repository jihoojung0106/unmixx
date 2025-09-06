# from madmom.features import downbeats, beats

# # 비트 확률 생성
# rnn_proc = downbeats.RNNDownBeatProcessor()

# path1 = '/home/jungji/real_sep/tiger_ver6/duet_svs/MedleyVox_4s_16k/duet/CelestialShore_DieForUs/seg_45/gt/CelestialShore_DieForUs_RAW_01_02 - seg_45_chunk000.wav'
# path2 = '/home/jungji/real_sep/tiger_ver6/duet_svs/MedleyVox_4s_16k/duet/CelestialShore_DieForUs/seg_45/gt/CelestialShore_DieForUs_RAW_02_01 - seg_45_chunk000.wav'
# activations = rnn_proc(path1)
# activations1 = rnn_proc(path2)
# proc = downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
# downbeats1 = proc(activations)
# downbeats2 = proc(activations1)
# print(downbeats1)
# print(downbeats2)
from madmom.features.beats import RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor
path1 = '/home/jungji/real_sep/tiger_ver6/duet_svs/24k/OpenSinger/ManRaw_0_今天是你的生日,妈妈_0.wav'
path2 = '/home/jungji/real_sep/tiger_ver6/duet_svs/24k/OpenSinger/WomanRaw_4_遇见_15.wav'

audio_path = '파일경로.wav'
# 1단계: beat activation 추출
act = RNNBeatProcessor()(path1)
# 2단계: 템포 추정
tempo_proc = TempoEstimationProcessor(fps=100)
tempi = tempo_proc(act)
act2 = RNNBeatProcessor()(path2)
tempo_proc2 = TempoEstimationProcessor(fps=100)
tempi2 = tempo_proc2(act2)
print('예상 템포(BPM)와 신뢰도:', tempi)
print('예상 템포(BPM)와 신뢰도:', tempi2)
