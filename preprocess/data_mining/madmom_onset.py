from madmom.features.onsets import RNNOnsetProcessor, OnsetPeakPickingProcessor,SpectralOnsetProcessor
path1 = '/home/jungji/real_sep/tiger_ver6/duet_svs/MedleyVox_4s_16k/unison/CassandraJenkins_PerfectDay/seg_6/gt/CassandraJenkins_PerfectDay_RAW_07_02 - seg_6_chunk000.wav'
path2 = '/home/jungji/real_sep/tiger_ver6/duet_svs/MedleyVox_4s_16k/unison/FilthyBird_IdLikeToKnow/seg_7/gt/FilthyBird_IdLikeToKnow_RAW_09_01 - seg_7_chunk000.wav'
    
# 오디오 파일을 처리해서 온셋 활성화 함수 추출
act = tmux RNNOnsetProcessor()(path1)
#onsets = OnsetPeakPickingProcessor(fps=100)(act)  # fps는 활성화 벡터의 frame rate
print(act)  # 초 단위로 온셋 시점이 출력됨
act2 = RNNOnsetProcessor()(path2)
#onsets2 = OnsetPeakPickingProcessor(fps=100)(act2)
print(act2)  # 초 단위로 온셋 시점이 출력됨

proc = SpectralOnsetProcessor(onset_method='superflux', fps=100)
activation = proc.processor(path1)
print(activation)  # 초 단위로 온셋 활성화 함수가 출력됨