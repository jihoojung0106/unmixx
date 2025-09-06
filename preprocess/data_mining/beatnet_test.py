from BeatNet.BeatNet import BeatNet
path1 = '/home/jungji/real_sep/tiger_ver6/duet_svs/24k/OpenSinger/ManRaw_0_今天是你的生日,妈妈_0.wav'
path2 = '/home/jungji/real_sep/tiger_ver6/duet_svs/24k/OpenSinger/WomanRaw_4_遇见_15.wav'
# BeatNet 객체 생성 (예: 실시간 모드)
estimator = BeatNet(1, mode='realtime', inference_model='PF', plot=['beat_particles'], thread=False)

# 오디오 파일 경로를 넣어 처리
output = estimator.process(path1)

print(output)  # 결과 출력 (numpy 배열)
