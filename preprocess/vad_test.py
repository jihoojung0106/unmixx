import json
from funasr import AutoModel
import glob
from tqdm import tqdm
import os
data_list=glob.glob("duet_svs/MedleyVox/**/*.wav",recursive=True)
# 모델 로딩
model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")

for wav_file in tqdm(data_list):
    if os.path.exists(wav_file.replace(".wav",".json")):
        print(f"Skipping {wav_file}, already processed.")
        continue
    # VAD inference 수행
    res = model.generate(input=wav_file)
    # 결과 출력
    #print(res)
    
    # JSON 파일로 저장
    with open(wav_file.replace(".wav",".json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
        print(f'Saved VAD results to {wav_file.replace(".wav",".json")}')
