from look2hear.models.metricgan import MetricDiscriminator
import os
import torch
import torchaudio
from collections import OrderedDict
from torch.nn.functional import sigmoid

def get_mag(input):
        stft_spec = torch.stft(input, n_fft=960, hop_length=240, 
                          window=torch.hann_window(960).to(input.device).type(input.type()),
                          return_complex=True) #(batch,321,401)
        stft_spec = torch.view_as_real(stft_spec)
        mag = torch.sqrt(stft_spec.pow(2).sum(-1)+(1e-9)) #(1,321,401)
        return mag

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
discriminator=MetricDiscriminator()
discriminator.eval()
state_dict = torch.load("Experiments/checkpoint/gan_dynamic5/20250720_11/epoch=3.ckpt")["state_dict"]

# prefix 제거
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith("discriminator."):
        new_key = k[len("discriminator."):]  # prefix 제거
        new_state_dict[new_key] = v
missing,unload=discriminator.load_state_dict(new_state_dict,strict=False)
print(f"Missing keys: {len(missing)}, Unloaded keys: {len(unload)}")
discriminator.to(device)

audio_path="separated_audio/fin_van1_epoch=266/shwn/spk1.wav"
waveform, original_sr = torchaudio.load(audio_path)
start_sec = 4.0
end_sec = 6.0
target_sr = 24000
start_sample = int(start_sec * target_sr)
end_sample = int(end_sec * target_sr)
# 모노 변환 및 자르기
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)  # (1, T)
waveform = waveform[:, start_sample:end_sample]
mag = get_mag(waveform).to(device)
output= discriminator(mag) #mix
prob = sigmoid(output)       # 확률로 변환
pred = (prob >= 0.5).float() # threshold 0.5 기준으로 binary 예측

print(f"Output: {output}, Probability: {prob}, Prediction: {pred}")