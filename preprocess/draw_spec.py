import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
path="Experiments/checkpoint/penalty0/20250808_09/result_best_model/examples_unison_24k/ex_146/s0.wav"
#path = "Experiments/checkpoint/penalty5/20250812_04/result_best_model/examples_unison_24k/ex_146/s0_estimate.wav"

y, sr = librosa.load(path, sr=24000)
S = np.abs(librosa.stft(y, n_fft=960, hop_length=240))**2
S_db = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(10,4), facecolor="black")
ax = plt.gca()
ax.set_facecolor("black")

img = librosa.display.specshow(
    S_db, sr=sr, hop_length=240,
    x_axis="time", y_axis="hz",
    cmap="viridis",                 # ← 이거!
    vmin=S_db.max()-80, vmax=S_db.max()  # 다이내믹레인지(느낌 조절)
)
ax.set_axis_off()              # ← 축/틱/라벨/프레임 off
plt.margins(0)                 # 여백 제거
plt.subplots_adjust(0, 0, 1, 1)

save_path = path.replace(".wav", ".png")
plt.savefig(save_path, dpi=300, facecolor="black",
            bbox_inches="tight", pad_inches=0)  # ← 테두리 없이 저장

print(save_path)