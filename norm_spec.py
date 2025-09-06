# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# audio_path를 입력으로 받아
# - STFT magnitude 스펙트로그램(win=960, hop=240) 생성
# - 전체(mean/std)로 z-score 정규화
# - 이미지를 PNG로 저장

# 사용:
#     python save_spec.py --audio path/to/audio.wav --outdir specs_out
# """

# import os
# import math
# import argparse
# from pathlib import Path
# import numpy as np
# import torch
# import torchaudio
# import matplotlib.pyplot as plt

# # -------------------------
# # Util: normalize (mel 예시와 동일한 로직)
# # -------------------------
# def tensor_like(value, ref: torch.Tensor):
#     """value를 ref와 같은 device/dtype의 Tensor로 바꿔줌."""
#     if isinstance(value, (float, int)):
#         return torch.tensor(value, dtype=ref.dtype, device=ref.device)
#     if isinstance(value, list):
#         return torch.tensor(value, dtype=ref.dtype, device=ref.device)
#     if isinstance(value, np.ndarray):
#         return torch.from_numpy(value).to(ref.device, dtype=ref.dtype)
#     if isinstance(value, torch.Tensor):
#         return value.to(ref.device, dtype=ref.dtype)
#     raise TypeError(f"Unsupported type for conversion: {type(value)}")

# def spec_normalize(data: torch.Tensor, mu, std):
#     """
#     data: (freq_bins, frames) Tensor
#     mu, std: float/int/list/np.ndarray/torch.Tensor 허용
#     """
#     mu_t = tensor_like(mu, data)
#     std_t = tensor_like(std, data)
#     # (freq, T) -> (freq, T), mu/std를 (freq,1)로 넣고 싶다면 unsqueeze 사용
#     if mu_t.ndim == 1:
#         mu_t = mu_t.unsqueeze(-1)
#     if std_t.ndim == 1:
#         std_t = std_t.unsqueeze(-1)
#     return (data - mu_t) / (std_t + 1e-12)

# # -------------------------
# # Core
# # -------------------------
# def load_audio_mono(path: str, target_sr: int) -> tuple[torch.Tensor, int]:
#     """
#     오디오 로드 -> 모노로 변환
#     returns: (waveform[T], sr)
#     """
#     wav, sr = torchaudio.load(path)  # (C, T)
#     if wav.shape[0] > 1:
#         wav = torch.mean(wav, dim=0, keepdim=True)
#     # Resample if needed
#     if (target_sr is not None) and (sr != target_sr):
#         wav = torchaudio.functional.resample(wav, sr, target_sr)
#         sr = target_sr
#     return wav.squeeze(0), sr  # (T,), sr

# def compute_mag_spectrogram(
#     wav: torch.Tensor,
#     n_fft: int = 960,
#     win_length: int = 960,
#     hop_length: int = 240,
#     window_fn=torch.hann_window,
#     center: bool = True,
# ) -> torch.Tensor:
#     """
#     STFT magnitude spectrogram |X| 반환
#     output: (freq_bins, frames)
#     """
#     device = wav.device
#     window = window_fn(win_length, periodic=True, dtype=wav.dtype, device=device)
#     spec = torch.stft(
#         wav,
#         n_fft=n_fft,
#         hop_length=hop_length,
#         win_length=win_length,
#         window=window,
#         center=center,
#         return_complex=True,
#     )
#     mag = spec.abs()  # (freq_bins, frames)
#     return mag

# def global_mean_std(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     x 전체(mean, std) 계산. (freq, T) 기준.
#     mel 예시처럼 E[x], E[x^2]로 계산.
#     """
#     total_len = x.numel()
#     x_sum = torch.sum(x)
#     x_sq_sum = torch.sum(x ** 2)
#     mean = x_sum / total_len
#     var = (x_sq_sum / total_len) - mean ** 2
#     std = torch.sqrt(torch.clamp(var, min=0.0))
#     return mean, std

# def save_spectrogram_image(
#     spec: torch.Tensor,
#     out_path: str,
#     vmin: float=None,
#     vmax: float=None,
#     add_colorbar: bool = True,
#     title: str=None,
# ):
#     """
#     spec: (freq, T) Tensor (CPU)
#     """
#     spec_np = spec.detach().cpu().numpy()
#     plt.figure(figsize=(10, 4))
#     im = plt.imshow(
#         spec_np,
#         origin="lower",
#         aspect="auto",
#         vmin=vmin,
#         vmax=vmax,
#         cmap="magma",  # 선호하는 colormap으로 변경 가능
#     )
#     if title:
#         plt.title(title)
#     plt.xlabel("Frames")
#     plt.ylabel("Frequency bins")
#     if add_colorbar:
#         plt.colorbar(im, fraction=0.046, pad=0.04)
#     plt.tight_layout()
#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(out_path, dpi=200)
#     plt.close()

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--audio", type=str, default='sample_music/ex_4/s0_estimate.wav', help="Path to input audio (wav/flac/etc.)")
#     ap.add_argument("--sr", type=int, default=24000, help="Optional resample rate (e.g., 24000). None = keep original.")
#     ap.add_argument("--n_fft", type=int, default=960)
#     ap.add_argument("--win", type=int, default=960)
#     ap.add_argument("--hop", type=int, default=240)
#     ap.add_argument("--log1p", action="store_true", help="Apply log1p to magnitude before normalization/plot")
#     args = ap.parse_args()

#     wav, sr = load_audio_mono(args.audio, args.sr)

#     # 1) 스펙트로그램 (magnitude)
#     mag = compute_mag_spectrogram(
#         wav, n_fft=args.n_fft, win_length=args.win, hop_length=args.hop
#     )  # (freq, T)

#     # (선택) 시각 안정성을 위해 log1p 적용 가능
#     if args.log1p:
#         mag = torch.log1p(mag)

#     # 2) 전체 mean/std (mel 예시와 동일한 방식)
#     mu, sigma = global_mean_std(mag)

#     # 3) z-score 정규화
#     mag_norm = spec_normalize(mag, mu, sigma)

#     # 4) 이미지 저장
#     args.outdir=os.path.dirname(args.audio) #if not args.outdir else args.outdir
#     stem = Path(args.audio).stem
#     out_png = args.audio.replace('.wav', '_spec_norm.png')#.replace('.flac', '_spec_norm.png')
#     save_spectrogram_image(
#         mag_norm, str(out_png),
#         title=f"{stem} (normed) | win={args.win}, hop={args.hop}, sr={sr}",
#     )
#     print(f"[OK] Saved: {out_png}")

# if __name__ == "__main__":
#     main()
