import numpy as np
import torch
import math
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple
import librosa



import math
import torch
from typing import Optional, Tuple

def _frame_signal(x: torch.Tensor, frame_length: int, hop: int) -> torch.Tensor:
    """
    x: (B, T)  -> frames: (B, F, N)  (F=frame_length, N=num_frames)
    """
    B, T = x.shape
    if T < frame_length:
        # pad to one frame
        pad = frame_length - T
        x = torch.nn.functional.pad(x, (0, pad))
        T = frame_length

    N = 1 + (T - frame_length) // hop
    # as_strided로 프레이밍 (복사 없이 뷰)
    stride_B, stride_T = x.stride()
    frames = x.as_strided(
        size=(B, frame_length, N),
        stride=(stride_B, stride_T, hop),
    )
    return frames

def pit_segmental_ssnr_loss(
    clean1: torch.Tensor,   # (B, T)
    est1: torch.Tensor,     # (B, T)    
    clean2: torch.Tensor,   # (B, T)
    est2: torch.Tensor,     # (B, T)
    sample_rate: int = 24000,
    win_ms: float = 30.0,
    hop_ratio: float = 0.25,      # hop = win * (1 - overlap) ; overlap=0.75
    eps: float = 1e-12,
    clamp: Optional[Tuple[float, float]] = (-10.0, 35.0),  # dB 클리핑. 끄려면 None
    reduction: str = "mean",       # "mean" | "sum" | "none"
) -> torch.Tensor:
    """
    PIT 기반 segmental SSNR(dB)을 계산하고, loss(= -mean SSNR)를 반환.
    반환값은 dB의 음수 평균(=loss). reduction="none"이면 프레임별/배치별 반환.

    수식(원 코드와 동일):
      snr1 = 10*log10( sig1/(n11+eps) + eps ) + 10*log10( sig2/(n22+eps) + eps )
      snr2 = 10*log10( sig1/(n12+eps) + eps ) + 10*log10( sig2/(n21+eps) + eps )
      snr  = 0.5 * max(snr1, snr2)    # frame-wise PIT
    """
    assert clean1.shape == clean2.shape == est1.shape == est2.shape, "All shapes must match (B,T)."
    assert clean1.dim() == 2, "Inputs must be (B,T)."

    B, T = clean1.shape
    win = int(round(win_ms * sample_rate / 1000.0))
    hop = max(1, int(math.floor(win * hop_ratio)))

    # 프레이밍
    c1_f = _frame_signal(clean1, win, hop)  # (B,F=720,N=530)
    c2_f = _frame_signal(clean2, win, hop)
    e1_f = _frame_signal(est1,   win, hop)
    e2_f = _frame_signal(est2,   win, hop)

    # Hann window (F,) -> (1,F,1)
    window = torch.hann_window(win, periodic=True, device=clean1.device, dtype=clean1.dtype)
    window = window.view(1, win, 1)

    # 윈도우 적용
    c1_f = c1_f * window
    c2_f = c2_f * window
    e1_f = e1_f * window
    e2_f = e2_f * window

    # 에너지 (프레임 합)
    # sum over F -> (B,N)
    def _pow2_sum(x):  # (B,F,N) -> (B,N)
        return (x * x).sum(dim=1)
    # (2) Compute the Segmental SNR
    sig1 = _pow2_sum(c1_f)                 # (B,N)
    sig2 = _pow2_sum(c2_f)
    n11  = _pow2_sum(c1_f - e1_f)
    n12  = _pow2_sum(c1_f - e2_f)
    n21  = _pow2_sum(c2_f - e1_f)
    n22  = _pow2_sum(c2_f - e2_f)

    # dB 계산 (원 코드와 동일한 형태; +eps는 안정화용)
    # torch.log10 사용 → 미분 가능
    def _snr_term(sig, noise):
        return 10.0 * torch.log10(sig / (noise + eps) + eps)

    snr1 = _snr_term(sig1, n11) + _snr_term(sig2, n22)   # (B,N)
    snr2 = _snr_term(sig1, n12) + _snr_term(sig2, n21)   # (B,N)

    # 프레임별 PIT: 더 큰 쪽 선택
    snr = 0.5 * torch.maximum(snr1, snr2)   # (B,N)

    # dB 클리핑(원 코드 준수). 불필요하면 clamp=None
    if clamp is not None:
        lo, hi = clamp
        snr = torch.clamp(snr, min=lo, max=hi)

    # loss: SSNR을 최대화하고 싶으므로 음수 부호
    if reduction == "mean":
        loss = -snr.mean()
    elif reduction == "sum":
        loss = -snr.sum()
    elif reduction == "none":
        loss = -snr   # (B,N)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    return loss




def compute_vad_ssnr(
    clean_speech1: np.ndarray,      # shape (T,)
    processed_speech1: np.ndarray,  # shape (T,)
    clean_speech2: np.ndarray,      # shape (T,)
    processed_speech2: np.ndarray,  # shape (T,)
    vad_intervals: List[Tuple[int, int]],  # [(start_ms, end_ms), ...]
    sample_rate: int = 24000,
    min_snr: float = -10.0,
    max_snr: float = 35.0,
) -> np.ndarray:
    """
    VAD 구간별 Segmental PIT-SSNR (dB)를 계산.
    각 구간 길이에 맞춘 Hann window를 곱해 에너지를 계산하고,
    (1->1 & 2->2) vs (1->2 & 2->1) 두 경우를 비교해 더 큰 합을 선택(PIT).
    최종 반환값은 각 구간의 (선택된 페어의 두 SNR 평균)이며, [min_snr, max_snr]로 클리핑.

    Returns
    -------
    ssnr_per_segment : np.ndarray, shape (num_valid_intervals,)
    """
    # 길이 체크
    if not (len(clean_speech1) == len(processed_speech1) == len(clean_speech2) == len(processed_speech2)):
        raise ValueError("All input waveforms must have the same length.")

    T = len(clean_speech1)
    EPS = np.spacing(1)

    out_vals = []
    for (start_ms, end_ms) in vad_intervals:
        # ms -> sample index
        s = max(0, int(round(start_ms * sample_rate / 1000.0)))
        e = min(T, int(round(end_ms   * sample_rate / 1000.0)))

        # 유효성 체크
        seg_len = e - s
        if seg_len < 2:
            # 너무 짧으면 스킵
            continue

        # 구간 추출
        c1 = clean_speech1[s:e].astype(np.float64, copy=False)
        p1 = processed_speech1[s:e].astype(np.float64, copy=False)
        c2 = clean_speech2[s:e].astype(np.float64, copy=False)
        p2 = processed_speech2[s:e].astype(np.float64, copy=False)

        # Hann window (구간 길이에 맞춤)
        # 0.5*(1 - cos(2π n/(N+1))) 정의를 유지
        n = np.arange(1, seg_len + 1, dtype=np.float64)
        window = 0.5 * (1.0 - np.cos(2.0 * math.pi * n / (seg_len + 1.0)))

        # 윈도우 적용 (네 신호 모두)
        c1w = c1 * window
        p1w = p1 * window
        c2w = c2 * window
        p2w = p2 * window

        # 에너지 계산
        sig_e1 = np.sum(c1w * c1w)
        sig_e2 = np.sum(c2w * c2w)

        # 두 가지 매핑에 대한 "노이즈" 에너지
        # 케이스 A: 1->1, 2->2
        n11 = np.sum((c1w - p1w) ** 2)
        n22 = np.sum((c2w - p2w) ** 2)
        # 케이스 B: 1->2, 2->1 (스왑)
        n12 = np.sum((c1w - p2w) ** 2)
        n21 = np.sum((c2w - p1w) ** 2)

        # 각 채널 SNR (dB)
        # 작은 수 안정화용 EPS 추가
        ch1_A = 10.0 * math.log10((sig_e1 + EPS) / (n11 + EPS))
        ch2_A = 10.0 * math.log10((sig_e2 + EPS) / (n22 + EPS))
        ch1_B = 10.0 * math.log10((sig_e1 + EPS) / (n12 + EPS))
        ch2_B = 10.0 * math.log10((sig_e2 + EPS) / (n21 + EPS))

        # PIT: 합이 큰 쪽을 선택
        sum_A = ch1_A + ch2_A
        sum_B = ch1_B + ch2_B
        if sum_A >= sum_B:
            snr_seg = 0.5 * (ch1_A + ch2_A)
        else:
            snr_seg = 0.5 * (ch1_B + ch2_B)

        # 클리핑
        snr_seg = max(min_snr, min(max_snr, snr_seg))
        out_vals.append(snr_seg)

    return np.asarray(out_vals, dtype=np.float64)

def compute_whole_ssnr(
    clean_speech1,          # np.ndarray, shape (T,)
    processed_speech1,      # np.ndarray, shape (T,)
    clean_speech2,          # np.ndarray, shape (T,)
    processed_speech2,
    sample_rate=24000): # np.ndarray, shape (T,)):
    # Check the length of the clean and processed speech. Must be the same.
    clean_length1 = len(clean_speech1)
    processed_length1 = len(processed_speech1)
    clean_length2 = len(clean_speech2)
    processed_length2 = len(processed_speech2)
    if clean_length1 != processed_length1 or clean_length2 != processed_length2:
        raise ValueError('Both Speech Files must be same length.')

    overall_snr1 = 10 * np.log10(np.sum(np.square(clean_speech1)) / np.sum(np.square(clean_speech1 - processed_speech1)))\
                + 10 * np.log10(np.sum(np.square(clean_speech2)) / np.sum(np.square(clean_speech2 - processed_speech2)))
    overall_snr2 = 10 * np.log10(np.sum(np.square(clean_speech1)) / np.sum(np.square(clean_speech1 - processed_speech2)))\
                + 10 * np.log10(np.sum(np.square(clean_speech2)) / np.sum(np.square(clean_speech2 - processed_speech1)))
    overall_snr = max(overall_snr1, overall_snr2) / 2
    if overall_snr1< overall_snr2:
        temp = processed_speech1
        processed_speech1 = processed_speech2
        processed_speech2 = temp
    # Global Variables
    winlength = round(30 * sample_rate / 1000)    # window length in samples
    skiprate = math.floor(winlength / 4)     # window skip in samples
    MIN_SNR = -10    # minimum SNR in dB
    MAX_SNR = 35     # maximum SNR in dB

    # For each frame of input speech, calculate the Segmental SNR
    num_frames = int(clean_length1 / skiprate - (winlength / skiprate))   # number of frames
    start = 0      # starting sample
    window = 0.5 * (1 - np.cos(2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1)))

    segmental_snr = np.empty(num_frames)
    EPS = np.spacing(1)
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
        clean_frame1 = clean_speech1[start:start + winlength]
        clean_frame2 = clean_speech2[start:start + winlength]
        processed_frame1 = processed_speech1[start:start + winlength]
        processed_frame2 = processed_speech2[start:start + winlength]
        clean_frame1 = np.multiply(clean_frame1, window)
        clean_frame2 = np.multiply(clean_frame2, window)
        processed_frame1 = np.multiply(processed_frame1, window)
        processed_frame2 = np.multiply(processed_frame2, window)

        # (2) Compute the Segmental SNR
        signal_energy1 = np.sum(np.square(clean_frame1))
        signal_energy2 = np.sum(np.square(clean_frame2))
        noise_energy1 = np.sum(np.square(clean_frame1 - processed_frame1))
        noise_energy2 = np.sum(np.square(clean_frame2 - processed_frame2))
        segmental_snr[frame_count] = (10 * math.log10(signal_energy1 / (noise_energy1 + EPS) + EPS) + 10 * math.log10(signal_energy2 / (noise_energy2 + EPS) + EPS))/2
        segmental_snr[frame_count] = max(segmental_snr[frame_count], MIN_SNR)
        segmental_snr[frame_count] = min(segmental_snr[frame_count], MAX_SNR)

        start = start + skiprate

    return overall_snr, segmental_snr

def compute_ssnr(
    clean_speech1,          # np.ndarray, shape (T,)
    processed_speech1,      # np.ndarray, shape (T,)
    clean_speech2,          # np.ndarray, shape (T,)
    processed_speech2,      # np.ndarray, shape (T,)
    sample_rate=24000,             # int
):
    # Check the length of the clean and processed speech. Must be the same.
    clean_length1 = len(clean_speech1)
    processed_length1 = len(processed_speech1)
    clean_length2 = len(clean_speech2)
    processed_length2 = len(processed_speech2)
    if clean_length1 != processed_length1 or clean_length2 != processed_length2:
        raise ValueError('Both Speech Files must be same length.')
    MIN_SNR = -10    # minimum SNR in dB
    MAX_SNR = 35     # maximum SNR in dB
    # Global Variables
    
    winlength = round(30 * sample_rate / 1000)    # window length in samples
    skiprate = math.floor(winlength / 4)     # window skip in samples
    # For each frame of input speech, calculate the Segmental SNR
    num_frames = int(clean_length1 / skiprate - (winlength / skiprate))   # number of frames
    start = 0      # starting sample
    window = 0.5 * (1 - np.cos(2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1)))
    segmental_snr = np.empty(num_frames)
    EPS = np.spacing(1)
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
        clean_frame1 = clean_speech1[start:start + winlength]
        processed_frame1 = processed_speech1[start:start + winlength]
        clean_frame2 = clean_speech2[start:start + winlength]
        processed_frame2 = processed_speech2[start:start + winlength]
        clean_frame1 = np.multiply(clean_frame1, window)
        processed_frame1 = np.multiply(processed_frame1, window)
        # (2) Compute the Segmental SNR
        signal_energy1 = np.sum(np.square(clean_frame1))
        signal_energy2 = np.sum(np.square(clean_frame2))
        noise_energy11 = np.sum(np.square(clean_frame1 - processed_frame1))
        noise_energy12 = np.sum(np.square(clean_frame1 - processed_frame2))
        noise_energy21 = np.sum(np.square(clean_frame2 - processed_frame1))
        noise_energy22 = np.sum(np.square(clean_frame2 - processed_frame2))
        snr1=10 * math.log10(signal_energy1 / (noise_energy11 + EPS) + EPS)+10 * math.log10(signal_energy2 / (noise_energy22 + EPS) + EPS) #1122 
        snr2=10 * math.log10(signal_energy1 / (noise_energy12 + EPS) + EPS)+10 * math.log10(signal_energy2 / (noise_energy21 + EPS) + EPS) #1211
        snr = max(snr1, snr2)/2
        segmental_snr[frame_count] = snr
        segmental_snr[frame_count] = max(segmental_snr[frame_count], MIN_SNR)
        segmental_snr[frame_count] = min(segmental_snr[frame_count], MAX_SNR)
        start = start + skiprate

    return segmental_snr
if __name__ == "__main__":
    # Example usage
    clean_speech1, _ = librosa.load("Experiments/checkpoint/fin_van1/20250716_23/result_best_model/examples_duet_24k/ex_13/s0.wav", sr=24000)  # 2 seconds of random noise
    clean_speech2, _ = librosa.load("Experiments/checkpoint/fin_van1/20250716_23/result_best_model/examples_duet_24k/ex_13/s1.wav", sr=24000)  # 2 seconds of random noise
    processed_speech1,_ = librosa.load("Experiments/checkpoint/fin_van1/20250716_23/result_best_model/examples_duet_24k/ex_13/s0_estimate.wav",sr=24000)  # 2 seconds of random noise
    processed_speech2,_ = librosa.load("Experiments/checkpoint/fin_van1/20250716_23/result_best_model/examples_duet_24k/ex_13/s1_estimate.wav",sr=24000)  # 2 seconds of random noise
    ssnr = compute_ssnr(clean_speech1, processed_speech1, clean_speech2, processed_speech2)
    snr,ssnr_whole= compute_whole_ssnr(clean_speech1, processed_speech1, clean_speech2, processed_speech2)
    print("Segmental SNR:", ssnr.mean())
    print("SNR:", snr)
    print("Segmental SNR (whole):", ssnr_whole.mean())