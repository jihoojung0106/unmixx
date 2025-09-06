import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import random
import string
def save_waveform_from_(waveform: torch.Tensor, filepath: str, sample_rate: int):
    """
    waveform: (C, T) or (T,) 텐서. float32 [-1, 1] 권장
    filepath: 저장할 경로 (예: "vis/output.wav")
    sample_rate: 샘플링 레이트 (예: 16000)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # (T,) → (1, T)로 reshape (torchaudio는 (C, T) 형태 요구)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # mono

    # float32 타입으로 변환 (torchaudio 요구사항)
    if waveform.dtype != torch.float32:
        waveform = waveform.float()

    # 저장
    torchaudio.save(filepath, waveform.cpu(), sample_rate)
    print(f"Waveform saved to: {filepath}")
def save_state_dict_keys(state_dict, filename):
    with open(filename, 'w') as f:
        for k in state_dict.keys():
            f.write(f"{k}\n")
def visualize_magnitude(spec_RI, save_path="magnitude.png", sample_idx=0):
    """
    spec_RI: torch.Tensor of shape (B, 2, F, T), real & imag
    sample_idx: which sample in the batch to visualize
    """
    # 1. Real / Imag 분리
    real = spec_RI[sample_idx, 0].cpu().numpy()  # (F, T)
    imag = spec_RI[sample_idx, 1].cpu().numpy()  # (F, T)

    # 2. Magnitude 계산
    mag = np.sqrt(real**2 + imag**2)  # (F, T)

    # 3. 시각화
    plt.figure(figsize=(8, 4))
    plt.imshow(mag, aspect='auto', origin='lower', cmap='viridis')
    plt.title(f"Magnitude Spectrogram - Sample {sample_idx}")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()

    # 4. 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Magnitude image saved to {save_path}")
def visualize_ridge(tf_transf, ridge, flip_plot=True, path='ridge.png'):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 4))

    tf_disp = np.abs(tf_transf)
    
    if flip_plot:
        tf_disp = np.flipud(tf_disp)  # 이미지 자체 뒤집기
        plt.imshow(tf_disp, aspect='auto', cmap='jet')

        for i, color in enumerate(['red', 'green']):
            flipped_ridge = tf_transf.shape[0] - 1 - ridge[:, i]  # y좌표 뒤집기
            plt.plot(np.arange(ridge.shape[0]), flipped_ridge, linestyle='--', color=color, label=f'Ridge {i+1}')
    else:
        plt.imshow(tf_disp, aspect='auto', cmap='jet')
        for i, color in enumerate(['red', 'green']):
            plt.plot(np.arange(ridge.shape[0]), ridge[:, i], linestyle='--', color=color, label=f'Ridge {i+1}')

    plt.axis('off')
    plt.legend(loc='upper right')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def visualize_salience_map_for_basicpitch(contour,path):
    """
    Visualize the salience map from the model output.
    
    Args:
        model_output (dict): Output from the model containing 'contour'.
    """
    # Extract the salience map (contour)
    # Assuming 'contour' is a 2D array of shape (num_bins, num_frames)
    # and represents the salience map
    if contour.ndim == 3:
        contour = contour[0]
    contour=contour.detach().cpu().numpy()
    #contour=contour[:,50:200]
    plt.figure(figsize=(12, 6))
    plt.imshow(contour, origin='lower', aspect='auto', cmap='magma')
    plt.colorbar(label='Salience')
    plt.xlabel('Frame')
    plt.ylabel('Frequency Bin')
    plt.title(f'{os.path.basename(path)}')
    plt.savefig(path)
    print(f"Saved salience map to {path}")
def visualize_salience_map_for_basicpitch_with_grid(contour, path, sample_rate=16000, hop_length=512):
    """
    Visualize the salience map by index (100~170) with horizontal lines at every index.

    Args:
        contour: torch.Tensor or np.ndarray of shape (140, T)
        path: output image path
    """
    if isinstance(contour, torch.Tensor):
        contour = contour.detach().cpu().numpy()
    elif contour.ndim == 3:
        contour = contour[0]
    start_index=20
    end_index=100
    # Crop to index 100~170
    contour_cropped = contour[start_index:end_index, :]  # 171 포함 안 되므로 100~170까지
    num_bins, num_frames = contour_cropped.shape

    # 시간 축 (초 단위)
    duration = num_frames * hop_length / sample_rate
    time_axis = np.linspace(0, duration, num=num_frames)

    # 시각화
    plt.figure(figsize=(12, 6))
    extent = [time_axis[0], time_axis[-1], start_index, end_index]
    plt.imshow(contour_cropped, origin='lower', aspect='auto', cmap='magma', extent=extent)

    # 수평선: 모든 index마다 촘촘하게
    for i in range(start_index, end_index):
        plt.axhline(y=i, color='white', linestyle='--', linewidth=0.3, alpha=0.4)

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency Bin Index')
    plt.title(os.path.basename(path))
    plt.colorbar(label='Salience')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved salience map to {path}")
def generate_random_string(length=3):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
def save_waveform(waveform,path,sr=16000):
    import soundfile as sf
    import numpy as np
    waveform=waveform.detach().cpu().numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, np.ravel(waveform), sr)
    print(f"Waveform saved to {path}")
def estim_src2transcription_ready(estim_src):
    '''
    estim_src:[B,T]
    return 
    '''
    src_tensor0 = estim_src / (estim_src.abs().max() + 1e-8)  # normalize [-1,1]
    src_tensor0 = src_tensor0.unsqueeze(1)   # (B,1,T)
    return src_tensor0
                            
def visualize_pitch_salience(salience, sr=22050, hop_size=256, fmin=16.76, bins_per_octave=60,path="pitch_salience.png"):
    """
    salience: torch.Tensor or np.ndarray of shape [F, T] or [1, F, T]
    """

    if isinstance(salience, torch.Tensor):
        salience = salience.detach().cpu().numpy()

    # If batch dim exists
    if salience.ndim == 3:
        salience = salience[0]
    
    plt.figure(figsize=(12, 6))
    plt.imshow(salience, origin="lower", aspect="auto", cmap="magma")
    plt.colorbar(label="Salience")
    plt.xlabel("Time (s)")
    plt.ylabel("MIDI Pitch")
    plt.title("Pitch Salience Map")
    #plt.ylim(40, 80)
    plt.tight_layout()
    
    plt.savefig(path)
    print(f"Saved pitch salience map to {path}")

def plot_piano_roll(piano_roll: torch.Tensor, save_path=None, title='Piano Roll', vmax=1.0):
    """
    piano_roll: [128, T] 또는 [B, 128, T]
    save_path: 저장 경로
    """
    if piano_roll.dim() == 3:
        piano_roll = piano_roll[0]  # 배치 중 첫 번째만 시각화

    piano_roll_np = piano_roll.detach().cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.imshow(piano_roll_np, origin='lower', aspect='auto', cmap='hot', interpolation='nearest', vmax=vmax)
    plt.colorbar(label='Intensity')
    plt.xlabel("Time")
    plt.ylabel("Pitch")
    plt.title(title)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        
def generate_random_string(length=3):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))