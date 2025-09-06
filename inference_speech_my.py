import yaml
import os
import look2hear.models
import argparse
import torch
import torchaudio
import torchaudio.transforms as T # Added for resampling
from basic_pitch_torch.inference import predict
import numpy as np
import matplotlib.pyplot as plt
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
    if isinstance(contour, torch.Tensor):
        contour = contour.detach().cpu().numpy()
    plt.figure(figsize=(12, 6))
    plt.imshow(contour, origin='lower', aspect='auto', cmap='magma')
    plt.colorbar(label='Salience')
    plt.xlabel('Frame')
    plt.ylabel('Frequency Bin')
    plt.title('Pitch Contour (Salience Map)')
    plt.savefig(path)
    print(f"Saved salience map to {path}")
def make_pitch_salience_map(audio_path):
    model_output, midi_data, note_events = predict(audio_path)
    visualize_salience_map_for_basicpitch(model_output['contour'].T,audio_path.replace(".wav","_salience.png"))

parser = argparse.ArgumentParser()
# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Separate speech sources using Look2Hear TIGER model.")
parser.add_argument("--audio_path", default="sample_music/project.wav", help="Path to audio file (mixture).")
parser.add_argument("--output_dir", default="separated_audio", help="Directory to save separated audio files.")
parser.add_argument("--model_cache_dir", default="cache", help="Directory to cache downloaded model.")

# Parse arguments once at the beginning

args = parser.parse_args()

audio_path = args.audio_path

output_dir = args.output_dir

cache_dir = args.model_cache_dir
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load model

print("Loading TIGERALL model...")
# Ensure cache directory exists if specified
if cache_dir:
    os.makedirs(cache_dir, exist_ok=True)
# Load the pretrained model #TIGERMPNET2
model_class = getattr(look2hear.models, "TIGERALL")
model = model_class(
    sample_rate=24000,
    in_channels=256,
    num_blocks=8,
    num_sources=2,
    out_channels= 128,
    stride= 240,
    upsampling_depth= 5,
    win= 960,
)
ckpt_path="Experiments/checkpoint/fin_all_rev5/20250827_01/epoch=226.ckpt"
#ckpt_path="Experiments/checkpoint/mpnet4/20250721_14/last.ckpt"
model_name=ckpt_path.split("/")[-3]+ "_" + ckpt_path.split("/")[-1].split(".")[0]
state_dict = torch.load(ckpt_path, map_location='cpu')["state_dict"]  # or 'cuda'
try:
    model.load_state_dict(state_dict)
    print("한 번에 로드 성공")  
except :
    converted_state_dict = {}
    prefix = "audio_model."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]  # prefix 제거
            converted_state_dict[new_key] = v
    # 모델에 로드 (strict=False는 일부 키 mismatch 허용)
    model.load_state_dict(converted_state_dict, strict=True)
    print(set(converted_state_dict.keys()).difference(set(model.state_dict().keys())))
    print("한 번에 로드 실패, prefix 제거 후 로드 성공")    

#.from_pretrained("JusperLee/TIGER-speech", cache_dir=cache_dir)
model.to(device)
model.eval()

## --- Audio Loading and Preprocessing ---
# Define the target sample rate expected by the model (usually 16kHz for TIGER)

target_sr = 24000
print(f"Loading audio from: {audio_path}")
try:
    # Load audio and get its original sample rate
    waveform, original_sr = torchaudio.load(audio_path)
    
except Exception as e:
    print(f"Error loading audio file {audio_path}: {e}")
    exit(1)
print(f"Original sample rate: {original_sr} Hz, Target sample rate: {target_sr} Hz")

# Resample if necessary
if original_sr != target_sr:
    print(f"Resampling audio from {original_sr} Hz to {target_sr} Hz...")
    resampler = T.Resample(orig_freq=original_sr, new_freq=target_sr)
    waveform = resampler(waveform)
    print("Resampling complete.")
    
# Move waveform to the target device
audio = waveform.to(device)

# Prepare the input tensor for the model
# Model likely expects a batch dimension [B, T] or [B, C, T]
# Assuming input is mono or model handles channels; add batch dim
# If audio has channel dim [C, T], keep it. If it's just [T], add channel dim first.
if audio.dim() == 2:
    if audio.shape[0]==2:
        audio=audio[0].unsqueeze(0) # Add batch dimension -> [1, C, T]
elif audio.dim() == 1:
    audio = audio.unsqueeze(0) # Add channel dimension -> [1, T]

# Add batch dimension -> [1, C, T]
# The original audio[None] is equivalent to unsqueeze(0) on the batch dimension
audio_input = audio.unsqueeze(0).to(device)
print(f"Audio tensor prepared with shape: {audio_input.shape}")
#Experiments/checkpoint/fin_van1/20250716_23/result_epoch=266/examples_unison_24k/ex_28/s0_estimate.wav

# --- Speech Separation ---

# Create output directory if it doesn't exist

os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")
print("Performing separation...")

with torch.no_grad():
    # Pass the prepared input tensor to the model
    ests_speech = model(audio_input,istest=True)  #  Expected output shape: [B, num_spk, T]
if isinstance(ests_speech, tuple):
    if len(ests_speech) >8:
        ests_speech = ests_speech[1]
    elif len(ests_speech) == 2:
        ests_speech = ests_speech[1]
    elif len(ests_speech) == 3 or len(ests_speech) == 5:
        ests_speech = ests_speech[0]
    else:
        ests_speech = ests_speech[0]

if isinstance(ests_speech, dict):
    try:
        ests_speech_original = ests_speech['output_original']  # Keep a copy of the original output
        ests_speech = ests_speech['output_final']  # Extract the tensor if it's a dict
    except:
        ests_speech_original = ests_speech['audio_out_original']  # Keep a copy of the original output
        ests_speech = ests_speech['audio_out_final']  # Extract the tensor if it's a dict
    
else:
    ests_speech_original=ests_speech
ests_speech = ests_speech.squeeze(0)
ests_speech_original = ests_speech_original.squeeze(0) if 'output_original' in locals() else ests_speech
num_speakers = ests_speech.shape[0]
print(f"Separation complete. Detected {num_speakers} potential speakers.")
for i in range(num_speakers):
    output_filename = os.path.join(output_dir, model_name,os.path.basename(audio_path).replace(".wav",""),f"spk{i+1}.wav")
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)  # Ensure the directory exists
    speaker_track = ests_speech[i].unsqueeze(0).cpu() # Get the i-th speaker track and move to CPU
    speaker_track_original = ests_speech_original[i].unsqueeze(0).cpu() # Get the i-th speaker track and move to CPU
    print(f"Saving speaker {i+1} to {output_filename}")
    try:
        torchaudio.save(
            output_filename,
            speaker_track, # Save the individual track
            target_sr      # Save with the target sample rate
        )
        torchaudio.save(
            output_filename.replace(f"spk{i+1}", f"spk{i+1}_original"), # Save the original track
            speaker_track_original, # Save the individual track
            target_sr      # Save with the target sample rate
        )
    except Exception as e:
        print(f"Error saving file {output_filename}: {e}")
    make_pitch_salience_map(output_filename)
    make_pitch_salience_map(output_filename.replace(f"spk{i+1}", f"spk{i+1}_original"))
torchaudio.save(
    output_filename.replace(f"spk{i+1}", "mixture"), # Save the mixture
    audio_input[0].cpu(), # Save the individual track
    target_sr      # Save with the target sample rate
)
make_pitch_salience_map(output_filename.replace(f"spk{i+1}", "mixture"))
