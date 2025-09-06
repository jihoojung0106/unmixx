import numpy as np
import librosa
import torch
import os
import random
def generate_random_string(length=10):
    """Generate a random string of fixed length."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return ''.join(random.choice(letters) for i in range(length))
def hz_to_cents(frequency_hz):
    """Convert frequency in Hz to 20-cent resolution scale, centered at A4 (440Hz)."""
    return np.round(60 * np.log2(frequency_hz / 440.0)) + 345  # so A4 = 345

def compute_overtones(f0_hz, n_overtones=16):
    """Compute overtone frequencies and convert to cents scale."""
    overtones = []
    for j in range(1, n_overtones + 1):
        overtone_hz = j * f0_hz
        overtone_cents = hz_to_cents(overtone_hz)
        overtones.append(overtone_cents)
    return np.array(overtones, dtype=int)

def cents_to_binary_vector(cents, vector_size=1000):
    """Convert list of cent values to binary vector (1 where harmonic exists)."""
    vec = np.zeros(vector_size, dtype=int)
    cents = cents[(cents >= 0) & (cents < vector_size)]
    vec[cents] = 1
    return vec

def harmonic_overlap_score(f0_track_1, f0_track_2, activity_mask=None):
    """
    f0_track_*: shape (T,), in Hz. Use 0 or np.nan for unvoiced.
    activity_mask: optional (T,) boolean mask where both sources are active.
    """
    n_frames = len(f0_track_1)
    total_overlap = 0
    total_active = 0

    for t in range(n_frames):
        f0_1 = f0_track_1[t]
        f0_2 = f0_track_2[t]

        if f0_1 <= 0 or f0_2 <= 0:
            continue
        if activity_mask is not None and not activity_mask[t]:
            continue

        cents_1 = compute_overtones(f0_1)
        cents_2 = compute_overtones(f0_2)
        bin_1 = cents_to_binary_vector(cents_1)
        bin_2 = cents_to_binary_vector(cents_2)

        total_overlap += np.dot(bin_1, bin_2)
        total_active += 1

    if total_active == 0:
        return 0.0
    return total_overlap / total_active
import numpy as np
import librosa
import soundfile as sf  # 저장용
import torch

def hz_to_cents(frequency_hz):
    """Convert frequency in Hz to 20-cent resolution scale, centered at A4 (440Hz)."""
    return np.round(60 * np.log2(frequency_hz / 440.0)) + 345  # so A4 = 345

def compute_overtones(f0_hz, n_overtones=16):
    """Compute overtone frequencies and convert to cents scale."""
    overtones = []
    for j in range(1, n_overtones + 1):
        overtone_hz = j * f0_hz
        overtone_cents = hz_to_cents(overtone_hz)
        overtones.append(overtone_cents)
    return np.array(overtones, dtype=int)

def cents_to_binary_vector(cents, vector_size=1000):
    """Convert list of cent values to binary vector (1 where harmonic exists)."""
    vec = np.zeros(vector_size, dtype=int)
    cents = cents[(cents >= 0) & (cents < vector_size)]
    vec[cents] = 1
    return vec

def harmonic_overlap_score(f0_track_1, f0_track_2, activity_mask=None):
    """
    f0_track_*: shape (T,), in Hz. Use 0 or np.nan for unvoiced.
    activity_mask: optional (T,) boolean mask where both sources are active.
    """
    n_frames = min(len(f0_track_1), len(f0_track_2))
    total_overlap = 0
    total_active = 0

    for t in range(n_frames):
        f0_1 = f0_track_1[t]
        f0_2 = f0_track_2[t]

        if f0_1 <= 0 or f0_2 <= 0:
            continue
        if activity_mask is not None and not activity_mask[t]:
            continue

        cents_1 = compute_overtones(f0_1)
        cents_2 = compute_overtones(f0_2)
        bin_1 = cents_to_binary_vector(cents_1)
        bin_2 = cents_to_binary_vector(cents_2)

        total_overlap += np.dot(bin_1, bin_2)
        total_active += 1

    if total_active == 0:
        return 0.0
    return total_overlap / total_active

# 1. Load audio files
path1 = "duet_svs/jaCappella/bossa_nova/haruyokoi/tenor_24K.wav"
path2 = "duet_svs/jaCappella/bossa_nova/haruyokoi/soprano_24K.wav"

y1, sr = librosa.load(path1, sr=22050)
y2, _ = librosa.load(path2, sr=22050)

# 2. Cut to the same length
min_len = min(len(y1), len(y2))
y1 = y1[:min_len]
y2 = y2[:min_len]

# 3. Compute f0
f0_1 = librosa.yin(y1, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
f0_2 = librosa.yin(y2, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))

# 4. Harmonic overlap score
score = harmonic_overlap_score(f0_1, f0_2)
print("🎵 Harmonic Overlap Score:", score)

# 5. Mix audio (simple average)
mixed = 0.5 * (y1 + y2)

# 6. Save
save_folder=f"result/harmony/{generate_random_string(3)}"
os.makedirs(save_folder, exist_ok=True)
sf.write(save_folder+"/mix.wav", mixed, sr)
print(save_folder+"/mix.wav")
