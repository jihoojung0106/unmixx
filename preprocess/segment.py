from __future__ import print_function
import msaf
import librosa
import seaborn as sns

# Choose an audio file and listen to it
audio_file = "duet_svs/24k/k_multisinger/ba_00118_-2_s_s12_m_04.wav"
# Segment the file using the default MSAF parameters
boundaries, labels = msaf.process(audio_file)
print(boundaries)
# Sonify boundaries
sonified_file = "my_boundaries.wav"
sr = 44100
boundaries, labels = msaf.process(audio_file, sonify_bounds=True, 
                                  out_bounds=sonified_file, out_sr=sr)
print(labels)
# Listen to results
audio = librosa.load(sonified_file, sr=sr)[0]
# Save as new file (e.g., WAV)
import soundfile as sf

sf.write("my_boundaries_saved.wav", audio, sr)
