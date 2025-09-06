import librosa
import soundfile as sf
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np

from transformers import WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("0x2a34/trained_model")

from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained("0x2a34/trained_model")

from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("byoussef/whisper-large-v2-Ko", language="Korean", task="transcribe")

model = model.to(device)

def transcribe(path):
    speech_array, sampling_rate = sf.read(path)
    speech_array = librosa.resample(np.asarray(speech_array), orig_sr=sampling_rate, target_sr=16000)

    max_duration = 30  # 30 seconds
    sampling_rate = 16000
    max_samples = max_duration * sampling_rate

    # Split the audio into 30-second segments
    audio_segments = [
        speech_array[i:i + max_samples] for i in range(0, len(speech_array), max_samples)
    ]

    transcriptions = []

    for segment in audio_segments:
        input_features = feature_extractor(segment, sampling_rate=sampling_rate, return_tensors="pt").input_features
        predicted_ids = model.generate(inputs=input_features.to(device))[0]
        transcription = processor.decode(predicted_ids, skip_special_tokens=True)
        transcriptions.append(transcription)

    # Concatenate transcriptions from all segments
    result_transcription = ''.join(transcriptions)

    return result_transcription
def transcribe_with_timing(path):
    speech_array, sampling_rate = sf.read(path)
    speech_array = librosa.resample(np.asarray(speech_array), orig_sr=sampling_rate, target_sr=16000)

    max_duration = 30  # seconds
    sampling_rate = 16000
    max_samples = max_duration * sampling_rate

    audio_segments = [
        speech_array[i:i + max_samples] for i in range(0, len(speech_array), max_samples)
    ]

    results = []

    for segment in audio_segments:
        inputs = processor(segment, sampling_rate=sampling_rate, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                inputs["input_features"].to(device),
                return_timestamps=True
            )

        # word-level timestamp decoding
        result = processor.batch_decode(output, skip_special_tokens=True, return_timestamps="word")[0]
        results.append(result)

    return results

if __name__ == "__main__":
    

    audio_path ="duet_svs/004.k_multisinger/01.data/1.Training/original_data/01.╣▀╢є╡хR&B/A. │▓╝║/01. 20┤ы/░б├в└┌_s02/ba_05688_-4_a_s02_m_02.wav"
    transcription = transcribe_with_timing(audio_path)
    print(transcription)
    
    
    #whisperx hey.wav --language Korean
