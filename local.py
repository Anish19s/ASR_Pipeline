from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import sounddevice as sd
from scipy.io.wavfile import write
import librosa

device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

fs = 16000
seconds = 15

print("Recording will start...")

recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()

write("input.wav", fs, recording)

print("Recording complete")

audio, sr = librosa.load("input.wav", sr=16000)

input_features = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
).input_features.to(device)

with torch.no_grad():
    predicted_ids = model.generate(input_features,task="translate")

text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print("\nTranscript:")
print(text)