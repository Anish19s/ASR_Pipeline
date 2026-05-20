from fastapi import FastAPI,UploadFile
import librosa
import os
import torch

app=FastAPI()
UPLOADED_DIR="uploads"
os.makedirs(UPLOADED_DIR,exist_ok=True)

@app.get("/")
def home():
    return {"message": "ASR API Running"}

@app.post("/transcribe")
async def trans_audio(file:UploadFile):
    from model import transcribe

    file_path=f"{UPLOADED_DIR}/{file.filename}"

    with open(file_path,"wb") as f:
        f.write(await file.read())

    audio,sr=librosa.load(file_path,sr=16000)

    waveform=torch.tensor(audio)
    text=transcribe(waveform,sr)

    return {"transcription":text}

