# ASR Pipeline using OpenAI Whisper

A complete Automatic Speech Recognition (ASR) pipeline built using OpenAI Whisper, covering everything from:
- Audio loading & preprocessing
- Chunked transcription for long audio
- Testing in Google Colab
- Live microphone transcription locally
- FastAPI deployment setup
- Swagger UI API testing
- Render deployment structure

This project was built to deeply understand how modern speech recognition systems work using Transformers and Whisper models.

---

## Features

- Audio transcription using Whisper
- Chunking support for long audio files
- Tested with pretrained Whisper models
- Live microphone transcription locally
- FastAPI backend for deployment
- Swagger UI interactive API docs
- Deployment-ready structure for Render
- Transformer-based ASR workflow exploration

---

## Tech Stack
- Python
- OpenAI Whisper
- Transformers
- PyTorch
- FastAPI
- Uvicorn
- Swagger UI
- Google Colab

---

## How It Works:
1️ Audio Input

The pipeline accepts:

- Audio files
- Live microphone input
  
2️ Preprocessing

Audio is:

- Loaded
- Converted
- Chunked for long-duration handling
  
3️ Whisper Inference

The processed chunks are passed through pretrained Whisper models for transcription.

4️ API Deployment

FastAPI exposes endpoints for:

-Uploading audio
-Running inference
-Returning transcribed text

---

## What I Learned

Through this project, I explored:

- Transformer-based speech recognition
- Whisper architecture & inference
- Chunking strategies for long audio
- API deployment workflows
- FastAPI backend development
- Real-time audio transcription
- Model serving basics
