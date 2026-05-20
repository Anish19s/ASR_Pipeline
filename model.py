from transformers import WhisperProcessor,WhisperForConditionalGeneration
import torch

processor=WhisperProcessor.from_pretrained("openai/whisper-small")
model=WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")


device="cuda" if torch.cuda.is_available() else "cpu"
model=model.to(device)



def transcribe(waveform,sampling_rate):

    waveform = waveform.float()

    if waveform.abs().max() > 0:
        waveform = waveform / waveform.abs().max()


    chunk_len=30*sampling_rate
    stride=5*sampling_rate
    chunks=[]

    for i in range(0,waveform.shape[0],chunk_len-stride):
        chunk=waveform[i:i+chunk_len]
        chunks.append(chunk)


    final_text=""

    for chunk in chunks:

        inputs=processor(chunk,sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
        input_features=inputs["input_features"].to(device)


        with torch.no_grad():
            predicted_ids=model.generate(input_features)


        transcription=processor.batch_decode(predicted_ids,skip_special_tokens=True)[0]

        final_text+=transcription+" "


    return final_text
