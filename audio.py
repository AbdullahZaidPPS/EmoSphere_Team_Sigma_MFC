!pip install transformers
!pip install torch
!pip install soundfile
!pip install wavio
!pip install scripy

from transformers import Wav2Vec2Processor, Wav2Vec2ForMaskedLM
import torch
import wavio
import soundfile as sf

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-base-960h")
audio, sample_rate = sf.read("ENG_M.wav")

def predict_tone(audio_file):

  inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
  outputs = model(**inputs)
  predicted_tone_id = outputs.logits.argmax(-1).item()

  labels = ["neutral", "happy", "sad"]
  predicted_tone = labels[predicted_tone_id]
  return predicted_tone

print(predict_tone)
# Currently the emotion tags are'nt mapped with the predicted_tone_id , so it returns the id 
