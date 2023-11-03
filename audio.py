!pip install transformers
!pip install soundfile
!pip install librosa
!pip install torch

import soundfile as sf
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import warnings
warnings.filterwarnings("ignore")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

audio_data, sample_rate = sf.read("harvard.wav")

audio_array = np.array(audio_data)

# Resample
resampled_audio = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

# Extract features
audio_features = processor(resampled_audio,
                           sampling_rate=16000,
                           return_tensors="pt").input_values

# Pad if needed
min_size = 100
if audio_features.shape[1] < min_size:
  pad_amt = min_size - audio_features.shape[1]
  audio_features = torch.nn.functional.pad(audio_features, (0, pad_amt))

# Truncate
max_len = 4000
if audio_features.shape[1] > max_len:
  audio_features = audio_features[:,:max_len]

predictions = model(audio_features)

print(predictions.logits.argmax())
