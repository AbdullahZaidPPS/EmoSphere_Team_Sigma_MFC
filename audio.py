!pip install transformers
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

audio_data, sample_rate = sf.read("unseen_audio.wav")

audio_array = np.array(audio_data)

audio_features = processor(audio_array, sampling_rate=sample_rate)

predictions = model(audio_features)

print(predictions.logits.argmax())

