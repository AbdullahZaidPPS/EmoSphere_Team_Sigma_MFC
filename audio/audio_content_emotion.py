pip install SpeechRecognition

conda install pyaudio

import speech_recognition as sr

text=""

r=sr.Recognizer()

with sr.Microphone() as microphone:
  r.adjust_for_ambient_noise(microphone)
  print("listening to mic...")
  audio=r.listen(microphone)
  print('Done listening')
    
try:
  text=r.recognize_google(audio)
except:
  text="sorry couldn't understand"

print("Audio content:"+text)

!pip install transformers

from transformers import pipeline
import json
import random

def reply_message(text):
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    data=(classifier(text))
    sentiment=data[0][0]['label']
    return sentiment

print(reply_message(text))
