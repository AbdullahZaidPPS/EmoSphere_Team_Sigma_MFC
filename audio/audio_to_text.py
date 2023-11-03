!pip install SpeechRecognition
!conda install pyaudio

import speech_recognition as sr

r=sr.Recognizer()

with sr.Microphone() as microphone:
  print("Start talking now")
  audio=r.listen()
  print('Done listening')
try:
  print("Your text:"+r.recognize_google(audio))
except:
  print("sorry, couldn't understand")
