from fer import FER
from gtts import gTTS 
import os
import cv2
import pygame
import tempfile


emotion_model = FER()

emotions_cumulative = {
    'angry': 0,
    'disgust': 0,
    'fear': 0,
    'happy': 0,
    'sad': 0,
    'surprise': 0,
    'neutral': 0
}

detected_emotions = []

emotion_suggestions = {
    'angry': "Take a deep breath and count to 10. Try to relax and find a way to express your feelings.",
    'disgust': "Focus on something positive and try to change your perspective.",
    'fear': "Identify your fears and try to confront them gradually. Seek support from someone you trust.",
    'happy': "Celebrate the moment and share your happiness with others. Do something you enjoy.",
    'sad': "It's okay to feel sad sometimes. Talk to a friend or engage in activities that bring you comfort.",
    'surprise': "Embrace the unexpected and stay open to new experiences. Life is full of surprises!",
    'neutral': "You seem calm and composed. Reflect on your day and set positive intentions for the future."
}


video_capture = cv2.VideoCapture(0)  

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    emotions = emotion_model.detect_emotions(frame)

    if emotions:
        detected_emotions.append(emotions[0]['emotions'])

    if detected_emotions:
        frame = cv2.putText(frame, str(detected_emotions[-1]), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  
        break

video_capture.release()
cv2.destroyAllWindows()

for emotions in detected_emotions:
    for emotion, score in emotions.items():
        emotions_cumulative[emotion] += score

final_emotion = max(emotions_cumulative, key=emotions_cumulative.get)

print("Final Emotion:", final_emotion)

suggestion = emotion_suggestions.get(final_emotion, "No specific suggestion available.")
print("Suggestion:", suggestion)

with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
    tts = gTTS(text=suggestion, lang='en')
    tts.save(temp_audio_file.name)

pygame.mixer.init()

pygame.mixer.music.load(temp_audio_file.name)
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    continue


