from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import json
import random

app = Flask(_name_)

classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
)

with open("sentiments.json", "r") as file:
    intents_data = json.load(file)


@app.route("/")
def chatbox():
    return render_template("chatbot.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    user_message = request.get_json().get("message", "")

    
    sentiment_result = classifier(user_message)
    sentiment = sentiment_result[0][0]["label"]

    
    intent = next(
        (item for item in intents_data["intents"] if item["tag"] == sentiment), None
    )
    if intent:
        responses = intent["response"]
        response = random.choice(responses)  
    else:
        response = "Sorry, I couldn't process your message."

    return jsonify({"answer": response})


if _name_ == "_main_":
    app.run(debug=True)
