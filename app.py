from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import json
import random

app = Flask(_name_)

# Load the sentiment classification model
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
)

# Load intent-response mappings from a JSON file
with open("sentiments.json", "r") as file:
    intents_data = json.load(file)


@app.route("/")
def chatbox():
    return render_template("chatbot.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    user_message = request.get_json().get("message", "")

    # Classify the sentiment of the user's message
    sentiment_result = classifier(user_message)
    sentiment = sentiment_result[0][0]["label"]

    # Find the intent with the specified sentiment
    intent = next(
        (item for item in intents_data["intents"] if item["tag"] == sentiment), None
    )
    if intent:
        responses = intent["response"]
        response = random.choice(responses)  # Randomly select a response
    else:
        response = "Sorry, I couldn't process your message."

    return jsonify({"answer": response})


if _name_ == "_main_":
    app.run(debug=True)
