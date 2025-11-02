import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

SAFETY_SYSTEM_PROMPT = (
    "You are MedMind, a supportive, empathetic mental health assistant. "
    "Your goal is to provide kind, non-judgmental, and practical support. "
    "Use simple, compassionate language. Encourage seeking professional help "
    "if the user expresses self-harm or harm to others. Avoid giving medical or legal advice. "
    "Offer grounding techniques, reframing, and validation. Keep responses concise and warm."
)

TONE_BY_MOOD = {
    "sad": "Use a soft, calm, and gently encouraging tone.",
    "angry": "Use short, clear, and logical language. Be steady and non-reactive.",
    "stressed": "Respond slowly and reassuringly. Break steps into small, manageable parts.",
    "happy": "Be energetic, positive, and concise without being overwhelming.",
}

if not API_KEY:
    # We don't raise here to allow app to start; requests will error with clear message
    pass

# Configure Gemini
if API_KEY:
    genai.configure(api_key=API_KEY)


def generate_reply(message: str, name: str, mood: str) -> str:
    if not API_KEY:
        return (
            "Server is missing GOOGLE_API_KEY. Please set it in your backend .env and restart."
        )

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        mood_key = (mood or "").strip().lower()
        tone_line = TONE_BY_MOOD.get(mood_key, "Use a balanced, warm, and supportive tone.")
        prompt = (
            f"{SAFETY_SYSTEM_PROMPT}\n\n"
            f"Tone guidance: {tone_line}\n\n"
            "--- Conversation Context ---\n"
            f"User name: {name or 'Unknown'}\n"
            f"Current mood: {mood or 'Neutral'}\n"
            f"User message: {message}"
        )
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None)
        if not text and hasattr(resp, "candidates") and resp.candidates:
            text = getattr(resp.candidates[0].content, "parts", [{}])[0].get("text")
        if not text:
            text = "I'm here with you. Let's take a slow breath together. Could you share a bit more?"
        return text
    except Exception as e:
        return (
            "I'm having trouble connecting to the AI service right now. "
            "Please try again in a moment."
        )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "has_api_key": bool(API_KEY)})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    name = (data.get("name") or "").strip()
    mood = (data.get("mood") or "").strip()

    if not message:
        return jsonify({"reply": "Please share a message so I can support you."}), 400

    reply = generate_reply(message, name, mood)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
