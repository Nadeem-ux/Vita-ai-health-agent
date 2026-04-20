import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HINDSIGHT_API_KEY = os.getenv("HINDSIGHT_API_KEY")
HINDSIGHT_COLLECTION_ID = os.getenv("HINDSIGHT_COLLECTION_ID")
USER_ID = os.getenv("USER_ID", "demo_user")

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
HINDSIGHT_BASE = "https://hindsight.vectorize.io/api"


def search_memories(query):
    if not HINDSIGHT_API_KEY or not HINDSIGHT_COLLECTION_ID:
        return ""
    try:
        resp = requests.post(
            HINDSIGHT_BASE + "/memory/search",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + HINDSIGHT_API_KEY
            },
            json={
                "collection_id": HINDSIGHT_COLLECTION_ID,
                "user_id": USER_ID,
                "query": query,
                "top_k": 5
            },
            timeout=10
        )
        if resp.ok:
            data = resp.json()
            memories = data.get("results") or data.get("memories") or []
            return "\n".join(m.get("content") or m.get("text") or "" for m in memories if m)
    except Exception as e:
        print("Hindsight search error: " + str(e))
    return ""


def save_memory(content, metadata=None):
    if not HINDSIGHT_API_KEY or not HINDSIGHT_COLLECTION_ID:
        return
    try:
        requests.post(
            HINDSIGHT_BASE + "/memory",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + HINDSIGHT_API_KEY
            },
            json={
                "collection_id": HINDSIGHT_COLLECTION_ID,
                "user_id": USER_ID,
                "content": content,
                "metadata": metadata or {}
            },
            timeout=10
        )
    except Exception as e:
        print("Hindsight save error: " + str(e))


def build_system_prompt(memories):
    if memories:
        memory_block = "---\n" + memories + "\n---"
    else:
        memory_block = "No previous health logs yet."

    return (
        "You are Vita, a warm and knowledgeable personal health companion AI.\n\n"
        "Relevant past health entries:\n"
        + memory_block +
        "\n\nGuidelines:\n"
        "- Be warm, supportive, and encouraging\n"
        "- Spot patterns across multiple days when you have the data\n"
        "- Give practical, actionable suggestions\n"
        "- Always recommend consulting a doctor for medical concerns\n"
        "- Keep responses concise (3-5 sentences unless a summary is requested)\n"
        "- Use **bold** for key insights\n"
        "- Never diagnose, you are a wellness companion not a doctor"
    )


def call_groq(system_prompt, messages):
    if not GROQ_API_KEY:
        return "Groq API key not configured. Please set GROQ_API_KEY in backend/.env"
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "max_tokens": 600,
        "temperature": 0.7
    }
    resp = requests.post(
        GROQ_ENDPOINT,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + GROQ_API_KEY
        },
        json=payload,
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = (data or {}).get("message", "").strip()
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    memories = search_memories(user_message)
    system_prompt = build_system_prompt(memories)

    try:
        reply = call_groq(system_prompt, [{"role": "user", "content": user_message}])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    today = datetime.now().strftime("%Y-%m-%d")
    convo = "User (" + today + "): " + user_message + "\nVita: " + reply[:300]
    save_memory(convo, {"type": "conversation", "date": datetime.now().isoformat()})

    return jsonify({"reply": reply})


@app.route("/log", methods=["POST"])
def log_health():
    data = request.get_json()
    log_entry = data or {}
    today = datetime.now().strftime("%A, %b %d")

    symptoms = log_entry.get("symptoms", ["none"])
    if not symptoms:
        symptoms = ["none"]

    log_text = (
        "Health log for " + today + ": "
        "Mood=" + log_entry.get("mood", "not logged") + ", "
        "Sleep=" + str(log_entry.get("sleep", 0)) + "hrs, "
        "Energy=" + str(log_entry.get("energy", 0)) + "/10, "
        "Symptoms=" + ", ".join(symptoms) + ", "
        "Exercise=" + log_entry.get("exercise", "not logged")
    )
    if log_entry.get("notes"):
        log_text += ", Notes: " + log_entry["notes"]

    meta = {"type": "health_log", "dateLabel": today}
    meta.update(log_entry)
    save_memory(log_text, meta)

    memories = search_memories("health patterns sleep mood energy")

    if memories:
        memory_block = "---\n" + memories + "\n---"
    else:
        memory_block = "No previous logs."

    system_prompt = (
        "You are Vita, a warm personal health companion AI.\n\n"
        "Past health context:\n" + memory_block + "\n\n"
        "Acknowledge the user health log warmly and briefly in 2-3 sentences. Be supportive, not medical."
    )

    try:
        reply = call_groq(system_prompt, [{"role": "user", "content": "I just logged: " + log_text}])
    except Exception as e:
        reply = "Thanks for logging your health today! Entry saved."

    return jsonify({"reply": reply, "logText": log_text})


@app.route("/memories", methods=["GET"])
def get_memories():
    if not HINDSIGHT_API_KEY or not HINDSIGHT_COLLECTION_ID:
        return jsonify({"memories": []})
    try:
        resp = requests.post(
            HINDSIGHT_BASE + "/memory/list",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + HINDSIGHT_API_KEY
            },
            json={
                "collection_id": HINDSIGHT_COLLECTION_ID,
                "user_id": USER_ID,
                "limit": 5
            },
            timeout=10
        )
        if resp.ok:
            data = resp.json()
            memories = data.get("results") or data.get("memories") or []
            return jsonify({"memories": memories})
    except Exception as e:
        print("List memories error: " + str(e))
    return jsonify({"memories": []})


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "groq": bool(GROQ_API_KEY),
        "hindsight": bool(HINDSIGHT_API_KEY and HINDSIGHT_COLLECTION_ID)
    })


if __name__ == "__main__":
    print("Vita backend starting on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)