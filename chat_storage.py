# chat_storage.py
import json
import os

CHAT_FILE = "chat_sessions.json"

def load_sessions():
    if not os.path.exists(CHAT_FILE):
        return {}
    with open(CHAT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_sessions(sessions):
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)

def add_message(session_id, role, content):
    sessions = load_sessions()
    if session_id not in sessions:
        sessions[session_id] = []
    sessions[session_id].append({"role": role, "content": content})
    save_sessions(sessions)