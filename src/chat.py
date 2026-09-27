"""
chat.py

Qwen2.5 7B streaming over the local Ollama server.

stream_chat()      asks the model and yields text pieces as they arrive.
warm_chat_model()  sends a single-token request to load weights onto the GPU.
asked_object()     reads a question and returns a CNN class name if the
                   operator is asking to see a specific object.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

# Ollama model tag. Must match what is registered on the local server.
CHAT_MODEL = "qwen2.5:7b-instruct-q4_K_M"
CHAT_URL = "http://127.0.0.1:11434/api/chat"

# The greeting typed on screen character by character when the window opens.
GREETING = (
    "I am Sentinel, version 1.0. "
    "I am a reconnaissance assistant built on three object detectors "
    "and the local language model Qwen2.5 7B. "
    "I watch the aerial demonstration. "
    "YOLO localizes each object. "
    "EfficientNet names the crop. "
    "I report frames seen so far and assess the situation when asked. "
    "A person remains in the decision. "
    "I am here to detect potential dangers."
)

# Instructions prepended to every request so the model stays in role.
SYSTEM_PROMPT = (
    "You are Sentinel, the assistant at the SENTINEL-X stand. "
    "SENTINEL-X is a technology demonstrator for aerial reconnaissance. "
    "A person remains in the decision. You are not a weapon system. "
    "Answer only from the video context below. "
    "While playback is still running, talk only about frames up to the current time. "
    "After playback is complete, you may summarize everything detected in the clip. "
    "This is a demonstration video, not a mapped zone. Do not invent places or coordinates. "
    "If asked for an assessment, say what you think from the categories and counts. "
    "For example, soldiers, artillery, and military vehicles together can look like a war zone. "
    "If little military equipment has been seen, say so. "
    "Do not recommend the use of weapons. "
    "Say what is in the frames. State each category, how many were in view together, and the total. "
    "Do not explain how the counts are made, which model produced them, or how the report is organized. "
    "Answer in English. You already introduced yourself with: "
    + GREETING
)

# A question containing any of these words may be asking to see a crop picture.
SHOW_WORDS = ("show", "picture", "image", "crop", "display", "open the", "see the", "see a")

# Maps spoken phrases onto CNN class names. Longer phrases come first so they match before
# a shorter substring would.
OBJECT_ALIASES = (
    ("military_warship",    ("warship", "war ship", "naval")),
    ("military_aircraft",   ("military aircraft", "aircraft", "airplane", "plane")),
    ("military_vehicle",    ("military vehicle", "vehicle", "tank", "apc")),
    ("camouflage_soldier",  ("camouflage soldier", "camouflaged soldier", "camouflage")),
    ("M. Rocket Launcher",  ("rocket launcher", "rocket")),
    ("Artillery",           ("artillery", "howitzer")),
    ("Missile",             ("missile",)),
    ("Radar",               ("radar",)),
    ("Soldier",             ("soldier", "soldiers", "infantry", "troops")),
    ("trench",              ("trench", "trenches")),
)


def stream_chat(history: list[dict[str, str]], context: str):
    """Ask Qwen on the local Ollama server and yield each text piece as it arrives."""
    # The video tally is the only context for this answer. No image is sent.
    system = SYSTEM_PROMPT + "\n\nVideo context:\n" + context
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "system", "content": system}] + history,
        "stream": True,
        "keep_alive": "30m",
        "options": {"temperature": 0.3, "num_ctx": 2048, "num_predict": 360},
    }
    req = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        for raw in response:
            if not raw.strip():
                continue
            data = json.loads(raw.decode("utf-8"))
            if data.get("error"):
                raise RuntimeError(str(data["error"]))
            piece = data.get("message", {}).get("content", "")
            if piece:
                yield piece
            if data.get("done"):
                break


def warm_chat_model() -> None:
    """Send a single-token request to load the 4-bit weights onto the GPU before the first question."""
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": "Bereit?"}],
        "stream": False,
        "keep_alive": "30m",
        "options": {"num_predict": 1, "num_ctx": 512},
    }
    req = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        response.read()


def asked_object(text: str) -> str | None:
    """Return a CNN class name when the operator asks to see a specific object, else None."""
    lowered = text.lower()
    # Only proceed when the question contains a word that suggests showing something.
    if not any(word in lowered for word in SHOW_WORDS):
        return None
    for label, words in OBJECT_ALIASES:
        for word in words:
            if word in lowered:
                return label
    return None
