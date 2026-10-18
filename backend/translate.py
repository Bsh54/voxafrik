"""Client for the translation service that makes flagged messages legible to moderators.

The service is a thin OpenAI-compatible proxy. Point it at your own endpoint with the
TRANSLATE_URL (and optional TRANSLATE_KEY) environment variables; see .env.example.
When no endpoint is configured, translation is skipped and the rest of the pipeline
keeps working.
"""
import os
import html
import requests

TRANSLATE_URL = os.environ.get("TRANSLATE_URL", "")
TRANSLATE_KEY = os.environ.get("TRANSLATE_KEY", "")
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {TRANSLATE_KEY}"}


def translate(text, target="en", source="auto"):
    if not TRANSLATE_URL:
        raise RuntimeError("TRANSLATE_URL is not configured")
    response = requests.post(
        TRANSLATE_URL,
        json={
            "model": "google-translate",
            "messages": [{"role": "user", "content": text}],
            "source_lang": source,
            "target_lang": target,
            "stream": False,
        },
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    return html.unescape(response.json()["choices"][0]["message"]["content"])
