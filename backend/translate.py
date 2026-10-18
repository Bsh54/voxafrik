"""Client for the translation service that makes flagged messages legible to moderators."""
import os
import html
import requests

TRANSLATE_URL = os.environ.get(
    "TRANSLATE_URL",
    "https://shadsai2api-cloudflare.shadobsh.workers.dev/v1/chat/completions",
)
HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer 1"}


def translate(text, target="en", source="auto"):
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
