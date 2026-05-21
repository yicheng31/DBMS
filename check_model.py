"""
Run this to find which models your Gemini API key supports.
Usage: python check_models.py
"""
import os
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    print("ERROR: GEMINI_API_KEY not set in .env")
    exit(1)

for api_version in ["v1beta", "v1"]:
    print(f"\n── {api_version} ──")
    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version=api_version),
        )
        for m in client.models.list():
            actions = [str(a) for a in (getattr(m, "supported_actions", []) or [])]
            if any("generateContent" in a for a in actions):
                print(f"  CHAT:  {m.name}")
            if any("embedContent" in a for a in actions):
                print(f"  EMBED: {m.name}")
    except Exception as e:
        print(f"  Error: {e}")