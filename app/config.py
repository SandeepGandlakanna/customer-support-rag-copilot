
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

APP_API_KEY = os.getenv("APP_API_KEY")


if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from the .env file")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from the .env file")

if not APP_API_KEY:
    raise ValueError("APP_API_KEY is missing from the .env file")
