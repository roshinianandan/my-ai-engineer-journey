import os
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL", "llama3.2")
TEMPERATURE = 0.7
MAX_TOKENS = 500