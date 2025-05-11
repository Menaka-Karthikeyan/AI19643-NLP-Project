import os
from dotenv import load_dotenv

# Load the environment variable from .env file
load_dotenv()

# Read the API Keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = os.getenv("NEWS_API_URL")

CLAIMBUSTER_API_KEY = os.getenv("CLAIMBUSTER_API_KEY")
CLAIMBUSTER_API_URL = os.getenv("CLAIMBUSTER_API_URL")

if not NEWS_API_KEY:
    raise EnvironmentError("❌ NEWS_API_KEY not found. Set it in .env file.")

if not NEWS_API_URL:
    raise EnvironmentError("❌ NEWS_API_URL not found. Set it in .env file.")

if not CLAIMBUSTER_API_KEY:
    raise EnvironmentError("❌ CLAIMBUSTER_API_KEY not found. Set it in .env file.")

if not CLAIMBUSTER_API_URL:
    raise EnvironmentError("❌ CLAIMBUSTER_API_URL not found. Set it in .env file.")