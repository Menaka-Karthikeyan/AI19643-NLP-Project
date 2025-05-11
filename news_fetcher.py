import requests
from config import NEWS_API_KEY, NEWS_API_URL
from nlp_utils import preprocess_news, summarise_text
from filters import is_safe_and_accurate
import time
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_news(category="world", language="en", max_articles=5):
    """Fetch the latest news articles from the API."""
    url = f"{NEWS_API_URL}?apikey={NEWS_API_KEY}&category={category}&language={language}"
    
    retries = 3
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json().get("results", [])[:max_articles]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error("Max retries reached, returning empty list.")
                return []

def is_valid_field(text):
    """Check if text is non-empty and does not contain 'ONLY AVAILABLE' placeholder."""
    return bool(text) and "only available" not in text.lower()

def process_and_summarise_news(news_items):
    """Process, enrich, check, and summarize the news articles."""
    formatted_news = []

    for item in news_items:
        title = item.get("title", "").strip()
        raw_description = (item.get("description") or "").strip()
        description = summarise_text(raw_description, max_words=25)

        if not (is_valid_field(title) and is_valid_field(description)):
            continue  # Skip if title or description is invalid

        # Combine only title and description
        full_text = f"{title}. {description}"

        # Preprocess the news text
        processed_text = preprocess_news(full_text)

        # Check if the processed text is safe and accurate
        if is_safe_and_accurate(processed_text):
            # Summarize the text after it is processed
            satire = summarise_text(processed_text, max_words=50)

            # Final structured output
            formatted = f"""Original:
Title: {title}
Description: {description}

Satire:
{satire}
"""
            formatted_news.append(formatted)

    return formatted_news