import string
import re
import html
import nltk  # type: ignore
import spacy
from textblob import TextBlob  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore
from nltk.corpus import stopwords  # type: ignore
from nltk.stem import WordNetLemmatizer  # type: ignore

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def summarise_text(text, max_words=50):
    """Summarise text by truncating to max_words."""
    if not isinstance(text, str):
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + '...'

def exaggerate_adjectives(text):
    """Exaggerates adjectives by prefixing them with 'super-'."""
    if isinstance(text, dict):
        text = text.get("text", "")
    if not isinstance(text, str):
        raise ValueError("Expected a string, but got: {}".format(type(text)))

    doc = nlp(text)
    exaggerated = []
    for token in doc:
        if token.pos_ == "ADJ":
            exaggerated.append("super-" + token.text)
        else:
            exaggerated.append(token.text)

    return " ".join(exaggerated)

def inject_irony(text):
    """Injects irony based on sentiment polarity."""
    if hasattr(text, "text"):  # if text is a spaCy Doc
        text = text.text
    if not isinstance(text, str):
        return ""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # type: ignore
    if polarity > 0.2:
        return "Clearly, the world needed more joy: " + text
    elif polarity < -0.2:
        return "This totally won't end badly: " + text
    return "Guess what happened now? " + text

def punify(text):
    """Injects puns into the text for certain keywords."""
    if not isinstance(text, str):
        return ""
    puns = {
        "politician": "poll-itician",
        "economy": "e-con-omy",
        "president": "prezz-ident",
        "inflation": "in-flab-tion",
    }
    for word, pun in puns.items():
        text = text.replace(word, pun)
    return text

def preprocess_news(news_text):
    """Preprocess the news text."""
    if not isinstance(news_text, str):
        return ""

    # 1. Unescape HTML entities and clean up non-printable characters
    news_text = html.unescape(news_text)
    news_text = news_text.encode("ascii", "ignore").decode("ascii")

    # 2. Lowercase
    news_text = news_text.lower()

    # 3. Remove URLs, emails, and extra spacing
    news_text = re.sub(r"http\S+|www\S+|https\S+", "", news_text)
    news_text = re.sub(r"\S+@\S+", "", news_text)
    news_text = re.sub(r"\s+", " ", news_text).strip()

    # 4. Remove punctuation
    news_text = news_text.translate(str.maketrans("", "", string.punctuation))

    # 5. Tokenize safely
    try:
        news_tokens = word_tokenize(news_text)
    except Exception:
        news_tokens = news_text.split()

    # 6. Remove stopwords
    stop_words = set(stopwords.words("english"))
    news_tokens = [word for word in news_tokens if word not in stop_words]

    # 7. Lemmatize
    lemmatiser = WordNetLemmatizer()
    news_tokens = [lemmatiser.lemmatize(word) for word in news_tokens]

    # 8. Reconstruct
    processed_news = " ".join(news_tokens)

    # 9. Enrich with exaggeration, irony, and puns
    processed_news = exaggerate_adjectives(processed_news)
    processed_news = inject_irony(processed_news)
    processed_news = punify(processed_news)

    return processed_news