import requests
from transformers.pipelines import pipeline
from config import CLAIMBUSTER_API_KEY, CLAIMBUSTER_API_URL

# Load once (efficient)
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)

def is_factually_correct(text):
    """Check the factual accuracy of the text using ClaimBuster."""
    headers = {"x-api-key": CLAIMBUSTER_API_KEY}
    params = {"input_text": text}
    
    try:
        response = requests.get(CLAIMBUSTER_API_URL, headers=headers, params=params)  # type: ignore
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            score = data[0].get("score", 0)
            return score < 0.7  # Assume factually correct if score is below threshold
        else:
            return True
    except requests.exceptions.RequestException as e:
        print(f"ClaimBuster API Error: {e}")
        return True  # Default to assuming it's factually correct on error

def is_ethically_safe(text, threshold=0.7):
    """Check if the text is ethically safe using a toxicity model."""
    predictions = toxicity_classifier(text)[0]  # type: ignore
    for result in predictions:
        if result['label'] in ['toxic', 'insult', 'obscene', 'identity_hate'] and result['score'] > threshold: # type: ignore
            return False
    return True

def is_safe_and_accurate(text):
    """Check both factual accuracy and ethical safety of the content."""
    return is_factually_correct(text) and is_ethically_safe(text)