from flask import Flask, render_template
from news_fetcher import fetch_news
from satire_generator import generate_satire

app = Flask(__name__)

@app.route("/")
def index():
    raw_news = fetch_news()
    if not raw_news:
        return "No news fetched. Please check API settings or logs."

    processed_originals = []
    satirical_versions = []

    # Enrich & summarize each article
    for item in raw_news:
        title = item.get("title", "").strip()
        description = (item.get("description") or "").strip()

        if not title or not description:
            continue  # skip invalid items

        # Combine original text
        original_text = f"Title: {title}\nDescription: {description}"
        satire = generate_satire(f"{title}. {description}")

        processed_originals.append(original_text)
        satirical_versions.append(satire)

    return render_template("index.html", news=zip(processed_originals, satirical_versions))

if __name__ == "__main__":
    app.run(debug=True)