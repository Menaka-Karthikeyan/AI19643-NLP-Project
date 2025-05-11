from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer # type: ignore
from transformers.trainer_utils import set_seed # type: ignore
from nlp_utils import exaggerate_adjectives, inject_irony, punify
from filters import is_safe_and_accurate
import os

set_seed(42)

# Check if model exists before loading
model_dir = "./satire_model"
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model not found in directory: {model_dir}")

tokeniser = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)
tokeniser.pad_token = tokeniser.eos_token
model.eval()

def generate_satire(news_text, max_retries=5):
    if isinstance(news_text, dict):
        title = news_text.get("title", "").strip()
        description = news_text.get("description", "").strip()
        if not title and not description:
            return "[No valid input provided]"
    elif isinstance(news_text, str):
        title = ""
        description = news_text.strip()
        if not description:
            return "[No valid input provided]"
    else:
        return "[Unsupported input format]"

    processed_summary = inject_irony(exaggerate_adjectives(description))

    prompt = (
        "Rewrite the following news article as a satirical version. "
        "Preserve the key ideas, but make it humorous or ironic.\n\n"
        f"TITLE: {title}\n"
        f"SUMMARY: {processed_summary}\n"
        "SATIRICAL ARTICLE:"
    )

    inputs = tokeniser(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    keyword_set = set((title + " " + description).lower().split())

    for attempt in range(max_retries):
        try:
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.8,
                top_p=0.92,
                top_k=50,
                no_repeat_ngram_size=3,
                eos_token_id=tokeniser.eos_token_id,
                pad_token_id=tokeniser.eos_token_id,
                num_return_sequences=1
            )

            generated_text = tokeniser.decode(output[0], skip_special_tokens=True).strip()
            satirical_text = inject_irony(punify(generated_text))

            # Extract satire
            if "SATIRICAL ARTICLE:" in satirical_text:
                satire = satirical_text.split("SATIRICAL ARTICLE:")[-1].strip()
            else:
                satire = satirical_text.strip()

            if "." in satire:
                satire = satire.rsplit(".", 1)[0] + "."

            # Relevance check
            important_keywords = [word for word in keyword_set if len(word) > 4]
            matched_keywords = [kw for kw in important_keywords if kw in satire.lower()]
            if len(matched_keywords) < max(1, len(important_keywords) // 6):
                print(f"Rejected for low relevance (attempt {attempt+1})")
                continue

            if is_safe_and_accurate(satire):
                return satire
            else:
                print(f"Rejected for safety (attempt {attempt+1})")

        except Exception as e:
            print(f"Error generating satire (attempt {attempt+1}/{max_retries}): {e}")
            continue

    return "[Satire removed due to ethical or factual concerns.]"