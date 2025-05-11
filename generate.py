from transformers.pipelines import pipeline
from transformers.trainer_utils import set_seed
from nlp_utils import exaggerate_adjectives, inject_irony, punify

generator = pipeline("text-generation", model="distilgpt2")

set_seed(42)

def generate_satire(news_text):
    processed = inject_irony(exaggerate_adjectives(news_text))
    prompt = f"Make this news funny and satirical:\n'{processed}'\nSatire:"

    output = generator(
        prompt,
        max_new_tokens=60,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        truncation=True,
        pad_token_id=50256
    )
    if isinstance(output, list) and isinstance(output[0], dict) and "generated_text" in output[0]:
        satire = output[0]['generated_text'].split("Satire:")[-1].strip()
        return punify(satire)
    else:
        return "[Error] Model did not return expected output format."