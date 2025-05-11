from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
import torch
import math
from datasets import Dataset # type: ignore
import random
import matplotlib.pyplot as plt

# Load satirical_data.txt and prepare validation dataset
with open("satirical_data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Filter empty or whitespace-only lines
lines = [line.strip() for line in lines if line.strip()]
random.shuffle(lines)
val_data = [{"text": line} for line in lines[-50:]]  # Use last 50 lines for evaluation
val_dataset = Dataset.from_list(val_data)

def evaluate_perplexity(model_name_or_path, dataset):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

    # Set pad token (important for GPT2)
    tokenizer.pad_token = tokenizer.eos_token

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # type: ignore
    model.eval()

    total_loss = 0
    count = 0

    for item in dataset:
        inputs = tokenizer(item["text"], return_tensors="pt", truncation=True, padding=True, max_length=128)

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Skip empty sequences
        if input_ids.shape[1] == 0:
            continue

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()
            count += 1

    if count == 0:
        return float("inf"), float("inf")  # Avoid divide by zero

    avg_loss = total_loss / count
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

# Evaluate pretrained distilgpt2
base_loss, base_ppl = evaluate_perplexity("distilgpt2", val_dataset)

# Evaluate fine-tuned model
fine_tuned_model_path = "./satire_model"
fine_loss, fine_ppl = evaluate_perplexity(fine_tuned_model_path, val_dataset)

# Print results
print("Base GPT-2 (distilgpt2):")
print(f"  Loss: {base_loss:.4f}")
print(f"  Perplexity: {base_ppl:.2f}")

print("\nFine-Tuned GPT-2:")
print(f"  Loss: {fine_loss:.4f}")
print(f"  Perplexity: {fine_ppl:.2f}")

# Plotting
labels = ['Base (Pre-trained)', 'Fine-tuned']
losses = [base_loss, fine_loss]
ppls = [base_ppl, fine_ppl]

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Loss Comparison
ax[0].bar(labels, losses, color=['gray', 'green'])
ax[0].set_title("Loss Comparison")
ax[0].set_ylabel("Loss")

# Perplexity Comparison
ax[1].bar(labels, ppls, color=['gray', 'green'])
ax[1].set_title("Perplexity Comparison")
ax[1].set_ylabel("Perplexity")

plt.suptitle("Pre-trained vs Fine-tuned GPT-2 on Satirical News")
plt.tight_layout()
plt.show()