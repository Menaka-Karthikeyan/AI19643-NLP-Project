import json
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import os
import math

# Load Trainer log
log_path = "./satire_model/checkpoint-9666/trainer_state.json"
if not os.path.exists(log_path):
    raise FileNotFoundError("trainer_state.json not found. Ensure training completed successfully.")

with open(log_path, 'r') as f:
    logs = json.load(f)

log_history = logs.get("log_history", [])

# Initialize lists
train_loss = []
eval_loss = []
eval_steps = []
train_steps = []

# Parse logs
for i, log in enumerate(log_history):
    if "loss" in log and "step" in log:
        train_loss.append(log["loss"])
        train_steps.append(log["step"])
    if "eval_loss" in log and "step" in log:
        eval_loss.append(log["eval_loss"])
        eval_steps.append(log["step"])

# Compute perplexity from eval loss
perplexity = [math.exp(l) if l < 100 else float("inf") for l in eval_loss]

# Plotting
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# Training Loss
if train_loss:
    plt.plot(train_steps, train_loss, label="Training Loss", marker='o')

# Evaluation Loss
if eval_loss:
    plt.plot(eval_steps, eval_loss, label="Validation Loss", marker='s')

# Perplexity
if perplexity:
    plt.plot(eval_steps, perplexity, label="Perplexity", marker='^')

plt.title("GPT-2 Fine-Tuning Performance Metrics")
plt.xlabel("Training Steps")
plt.ylabel("Loss / Perplexity")
plt.legend()
plt.tight_layout()
plt.show()