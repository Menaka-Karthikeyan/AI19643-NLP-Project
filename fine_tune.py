from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
from datasets import Dataset # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

def fine_tune_model():
    model_name = "distilgpt2"
    tokeniser = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    tokeniser.pad_token = tokeniser.eos_token  # required for GPT2

    # Load and prepare the dataset
    with open("satirical_data.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Create a Dataset object
    data = [{"text": line.strip()} for line in lines]
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # Tokenise the dataset
    def tokenise_function(examples):
        return tokeniser(examples['text'], truncation=True, padding=True, max_length=128)

    tokenised_train = train_dataset.map(tokenise_function, batched=True, remove_columns=["text"])
    tokenised_val = val_dataset.map(tokenise_function, batched=True, remove_columns=["text"])

    # Prepare the data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokeniser, mlm=False
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./satire_model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,  # Save every 500 steps
        save_total_limit=1,
        logging_steps=100,
        prediction_loss_only=True,
        weight_decay=0.01,
        warmup_steps=500,  
        lr_scheduler_type="linear",
        save_strategy="steps",
    )

    # Initialise the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenised_train,
        eval_dataset=tokenised_val,
    )

    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
    else:
        # Save the model and tokeniser only if training succeeds
        trainer.save_model("./satire_model")
        tokeniser.save_pretrained("./satire_model")

if __name__ == "__main__":
    fine_tune_model()