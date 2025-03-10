import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

# Load Data Function
def load_data(src_path, tgt_path):
    with open(src_path, "r", encoding="utf-8") as src_file, open(tgt_path, "r", encoding="utf-8") as tgt_file:
        src_lines = [line.strip() for line in src_file.readlines()]
        tgt_lines = [line.strip() for line in tgt_file.readlines()]
    return [{"input_text": f"fix: {src}", "target_text": tgt} for src, tgt in zip(src_lines, tgt_lines)]

# Load and Merge Datasets
github_data = load_data("github-typos.train.src", "github-typos.train.tgt")
artificial_data = load_data("artificial.train.src", "artificial.train.tgt")
combined_data = github_data + artificial_data  # Merge datasets

dataset = Dataset.from_list(combined_data)

# Load tokenizer and model
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenize data
def preprocess_data(samples):
    inputs = tokenizer(samples["input_text"], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(samples["target_text"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

dataset = dataset.map(preprocess_data, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./t5_spell_corrector",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_strategy="epoch"
)

# Trainer setup
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,  # Use a separate validation set for real use
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train model
trainer.train()

# Save trained model
trainer.save_model("./t5_spell_corrector")
tokenizer.save_pretrained("./t5_spell_corrector")

print("✅ Model training complete! Saved at './t5_spell_corrector'.")
