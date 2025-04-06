import pandas as pd
import evaluate
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Parameters
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
train_csv = "/home/vellard/playlist_continuation/binary/train_pairs.csv"
val_csv = "/home/vellard/playlist_continuation/binary/val_pairs.csv"
output_dir = "./fine_tuned_binary_pairs"
batch_size = 8
epochs = 50
learning_rate = 2e-5
warmup_steps = 100

#transformation in dataframes
train_df = pd.read_csv(train_csv, header=0, names=["TitleA", "TitleB", "Label"])
val_df = pd.read_csv(val_csv, header=0, names=["TitleA", "TitleB", "Label"])

train_df["TitleA"] = train_df["TitleA"].astype(str)
train_df["TitleB"] = train_df["TitleB"].astype(str)
train_df["Label"] = train_df["Label"].astype(int)

val_df["TitleA"] = val_df["TitleA"].astype(str)
val_df["TitleB"] = val_df["TitleB"].astype(str)
val_df["Label"] = val_df["Label"].astype(int)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Create inputs (binary pairs)
def tokenize_function(examples):
    return tokenizer(
        examples["TitleA"],
        examples["TitleB"],
        truncation=True,
        padding="max_length"
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

tokenized_train = tokenized_train.rename_column("Label", "labels")
tokenized_val = tokenized_val.rename_column("Label", "labels")

tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",# Evaluate at the end of each epoch
    save_strategy="epoch",
    learning_rate=learning_rate,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    warmup_steps=warmup_steps,
    logging_strategy="epoch"
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# Save the metrics in a json file
with open(f"{output_dir}/trainer_metrics.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=4)

print(f"Model fine-tuned and saved to {output_dir}")
