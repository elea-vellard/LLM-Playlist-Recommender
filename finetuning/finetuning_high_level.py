# Code to finetune, adapted from Hugging Face tutorials

import pandas as pd
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

#Parameters
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
train_csv = '/home/vellard/malis/clustering-no-split/split/clusters_train.csv'
val_csv = '/home/vellard/malis/clustering-no-split/split/clusters_val.csv'
output_dir = './fine_tuned_model'
batch_size = 8
epochs = 5
learning_rate = 2e-5
warmup_steps = 100

# 1. Load data using pandas
train_df = pd.read_csv(train_csv, low_memory=False)
val_df = pd.read_csv(val_csv, low_memory=False)

# Convert cluster ids to integers
train_df['Cluster ID'] = train_df['Cluster ID'].astype(int)
val_df['Cluster ID'] = val_df['Cluster ID'].astype(int)

# 2. Create a maping for cluster IDs based on training data
unique_train_labels = sorted(train_df['Cluster ID'].unique())
label_mapping = {orig_label: new_label for new_label, orig_label in enumerate(unique_train_labels)}

# Map original labels to new sequential labels in training and validaton sets
train_df['Mapped Label'] = train_df['Cluster ID'].map(label_mapping)
#filter out validation rows with labels not present in training set
val_df = val_df[val_df['Cluster ID'].isin(label_mapping.keys())].copy()
val_df['Mapped Label'] = val_df['Cluster ID'].map(label_mapping)

# Determine the number of labels from mapping
num_labels = len(label_mapping)
print(f"Number of labels: {num_labels}")

# 3. Convert dataframes to Hugging Face Datasets using mapped labels
train_dataset = Dataset.from_pandas(train_df[['Playlist Title', 'Mapped Label']])
val_dataset = Dataset.from_pandas(val_df[['Playlist Title', 'Mapped Label']])

# 4. Load tokenizer and model for sequence classificaton
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

# 5. Tokenize the datasets with robust conversion to string
def tokenize_function(examples):
    texts = [str(text) for text in examples["Playlist Title"]]
    return tokenizer(texts, truncation=True, padding="max_length")

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# 6. Set format for PyTorch and rename label column to labels
tokenized_train = tokenized_train.rename_column("Mapped Label", "labels")
tokenized_val = tokenized_val.rename_column("Mapped Label", "labels")
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 7. Define training arguments with matching strategies for evaluation and saving
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    warmup_steps=warmup_steps,
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 9. Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

# 10. Train our model
trainer.train()

trainer.save_model(output_dir)
print(f"Model fine-tuned and saved to {output_dir}")
