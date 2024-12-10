import os
import csv
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
from sentence_transformers import SentenceTransformer
from transformers import Trainer, TrainingArguments


# Dataset pour regrouper les chansons d'un cluster et calculer la moyenne de leurs embeddings
class ClusterEmbeddingDataset(Dataset):
    def __init__(self, csv_file, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.data = []
        self.sbert_model = SentenceTransformer(model_name)

        with open(csv_file, 'r', encoding='utf8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cluster_id = int(row["Cluster ID"])
                tracks = row["Tracks"].split(';')
                self.data.append((tracks, cluster_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tracks, label = self.data[idx]

        # Calculer les embeddings des chansons dans le cluster
        track_embeddings = self.sbert_model.encode(tracks, convert_to_numpy=True)

        # Calculer la moyenne des embeddings
        if len(track_embeddings) > 0:
            cluster_embedding = np.mean(track_embeddings, axis=0)
        else:
            embedding_dim = self.sbert_model.get_sentence_embedding_dimension()
            cluster_embedding = np.zeros(embedding_dim)

        return {
            "embeddings": torch.tensor(cluster_embedding, dtype=torch.float),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Modèle de classification basé sur Sentence-BERT
class SentenceBertForClassification(nn.Module):
    def __init__(self, embedding_dim, num_labels):
        super(SentenceBertForClassification, self).__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embedding_dim, num_labels)

    def forward(self, embeddings, labels=None):
        x = self.dropout(embeddings)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return (loss, logits) if loss is not None else logits


# Fine-tuning du modèle
def fine_tune_sentence_bert(train_csv, val_csv, output_dir, num_labels):
    train_dataset = ClusterEmbeddingDataset(train_csv)
    val_dataset = ClusterEmbeddingDataset(val_csv)

    embedding_dim = train_dataset.sbert_model.get_sentence_embedding_dimension()

    model = SentenceBertForClassification(embedding_dim=embedding_dim, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir=os.path.join(output_dir, "logs"),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")


# Évaluation du modèle
def evaluate_model(test_csv, model_dir, metrics_file, num_labels):
    test_dataset = ClusterEmbeddingDataset(test_csv)

    embedding_dim = test_dataset.sbert_model.get_sentence_embedding_dimension()

    # Charger directement le modèle sauvegardé
    model = SentenceBertForClassification(embedding_dim=embedding_dim, num_labels=num_labels)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=model_dir,
            logging_dir=os.path.join(model_dir, "logs"),
            per_device_eval_batch_size=8,
        ),
        eval_dataset=test_dataset,
    )

    metrics = trainer.evaluate()
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_file}")


# Main Function
def main():
    base_dir = "/home/vellard/playlist_continuation/clustering/clusters"
    train_csv = os.path.join(base_dir, "clusters_train.csv")
    val_csv = os.path.join(base_dir, "clusters_val.csv")
    test_csv = os.path.join(base_dir, "clusters_test.csv")
    output_dir = "/home/vellard/playlist_continuation/finetuning/finetuned-sbert-model-10"
    metrics_file = os.path.join(output_dir, "metrics.json")

    os.makedirs(output_dir, exist_ok=True)

    fine_tune_sentence_bert(train_csv, val_csv, output_dir, num_labels=50)
    evaluate_model(test_csv, output_dir, metrics_file, num_labels=50)


if __name__ == "__main__":
    main()
