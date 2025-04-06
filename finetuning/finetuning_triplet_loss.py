import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from tqdm.auto import tqdm


# Parameters
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
train_csv = '/home/vellard/playlist_continuation/clusters_train.csv'
val_csv = '/home/vellard/playlist_continuation/clusters_val.csv'
output_dir = './final_triplet_model'
batch_size = 8
epochs = 50
learning_rate = 2e-5

#transformation in dataframes
train_df = pd.read_csv(train_csv, low_memory=False)
val_df = pd.read_csv(val_csv, low_memory=False)

train_df['Cluster ID'] = train_df['Cluster ID'].astype(int)
val_df['Cluster ID'] = val_df['Cluster ID'].astype(int)

# Mapping of labels to a sequential mapping
# (some clusters have been removed during the cleaning of miscellaneous clusters
# resulting in a unsequential list of labels)
unique_train_labels = sorted(train_df['Cluster ID'].unique())
label_mapping = {orig_label: new_label for new_label, orig_label in enumerate(unique_train_labels)}

train_df['Mapped Label'] = train_df['Cluster ID'].map(label_mapping)
val_df = val_df[val_df['Cluster ID'].isin(label_mapping.keys())].copy()
val_df['Mapped Label'] = val_df['Cluster ID'].map(label_mapping)

# Create inputs for triplet loss
# One anchor, one positive pair, one negative pair
def create_input_examples(df):
    examples = []
    for _, row in df.iterrows():
        examples.append(InputExample(texts=[str(row['Playlist Title'])], label=row['Mapped Label']))
    return examples

train_examples = create_input_examples(train_df)
val_examples   = create_input_examples(val_df)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

model = SentenceTransformer(MODEL_NAME, device=device)

# Loss definition
train_loss = losses.BatchAllTripletLoss(model=model)

'''# Evaluator for validation
val_texts = [str(row['Playlist Title']) for _, row in val_df.iterrows()]
val_scores = [1.0] * len(val_texts)  # simplistic approach: each item with itself
evaluator = EmbeddingSimilarityEvaluator(val_texts, val_texts, val_scores)'''

# Fit
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    #evaluator=evaluator,
    epochs=epochs,
    warmup_steps=100,
    output_path=output_dir,
    optimizer_class=AdamW,
    optimizer_params={'lr': learning_rate},
    #evaluation_steps=100000,
    show_progress_bar=True
)

model.save(output_dir)
print(f"Model saved to: {output_dir}")
