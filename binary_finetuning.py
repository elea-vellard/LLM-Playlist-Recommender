import csv
import random
from itertools import combinations
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

# First, load the 158 clusters
def load_clusters(clusterscsv):
    playlist_title = {}
    playlist_cluster = {}
    with open(clusterscsv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="playlist"):
            playlist_title[row['Playlist ID'].strip()] = row['Playlist Title'].strip() #pid, title
            playlist_cluster[row['Playlist ID'].strip()] = row['Cluster ID'].strip() #pid, cluster ID
    return playlist_title, playlist_cluster

# Generate balanced pairs (for one positve pair, generate one negative pair)
def generate_pairs(playlist_title, playlist_cluster, max_pairs=2000):
    grouped_pids = {}
    for pid, cluster in playlist_cluster.items():# group playlists (pids) by cluster ID
        if cluster not in grouped_pids:
            grouped_pids[cluster] = []
        grouped_pids[cluster].append(pid) # {cluster ID, [pid1, pid2]}

    all_pids = list(playlist_title.keys())
    pairs = []

    for cluster_id, pids in tqdm(grouped_pids.items(), desc="Clusters"):
        if len(pids) < 2:
            continue # for (the two) playlists who only have 1 playlist (no positive pair)

        positive_pairs = list(combinations(pids, 2))[:max_pairs] # we only take 2k pairs max
        for pid1, pid2 in positive_pairs:
            pairs.append(InputExample(texts=[playlist_title[pid1], playlist_title[pid2]], label=1.0)) # Positve entry for the training


            other_clusters = [pid for pid in all_pids if playlist_cluster[pid] != cluster_id]
            pid_negative = random.choice(other_clusters)
            pairs.append(InputExample(texts=[playlist_title[pid1], playlist_title[pid_negative]], label=0.0)) # NEgative entry for the training
    return pairs

#Training
def train(pairs, output_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    loader = DataLoader(pairs, shuffle=True, batch_size=16)
    loss = losses.CosineSimilarityLoss(model)
    model.fit(train_objectives=[(loader, loss)], epochs=3, warmup_steps=100, show_progress_bar=True)
    model.save(output_path)

if __name__ == "__main__":
    csv_path = "/home/vellard/playlist_continuation/clusters/clusters_train.csv"
    output_path = "/home/vellard/playlist_continuation/binary/binary_model"
    titles, clusters = load_clusters(csv_path)
    pairs = generate_pairs(titles, clusters, max_pairs=2000)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = DataLoader(pairs, shuffle=True, batch_size=16)
    loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=3,
        warmup_steps=100,
        show_progress_bar=True
    )

    model.save(output_path)
    print(f"Model saved to: {output_path}")
