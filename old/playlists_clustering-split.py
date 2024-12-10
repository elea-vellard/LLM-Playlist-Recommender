import os
import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel

#########################
# Functions' definition #
#########################

# Load playlist titles into a dictionary
def load_playlist_titles(file_path):
    titles = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            pid, title = line.strip().split(',')[0], line.strip().split(',')[1]
            titles[pid] = title
    return titles

# Load track titles and group them by playlist
def load_playlist_track_titles(items_file_path, tracks_file_path):
    track_titles = {}
    with open(tracks_file_path, 'r', encoding='utf8') as f:
        for line in f:
            track_uri, track_name, *_ = line.strip().split(',')
            track_titles[track_uri] = track_name

    playlist_tracks = {}
    with open(items_file_path, 'r', encoding='utf8') as f:
        for line in f:
            pid, _, track_uri = line.strip().split(',')
            if pid not in playlist_tracks:
                playlist_tracks[pid] = []
            if track_uri in track_titles:
                playlist_tracks[pid].append(track_titles[track_uri])
    return playlist_tracks

# Compute track embeddings using BERT
def compute_track_embeddings(tokenizer, model, playlist_tracks):
    track_embeddings = {}
    unique_tracks = set(title for tracks in playlist_tracks.values() for title in tracks)

    for track_title in tqdm(unique_tracks, desc="Processing tracks"):
        inputs = tokenizer(track_title, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            track_embeddings[track_title] = embedding
    return track_embeddings

# Compute playlist embeddings (weighted average with playlist title embedding)
def compute_playlist_embeddings(tokenizer, model, playlist_titles, playlist_tracks, track_embeddings):
    playlist_embeddings = {}
    for pid, title in tqdm(playlist_titles.items(), desc="Processing playlists", unit="playlist"):
        # Compute embedding of the playlist title
        inputs = tokenizer(title, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            playlist_title_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

        # Compute mean embedding of the tracks in the playlist
        track_embeds = [track_embeddings[track] for track in playlist_tracks.get(pid, []) if track in track_embeddings]
        if track_embeds:
            track_mean_embedding = np.mean(track_embeds, axis=0)
            # Weighted average: Playlist title embedding gets more weight
            final_embedding = (playlist_title_embedding + track_mean_embedding) / 2
        else:
            final_embedding = playlist_title_embedding  # If no tracks, use only the playlist title embedding

        playlist_embeddings[pid] = final_embedding
    return playlist_embeddings

# Apply PCA for dimensionality reduction
def apply_pca(embeddings, n_components=50):
    pca = PCA(n_components=n_components, random_state=0)
    reduced_embeddings = pca.fit_transform(embeddings)
    print(f"PCA reduced embeddings to {n_components} dimensions")
    return reduced_embeddings

# Perform K-Means clustering
def cluster_playlists(playlist_embeddings, num_clusters, playlist_titles, playlist_tracks, output_file):
    # Convert playlist embeddings to a matrix
    embedding_matrix = np.array(list(playlist_embeddings.values()))
    pids = list(playlist_embeddings.keys())

    # Apply PCA
    reduced_embeddings = apply_pca(embedding_matrix, n_components=50)

    # K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(reduced_embeddings)

    # Save cluster results
    with open(output_file, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster ID", "Playlist ID", "Playlist Title", "Tracks"])
        for pid, label in zip(pids, cluster_labels):
            writer.writerow([label, pid, playlist_titles[pid], ";".join(playlist_tracks.get(pid, []))])
    print(f"Clusters saved to {output_file}")

###################
# Main Clustering #
###################

def main():
    base_dir = "/data/playlist_continuation_data/split_csvs"
    tracks_file = "/data/playlist_continuation_data/csvs/tracks.csv"
    output_dir = "/home/vellard/playlist_continuation/clusters"
    os.makedirs(output_dir, exist_ok=True)

    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Process each split (train, val, test)
    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split} set...")
        playlist_file = os.path.join(base_dir, f"playlists_{split}.csv")
        items_file = os.path.join(base_dir, f"items_{split}.csv")
        output_file = os.path.join(output_dir, f"clusters_{split}.csv")

        # Load data
        playlist_titles = load_playlist_titles(playlist_file)
        playlist_tracks = load_playlist_track_titles(items_file, tracks_file)

        # Compute track embeddings
        track_embeddings = compute_track_embeddings(tokenizer, model, playlist_tracks)

        # Compute playlist embeddings (weighted average)
        playlist_embeddings = compute_playlist_embeddings(tokenizer, model, playlist_titles, playlist_tracks, track_embeddings)

        # Cluster playlists and save results
        cluster_playlists(playlist_embeddings, num_clusters=25, playlist_titles=playlist_titles, playlist_tracks=playlist_tracks, output_file=output_file)

if __name__ == "__main__":
    main()
