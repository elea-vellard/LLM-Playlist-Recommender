import os
import numpy as np
import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm  # Progress bar

#########################
# Functions' definition #
#########################

# Preprocessing of data
def process_title(title):
    title = title.lower()
    title = title.replace("#", "")  # Remove hashtags
    title = title.strip()  # Remove spaces at the beginning and end
    return title

# Loading playlist titles into a dictionary
def load_playlist_titles(file_path):
    titles = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            pid, title = line.strip().split(',')[0], line.strip().split(',')[1]  # Assume first column is playlist ID, second is title
            titles[pid] = process_title(title)
    return titles

# Loading track titles and grouping them by playlist
def load_playlist_track_titles(items_file_path, tracks_file_path):
    # Load track titles
    track_titles = {}
    with open(tracks_file_path, 'r', encoding='utf8') as f:
        for line in f:
            track_uri, track_name, *_ = line.strip().split(',')
            track_titles[track_uri] = process_title(track_name)

    # Load track titles in a dictionary, linking them to their playlist
    playlist_tracks = {}
    with open(items_file_path, 'r', encoding='utf8') as f:
        for line in f:
            pid, _, track_uri = line.strip().split(',')
            if pid not in playlist_tracks:
                playlist_tracks[pid] = []
            if track_uri in track_titles:
                playlist_tracks[pid].append(track_titles[track_uri])
    return playlist_tracks

# Load BERT model and tokenizer
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

# Compute embeddings using both playlist and track titles
def compute_playlist_embeddings_with_bert(tokenizer, model, playlist_titles, playlist_tracks):
    embeddings = []

    print("Computing playlist embeddings...")
    for pid, title in tqdm(playlist_titles.items(), desc="Playlists Processed", unit="playlist"):
        # Embedding of the playlist title
        inputs = tokenizer(title, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            playlist_title_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

        # Mean embedding of all track titles in the playlist
        track_embeddings = []
        for track_title in playlist_tracks.get(pid, []):  # Skip if no tracks for this playlist
            inputs = tokenizer(track_title, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                track_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                track_embeddings.append(track_embedding)

        if track_embeddings:
            track_mean_embedding = np.mean(track_embeddings, axis=0)
            final_embedding = (playlist_title_embedding + track_mean_embedding) / 2
        else:
            final_embedding = playlist_title_embedding

        embeddings.append(final_embedding)

    return np.array(embeddings)

# K-means algorithm to create clusters
def cluster_playlists(embeddings, num_clusters=10):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    clusters = kmeans.labels_
    return clusters, kmeans

# We use dimension reduction (t-SNE) to visualize clusters
def visualize_clusters(embeddings, clusters, num_clusters, epoch=None, lr=None):
    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.6)
    plt.colorbar(scatter, ticks=range(num_clusters), label='Cluster ID')
    plt.title(f"Playlist clusters")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.savefig("/home/vellard/playlist_continuation/output-plots/output.svg")

# Using metrics to estimate the quality of the clusters
def evaluate_clustering(embeddings, clusters):
    silhouette_avg = silhouette_score(embeddings, clusters)  # Best case: close to 1
    davies_bouldin_avg = davies_bouldin_score(embeddings, clusters)  # Best case: as small as possible
    return silhouette_avg, davies_bouldin_avg

# Apply PCA for dimensionality reduction
def reduce_dimensionality(embeddings, n_components=200):
    pca = PCA(n_components=n_components, random_state=0)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

#############
# Main Code #
#############

def main():
    # Define file paths
    playlist_titles_path = '/data/playlist_data/split_csvs/playlists_train.csv'
    items_file_path = '/data/playlist_data/split_csvs/items_train.csv'
    tracks_file_path = '/data/playlist_data/csvs/tracks.csv'

    # Fixed parameters
    num_clusters = 10

    playlist_titles = load_playlist_titles(playlist_titles_path)
    playlist_tracks = load_playlist_track_titles(items_file_path, tracks_file_path)


    tokenizer, model = load_bert_model()

    embeddings = compute_playlist_embeddings_with_bert(tokenizer, model, playlist_titles, playlist_tracks)

    reduced_embeddings = reduce_dimensionality(embeddings, n_components=200)

    clusters, kmeans_model = cluster_playlists(reduced_embeddings, num_clusters=num_clusters)

    visualize_clusters(reduced_embeddings, clusters, num_clusters)

    silhouette_avg, davies_bouldin_avg = evaluate_clustering(reduced_embeddings, clusters)
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

if __name__ == "__main__":
    main()
