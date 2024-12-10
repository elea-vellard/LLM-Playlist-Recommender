import os
import csv
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords
import nltk

# Download NLTK stop words if not already downloaded
nltk.download('stopwords')

#########################
# Functions' definition #
#########################

# Preprocessing of data with stop word and custom word removal
def process_title(title):
    stop_words = set(stopwords.words('english'))  # Load English stop words
    custom_words = {'remastered', 'version', 'remix', 'radio', 'edit', 'feat', 'featuring', 'n°', '-', '/', 'live', '&', 'song', 'single', '(feat.', 'feat.', 'version)'}
    all_stop_words = stop_words.union(custom_words)
    title = title.lower()
    title = title.strip()
    processed_title = " ".join(word for word in title.split() if word not in all_stop_words)
    return processed_title

# Load playlist titles into a dictionary
def load_playlist_titles(file_path):
    titles = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            pid, title = line.strip().split(',')[0], line.strip().split(',')[1]
            titles[pid] = process_title(title)
    return titles

# Load track titles and group them by playlist
def load_playlist_track_titles(items_file_path, tracks_file_path):
    track_titles = {}
    with open(tracks_file_path, 'r', encoding='utf8') as f:
        for line in f:
            track_uri, track_name, *_ = line.strip().split(',')
            track_titles[track_uri] = process_title(track_name)

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

# Compute track embeddings using BERT
def compute_track_embeddings(tokenizer, model, playlist_tracks):
    track_embeddings = {}
    unique_tracks = set(title for tracks in playlist_tracks.values() for title in tracks)

    for track_title in tqdm(unique_tracks, desc="Tracks Processed", unit="track"):
        inputs = tokenizer(track_title, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            track_embeddings[track_title] = embedding
    return track_embeddings

# Compute mean playlist embeddings based on track embeddings
def compute_mean_playlist_embeddings(playlist_tracks, track_embeddings):
    playlist_embeddings = {}
    for pid, tracks in playlist_tracks.items():
        track_embeds = [track_embeddings[track] for track in tracks if track in track_embeddings]
        if track_embeds:
            playlist_embeddings[pid] = np.mean(track_embeds, axis=0)
    return playlist_embeddings

# Apply PCA for dimensionality reduction
def apply_pca(playlist_embeddings, n_components=50):
    embedding_matrix = np.array(list(playlist_embeddings.values()))
    pca = PCA(n_components=n_components, random_state=0)
    reduced_embeddings = pca.fit_transform(embedding_matrix)
    print(f"PCA reduced embeddings to {n_components} dimensions")
    return reduced_embeddings, list(playlist_embeddings.keys())

# Elbow Method to find the optimal number of clusters
def elbow_method(embeddings, max_clusters=15, output_path=None):
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    if output_path:
        plt.savefig(output_path)
        print(f"Elbow curve saved to {output_path}")
    else:
        plt.show()

# Cluster playlists using K-Means
def cluster_playlists_kmeans(embeddings, pids, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(embeddings)

    clustered_playlists = {}
    for pid, cluster in zip(pids, cluster_labels):
        if cluster not in clustered_playlists:
            clustered_playlists[cluster] = []
        clustered_playlists[cluster].append(pid)
    return clustered_playlists

# Visualize clusters using TSNE
def visualize_clusters(embeddings, cluster_labels, output_path):
    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.6)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title("Playlist Clusters")
    plt.savefig(output_path)
    print(f"Cluster visualization saved to {output_path}")

def analyze_cluster_themes(clustered_playlists, playlist_titles, playlist_tracks, output_file):
    cluster_data = []
    for cluster_id, pids in clustered_playlists.items():
        words = []
        for pid in pids:
            words.extend(playlist_titles[pid].split())
            words.extend(" ".join(playlist_tracks[pid]).split() if pid in playlist_tracks else [])
        word_counts = Counter(words).most_common(10)
        cluster_data.append({
            "Cluster ID": cluster_id,
            "Top Words": ", ".join([f"{word} ({count})" for word, count in word_counts]),
            "Number of Playlists": len(pids)
        })

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Cluster ID", "Top Words", "Number of Playlists"])
        writer.writeheader()
        writer.writerows(cluster_data)
    print(f"Cluster themes saved to {output_file}")

def save_clusters(clustered_playlists, playlist_titles, playlist_tracks, output_file_csv):
    with open(output_file_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # Write the header
        writer.writerow(["Cluster ID", "Playlist ID", "Playlist Title", "Tracks"])

        # Iterate over each cluster
        for cluster_id, pids in clustered_playlists.items():
            for pid in pids:
                playlist_title = playlist_titles.get(pid, "Unknown")
                tracks = playlist_tracks.get(pid, [])
                writer.writerow([
                    cluster_id,
                    pid,
                    playlist_title,
                    ", ".join(tracks)  # Join track titles into a single string
                ])
    print(f"Clusters saved to {output_file_csv}")


###################
# Main Clustering #
###################

def main():
    # Define file paths
    playlist_titles_path = '/data/playlist_continuation_data/split_csvs/playlists_train.csv'
    items_file_path = '/data/playlist_continuation_data/split_csvs/items_train.csv'
    tracks_file_path = '/data/playlist_continuation_data/csvs/tracks.csv'
    output_dir = "/home/vellard/playlist_continuation/output_k25"
    elbow_path = os.path.join(output_dir, "elbow_curve.svg")
    tsne_path = os.path.join(output_dir, "cluster_visualization.svg")
    cluster_themes_path = os.path.join(output_dir, "cluster_themes.csv")
    cluster_details_path = os.path.join(output_dir, "clusters.csv")

    playlist_titles = load_playlist_titles(playlist_titles_path)
    playlist_tracks = load_playlist_track_titles(items_file_path, tracks_file_path)

    tokenizer, model = load_bert_model()

    track_embeddings = compute_track_embeddings(tokenizer, model, playlist_tracks)

    playlist_embeddings = compute_mean_playlist_embeddings(playlist_tracks, track_embeddings)

    reduced_embeddings, pids = apply_pca(playlist_embeddings, n_components=50)

    #elbow_method(reduced_embeddings, max_clusters=200, output_path=elbow_path)

    num_clusters = 25
    clustered_playlists = cluster_playlists_kmeans(reduced_embeddings, pids, num_clusters)

    cluster_labels = [label for label in clustered_playlists for _ in clustered_playlists[label]]
    visualize_clusters(reduced_embeddings, cluster_labels, tsne_path)

    analyze_cluster_themes(clustered_playlists, playlist_titles, playlist_tracks, cluster_themes_path)

    save_clusters(clustered_playlists, playlist_titles, playlist_tracks, cluster_details_path)

if __name__ == "__main__":
    main()
