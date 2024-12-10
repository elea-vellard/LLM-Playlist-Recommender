import os
import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

#########################
# Functions' definition #
#########################

def load_playlist_titles(file_path):
    titles = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            pid, title = line.strip().split(',')[0], line.strip().split(',')[1]
            titles[pid] = title
    return titles

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

def compute_track_embeddings(model, playlist_tracks):
    unique_tracks = list(set(title for tracks in playlist_tracks.values() for title in tracks))
    track_embeddings_array = model.encode(unique_tracks, show_progress_bar=True, convert_to_numpy=True)
    track_embeddings = {track_title: emb for track_title, emb in zip(unique_tracks, track_embeddings_array)}
    return track_embeddings

def compute_playlist_embeddings(playlist_tracks, track_embeddings):
    playlist_embeddings = {}
    for pid, tracks in tqdm(playlist_tracks.items(), desc="Processing playlists", unit="playlist"):
        valid_embeddings = [track_embeddings[track] for track in tracks if track in track_embeddings]
        if valid_embeddings:
            final_embedding = np.mean(valid_embeddings, axis=0)
        else:
            # Si pas de chansons, vecteur de zéros
            embedding_dim = next(iter(track_embeddings.values())).shape[0]
            final_embedding = np.zeros(embedding_dim)
        playlist_embeddings[pid] = final_embedding
    return playlist_embeddings

def apply_pca(embeddings, n_components=50):
    pca = PCA(n_components=n_components, random_state=0)
    reduced_embeddings = pca.fit_transform(embeddings)
    print(f"PCA reduced embeddings to {n_components} dimensions")
    return reduced_embeddings

def cluster_playlists_dbscan(playlist_embeddings, playlist_titles, playlist_tracks, output_file, eps=5.0, min_samples=5):
    # Convert playlist embeddings to a matrix
    embedding_matrix = np.array(list(playlist_embeddings.values()))
    pids = list(playlist_embeddings.keys())

    # Apply PCA
    reduced_embeddings = apply_pca(embedding_matrix, n_components=50)

    # DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(reduced_embeddings)

    # Save cluster results
    with open(output_file, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster ID", "Playlist ID", "Playlist Title", "Tracks"])
        for pid, label in zip(pids, cluster_labels):
            writer.writerow([label, pid, playlist_titles.get(pid, ""), ";".join(playlist_tracks.get(pid, []))])
    print(f"Clusters saved to {output_file}")

    # Apply t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=0)
    tsne_embeddings = tsne.fit_transform(reduced_embeddings)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_embeddings[:,0], tsne_embeddings[:,1], c=cluster_labels, cmap='tab10', s=20)
    plt.colorbar()
    plt.title('t-SNE Visualization of DBSCAN Clusters')

    # Save the plot
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    plot_file = os.path.join(os.path.dirname(output_file), f"tsne_{base_name}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"t-SNE plot saved to {plot_file}")

###################
# Main Clustering #
###################

def main():
    base_dir = "/data/playlist_continuation_data/split_csvs"
    tracks_file = "/data/playlist_continuation_data/csvs/tracks.csv"
    output_dir = "/home/vellard/playlist_continuation/clustering/clusters/DBscan"
    os.makedirs(output_dir, exist_ok=True)

    # Charger le modèle Sentence-BERT
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

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
        track_embeddings = compute_track_embeddings(model, playlist_tracks)

        # Compute playlist embeddings (moyenne des embeddings de leurs chansons)
        playlist_embeddings = compute_playlist_embeddings(playlist_tracks, track_embeddings)

        # Cluster playlists with DBSCAN and visualize
        cluster_playlists_dbscan(playlist_embeddings, playlist_titles, playlist_tracks, output_file, eps=2, min_samples=5)

if __name__ == "__main__":
    main()
