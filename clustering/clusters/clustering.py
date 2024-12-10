###################
# Code to cluster #
###################

import os
import csv
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def apply_pca(embeddings, n_components=20):
    pca = PCA(n_components=n_components, random_state=0)
    reduced_embeddings = pca.fit_transform(embeddings)
    print(f"PCA reduced embeddings to {n_components} dimensions")
    return reduced_embeddings

def cluster_playlists(playlist_embeddings, num_clusters, playlist_titles, playlist_tracks, output_file):
    embedding_matrix = np.array(list(playlist_embeddings.values()))
    pids = list(playlist_embeddings.keys())

    reduced_embeddings = apply_pca(embedding_matrix, n_components=20)

    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
    cluster_labels = kmeans.fit_predict(reduced_embeddings)

    # Save the clusters in .csv files
    with open(output_file, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster ID", "Playlist ID", "Playlist Title", "Tracks"])
        for pid, label in zip(pids, cluster_labels):
            writer.writerow([label, pid, playlist_titles.get(pid, ""), ";".join(playlist_tracks.get(pid, []))])
    print(f"Clusters saved to {output_file}")

    # Visualization
    tsne = TSNE(n_components=2, random_state=0)
    tsne_embeddings = tsne.fit_transform(reduced_embeddings)
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    plot_file = os.path.join(os.path.dirname(output_file), f"{base_name}_tsne.png")
    cmap = plt.get_cmap('nipy_spectral', num_clusters)
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1],
                          c=cluster_labels, cmap=cmap, s=50, alpha=0.5)  # alpha réduit pour plus de transparence
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster ID', rotation=270, labelpad=15) 
    unique_labels = np.unique(cluster_labels)
    cbar.set_ticks(unique_labels)
    cbar.set_ticklabels([str(lbl) for lbl in unique_labels])

    plt.title("Playlist Clusters")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    print(f"Cluster visualization saved to {plot_file}")

def main():
    embeddings_dir = "/home/vellard/playlist_continuation/clustering/embeddings"
    output_dir = "/home/vellard/playlist_continuation/clustering/clusters"
    os.makedirs(output_dir, exist_ok=True)

    # Creation of three sets of clusters (on the already splitted sets)
    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split} set...")
        embeddings_file = os.path.join(embeddings_dir, f"{split}_embeddings.pkl")
        output_file = os.path.join(output_dir, f"clusters_{split}.csv")

        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)

        playlist_embeddings = data["playlist_embeddings"]
        playlist_titles = data["playlist_titles"]
        playlist_tracks = data["playlist_tracks"]

        # Choose the number of clusters below
        cluster_playlists(playlist_embeddings, num_clusters=50, playlist_titles=playlist_titles, playlist_tracks=playlist_tracks, output_file=output_file)

if __name__ == "__main__":
    main()
