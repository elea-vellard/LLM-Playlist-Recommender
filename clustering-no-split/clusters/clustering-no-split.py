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
from tqdm import tqdm

#----------------------------------------#
#Apply K-means and write the clusters in a csv file

def cluster_playlists(playlist_embeddings, num_clusters, playlist_titles, playlist_tracks, output_file):
    embedding_matrix = np.array(list(playlist_embeddings.values()))
    pids = list(playlist_embeddings.keys())

    #K-means algorithm (pre-existing fucnctions)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
    cluster_labels = kmeans.fit_predict(embedding_matrix) #No PCA

    #csv file
    with open(output_file, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster ID", "Playlist ID", "Playlist Title", "Tracks"])#Header
        for pid, label in tqdm(zip(pids, cluster_labels), total=len(pids), desc="Clustering", unit="playlist"):
            writer.writerow([label, pid, playlist_titles.get(pid, ""), ";".join(playlist_tracks.get(pid, []))])

def main():
    embeddings_dir = "/home/vellard/playlist_continuation/embeddings"
    output_dir = "/home/vellard/playlist_continuation/clustering-no-split/clusters/200"
    os.makedirs(output_dir, exist_ok=True)

    embeddings_file = os.path.join(embeddings_dir, f"embeddings.pkl")
    output_file = os.path.join(output_dir, f"clusters.csv")

    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)

    playlist_embeddings = data["playlist_embeddings"]
    playlist_titles = data["playlist_titles"]
    playlist_tracks = data["playlist_tracks"]

    cluster_playlists(playlist_embeddings, num_clusters=200, playlist_titles=playlist_titles, playlist_tracks=playlist_tracks, output_file=output_file)
    
if __name__ == "__main__":
    main()  

    # Creation of three sets of clusters (on the already splitted sets)
'''
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
        cluster_playlists(playlist_embeddings, num_clusters=100, playlist_titles=playlist_titles, playlist_tracks=playlist_tracks, output_file=output_file)
'''
