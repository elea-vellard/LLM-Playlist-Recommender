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


#########################
# Functions' definition #
#########################

# Preprocessing of data
def process_title(title):
    title = title.lower()
    title = title.replace("#", "")  # Remove hashtags
    title = title.strip()  # Remove spaces at the beginning and end
    return title

# Loading playlists' titles in a list
def load_playlist_titles(file_path):
    titles = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            title = line.split(',')[1]  # The second column in the csv file is the playlist's name
            titles.append(process_title(title))
    return titles

# Load BERT model and tokenizer
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    return tokenizer, model

# Compute embeddings with BERT
def compute_playlist_embeddings_with_bert(tokenizer, model, titles):
    embeddings = []
    for title in titles:
        # Tokenize and encode the title
        inputs = tokenizer(title, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        
        # Forward pass through BERT
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the CLS token embedding for sentence-level representation
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)

    return np.array(embeddings)

# K-means algorithm to create clusters
def cluster_playlists(embeddings, num_clusters=10):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    clusters = kmeans.labels_
    return clusters, kmeans

# We use dimension reduction (t-SNE) to visualize clusters
def visualize_clusters(embeddings, clusters, num_clusters, epoch, lr):
    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.6)
    plt.colorbar(scatter, ticks=range(num_clusters), label='Cluster ID')
    plt.title(f"Playlist clusters (epoch={epoch}, learning rate={lr})")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    #plt.show()
    plt.savefig("/home/vellard/playlist_continuation/output-plots/output.svg")  # Enregistre le graphique sous forme de fichier PNG

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
# Using PCA #
#############

def main():
    # Define files' path
    playlist_titles_path = '/data/playlist_data/split_csvs/playlists_train.csv'

    # Fixed parameters
    num_clusters = 10

    # Load playlists' titles
    print("Loading playlist titles...")
    titles = load_playlist_titles(playlist_titles_path)

    print("Loading BERT model...")
    tokenizer, model = load_bert_model()

    print("Computing playlist embeddings with BERT...")
    embeddings = compute_playlist_embeddings_with_bert(tokenizer, model, titles)

    print("Reducing dimensionality with PCA...")
    reduced_embeddings = reduce_dimensionality(embeddings, n_components=3)

    print("Clustering playlists...")
    clusters, kmeans_model = cluster_playlists(reduced_embeddings, num_clusters=num_clusters)

    print("Visualizing clusters...")
    visualize_clusters(reduced_embeddings, clusters, num_clusters, epoch=None, lr=None)

    print("Evaluating clusters...")
    silhouette_avg, davies_bouldin_avg = evaluate_clustering(reduced_embeddings, clusters)
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

if __name__ == "__main__":
    main()

