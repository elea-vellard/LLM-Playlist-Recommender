#######################################
# Code to generate the recommendation #
#######################################

import os
import torch
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from collections import Counter
import csv


# Load the finetuned model
def load_fine_tuned_model(model_dir):
    model = SentenceTransformer(model_dir)
    model.eval()
    return model


# Compute the embedding of the input playlist
def get_playlist_embedding(playlist_name, model):
    embedding = model.encode(playlist_name, convert_to_numpy=True)
    return embedding


# Load precomputed embeddings
def load_playlist_embeddings(embeddings_file):
    with open(embeddings_file, 'rb') as f:
        playlist_embeddings = pickle.load(f)
    return playlist_embeddings


# Load tracks titles and their artist
def load_playlist_tracks_with_artists(items_csv, tracks_csv):
    track_metadata = {}
    with open(tracks_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading track metadata", unit="track"):
            track_metadata[row["track_uri"]] = {
                "track_name": row["track_name"],
                "artist_name": row["artist_name"],
            }

    playlist_tracks = {}
    with open(items_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading playlist tracks", unit="playlist"):
            pid = row["pid"]
            track_uri = row["track_uri"]
            if pid not in playlist_tracks:
                playlist_tracks[pid] = []
            if track_uri in track_metadata:
                playlist_tracks[pid].append(track_metadata[track_uri])

    return playlist_tracks


# Similar playlists
def find_similar_playlists(playlist_name, playlist_embeddings, model, top_k):
    playlist_embedding = get_playlist_embedding(playlist_name, model)

    similarities = []
    for pid, metadata in tqdm(playlist_embeddings.items(), desc="Scoring Playlists", unit="playlist"):
        similarity = cosine_similarity([playlist_embedding], [metadata["embedding"]])[0][0]
        similarities.append((pid, similarity))

    sorted_playlists = sorted(similarities, key=lambda x: x[1], reverse=True)
    return sorted_playlists[:top_k]


# Most occurent songs among the relevant playlists
def get_top_songs_with_artists(similar_playlists, playlist_tracks, top_k=10):
    song_counter = Counter()
    for pid, _ in tqdm(similar_playlists, desc="Counting songs", unit="playlist"):
        if pid in playlist_tracks:
            for track_metadata in playlist_tracks[pid]:
                song_counter[(track_metadata["track_name"], track_metadata["artist_name"])] += 1
    return song_counter.most_common(top_k)


# Evaluation metrics
def compute_metrics(recommended_songs, relevant_songs, top_n=10):
    # Compute HIT@N
    relevant_hits = [song for song in recommended_songs[:top_n] if song in relevant_songs]
    hits = len(relevant_hits)
    hit_score = hits / min(top_n, len(relevant_songs))  # Normalized

    # Compute Precision@N
    precision = hits / len(recommended_songs[:top_n])

    # Compute Recall@N
    recall = hits / len(relevant_songs)

    # Compute MRR@N
    mrr = 0.0
    for i, song in enumerate(recommended_songs[:top_n]):
        if song in relevant_songs:
            mrr = 1 / (i + 1)
            break

    print(f"HIT@{top_n}: {hit_score:.4f}")
    print(f"Precision@{top_n}: {precision:.4f}")
    print(f"Recall@{top_n}: {recall:.4f}")
    print(f"MRR@{top_n}: {mrr:.4f}")

    return {"HIT@N": hit_score, "Precision@N": precision, "Recall@N": recall, "MRR@N": mrr}


#################
# Main Function #
#################
def main():
    model_dir = "/home/vellard/playlist_continuation/finetuning/fine_tuned_model"
    playlist_embeddings_file = "/home/vellard/playlist_continuation/embeddings/new-model/playlists_embeddings.pkl"
    items_csv = "/data/playlist_continuation_data/csvs/items.csv"
    tracks_csv = "/data/playlist_continuation_data/csvs/tracks.csv"
    playlists_csv = "/data/playlist_continuation_data/csvs/playlists.csv"

    model = load_fine_tuned_model(model_dir)

    playlist_embeddings = load_playlist_embeddings(playlist_embeddings_file)
    playlist_tracks = load_playlist_tracks_with_artists(items_csv, tracks_csv)

    # Input playlist PID
    pid = input("Enter the playlist PID: ")
    print(f"Generating recommendations for playlist PID: '{pid}'...")

    playlist_titles = {}
    with open(playlists_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading playlist titles", unit="playlist"):
            playlist_titles[row["pid"]] = row["name"]

    playlist_name = playlist_titles.get(pid, "Unknown Playlist Title")
    if playlist_name == "Unknown Playlist Title":
        print("Invalid PID entered.")
        return

    # Similar playlists
    top_playlists = find_similar_playlists(playlist_name, playlist_embeddings, model, top_k=50)

    print("\nTop Similar Playlists:")
    for i, (similar_pid, similarity) in enumerate(top_playlists, start=1):
        title = playlist_titles.get(similar_pid, "Unknown Playlist Title")
        print(f"{i}. Playlist Title: {title}, Similarity: {similarity:.4f}")

    top_songs = get_top_songs_with_artists(top_playlists, playlist_tracks, top_k=66)

    print("\nTop Recommended Songs:")
    for i, ((song, artist), count) in enumerate(top_songs, start=1):
        print(f"{i}. Song: {song}, Artist: {artist}, Occurrences: {count}")

    # Metrics
    relevant_songs = playlist_tracks.get(pid, [])
    relevant_songs = list(set((track["track_name"], track["artist_name"]) for track in relevant_songs))  # Remove duplicates

    recommended_songs = [song_artist for song_artist, _ in top_songs]  # Format for comparison

    # Compute metrics
    metrics = compute_metrics(recommended_songs, relevant_songs, top_n=66)

    print(f"\nRecommended Songs:", recommended_songs)
    print("Relevant Songs:", relevant_songs)


if __name__ == "__main__":
    main()
