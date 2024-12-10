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
        for row in reader:
            track_metadata[row["track_uri"]] = {
                "track_name": row["track_name"],
                "artist_name": row["artist_name"],
            }

    playlist_tracks = {}
    with open(items_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
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
    for pid, _ in similar_playlists:
        if pid in playlist_tracks:
            for track_metadata in playlist_tracks[pid]:
                song_counter[(track_metadata["track_name"], track_metadata["artist_name"])] += 1
    return song_counter.most_common(top_k)


#################
# Main Function #
#################
def main():
    model_dir = "/home/vellard/playlist_continuation/finetuning/finetuned-model"
    playlist_embeddings_file = "/home/vellard/playlist_continuation/embeddings/playlists_embeddings.pkl"
    items_csv = "/data/playlist_continuation_data/csvs/items.csv"
    tracks_csv = "/data/playlist_continuation_data/csvs/tracks.csv"
    playlists_csv = "/data/playlist_continuation_data/csvs/playlists.csv"

    model = load_fine_tuned_model(model_dir)

    playlist_embeddings = load_playlist_embeddings(playlist_embeddings_file)
    playlist_tracks = load_playlist_tracks_with_artists(items_csv, tracks_csv)

    # Input playlist
    playlist_name = input("Enter the playlist name: ")
    print(f"Generating recommendations for playlist: '{playlist_name}'...")

    playlist_titles = {}
    with open(playlists_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["pid"]
            title = row["name"]
            playlist_titles[pid] = title

    # Similar playlists
    top_playlists = find_similar_playlists(playlist_name, playlist_embeddings, model, top_k=100)

    print("\nTop Similar Playlists:")
    for i, (pid, similarity) in enumerate(top_playlists, start=1):
        if i > 10:
            break
        title = playlist_titles.get(pid, "Unknown Playlist Title")
        print(f"{i}. Playlist Title: {title}, Similarity: {similarity:.4f}")

    top_songs = get_top_songs_with_artists(top_playlists, playlist_tracks, top_k=10)

    print("\nTop Recommended Songs:")
    for i, ((song, artist), count) in enumerate(top_songs, start=1):
        print(f"{i}. Song: {song}, Artist: {artist}, Occurrences: {count}")


if __name__ == "__main__":
    main()
