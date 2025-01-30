#######################################
# Code to generate recommendations by Title
#######################################

import os
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from collections import Counter
import csv

# Hugging Face imports
from transformers import AutoTokenizer, AutoModel

########################################
# 1) Load the Fine-Tuned HF Model
########################################

def load_fine_tuned_model(model_dir, base_model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Loads both the tokenizer (from 'base_model_name') and
    the fine-tuned Hugging Face model (from 'model_dir').
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModel.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

########################################
# 2) Compute the embedding for a user input title
########################################

def get_playlist_embedding(playlist_name, tokenizer, model):
    """
    Tokenize and forward-pass using the fine-tuned model,
    then do mean pooling over the last hidden states.
    """
    if not isinstance(playlist_name, str):
        playlist_name = str(playlist_name)

    with torch.no_grad():
        inputs = tokenizer(playlist_name, return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
        embedding = last_hidden.mean(dim=1).squeeze().cpu().numpy()

    return embedding

########################################
# 3) Load Precomputed Embeddings
########################################

def load_playlist_embeddings(embeddings_file):
    with open(embeddings_file, 'rb') as f:
        playlist_embeddings = pickle.load(f)
    return playlist_embeddings

########################################
# 4) Build pid->tracks dictionary
########################################

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
            pid_str = row["pid"].strip()
            track_uri = row["track_uri"]

            if pid_str not in playlist_tracks:
                playlist_tracks[pid_str] = []
            if track_uri in track_metadata:
                playlist_tracks[pid_str].append(track_metadata[track_uri])

    return playlist_tracks

########################################
# 5) Find Similar Playlists
########################################

def find_similar_playlists(user_input_title, playlist_embeddings, tokenizer, model, top_k=50):
    """
    1) Generate an embedding for the user-input playlist title.
    2) Compare to the precomputed embeddings dictionary.
    3) Return the top_k most similar.
    """
    user_embedding = get_playlist_embedding(user_input_title, tokenizer, model)

    similarities = []
    for pid, data in tqdm(playlist_embeddings.items(), desc="Scoring Playlists", unit="playlist"):
        similarity = cosine_similarity([user_embedding], [data["embedding"]])[0][0]
        similarities.append((pid, similarity))

    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

########################################
# 6) Get the Top Occurring Songs
########################################

def get_top_songs_with_artists(similar_playlists, playlist_tracks, top_k=66):
    """
    For each similar playlist, retrieve songs from playlist_tracks
    and count their occurrences.
    """
    from collections import Counter
    song_counter = Counter()

    for pid, _ in tqdm(similar_playlists, desc="Counting songs", unit="playlist"):
        pid_str = str(pid)
        if pid_str in playlist_tracks:
            for track in playlist_tracks[pid_str]:
                song_counter[(track["track_name"], track["artist_name"])] += 1

    return song_counter.most_common(top_k)

########################################
# 7) Main Function
########################################

def main():
    # File paths
    model_dir = "/home/vellard/playlist_continuation/finetuning/fine_tuned_model"
    playlist_embeddings_file = "/home/vellard/playlist_continuation/embeddings/new-model/playlists_embeddings.pkl"
    items_csv = "/data/playlist_continuation_data/csvs/items.csv"
    tracks_csv = "/data/playlist_continuation_data/csvs/tracks.csv"

    # 1. Load the HF model & tokenizer
    tokenizer, model = load_fine_tuned_model(model_dir)
    print("Loaded tokenizer & fine-tuned model successfully.")

    # 2. Load the precomputed playlist embeddings
    playlist_embeddings = load_playlist_embeddings(playlist_embeddings_file)

    # 3. Load track data
    playlist_tracks = load_playlist_tracks_with_artists(items_csv, tracks_csv)

    # 4. Ask user for a playlist NAME (not PID)
    playlist_name = input("Enter a playlist name: ")
    print(f"Generating recommendations for custom playlist name: '{playlist_name}'...")

    # 5. Find the most similar playlists from the embedding dictionary
    top_playlists = find_similar_playlists(playlist_name, playlist_embeddings, tokenizer, model, top_k=10)

    print("\nTop Similar Playlists:")
    for i, (sim_pid, sim_score) in enumerate(top_playlists, start=1):
        # Each embedding dict item might have a "title" key
        # so we can display the known playlist title
        known_title = playlist_embeddings[sim_pid]["title"]
        print(f"{i}. PID={sim_pid}, Title='{known_title}', Similarity: {sim_score:.4f}")

    # 6. Retrieve top songs from these similar playlists
    top_songs = get_top_songs_with_artists(top_playlists, playlist_tracks, top_k=10)

    print("\nTop Recommended Songs:")
    for i, ((song, artist), count) in enumerate(top_songs, start=1):
        print(f"{i}. Song: '{song}', Artist: '{artist}', Occurrences: {count}")

    # 7. No metrics: we are skipping the evaluation step

if __name__ == "__main__":
    main()
