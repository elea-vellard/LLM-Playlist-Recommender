#######################################
# Code to generate the recommendation #
#######################################

import os
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from collections import Counter
import csv

# NEW: Hugging Face imports
from transformers import AutoTokenizer, AutoModel

##########################
# 1) Load Fine-tuned Model
##########################

def load_fine_tuned_model(model_dir, base_model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Loads both the tokenizer (from the 'base_model_name') and
    the fine-tuned Hugging Face model (from 'model_dir').
    Returns (tokenizer, model).
    """
    # Load the tokenizer from the same base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load the fine-tuned transformer model
    model = AutoModel.from_pretrained(model_dir)
    model.eval()

    return tokenizer, model


##############################
# 2) Compute the playlist embedding
##############################

def get_playlist_embedding(playlist_name, tokenizer, model):
    """
    Equivalent to the hugging-face approach in your embeddings script:
    - tokenize
    - forward pass
    - mean pooling over last_hidden_state
    - return numpy array
    """
    if not isinstance(playlist_name, str):
        playlist_name = str(playlist_name)

    with torch.no_grad():
        inputs = tokenizer(playlist_name, return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
        embedding = last_hidden.mean(dim=1).squeeze().cpu().numpy()

    return embedding


###########################
# 3) Load Precomputed Embeddings
###########################

def load_playlist_embeddings(embeddings_file):
    with open(embeddings_file, 'rb') as f:
        playlist_embeddings = pickle.load(f)
    return playlist_embeddings


#########################################
# 4) Load Track Metadata & Build pid->tracks
#########################################

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
            # Store PID as a string for consistent lookups
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

def find_similar_playlists(playlist_name, playlist_embeddings, tokenizer, model, top_k):
    """
    Replaces 'model.encode' with the hugging-face approach via get_playlist_embedding.
    """
    playlist_embedding = get_playlist_embedding(playlist_name, tokenizer, model)

    similarities = []
    for pid, metadata in tqdm(playlist_embeddings.items(), desc="Scoring Playlists", unit="playlist"):
        # 'pid' is likely an integer in the embeddings dictionary
        similarity = cosine_similarity([playlist_embedding], [metadata["embedding"]])[0][0]
        similarities.append((pid, similarity))

    # Sort by descending similarity and take top_k
    sorted_playlists = sorted(similarities, key=lambda x: x[1], reverse=True)
    return sorted_playlists[:top_k]


########################################
# 6) Retrieve the Top Songs from those playlists
########################################

def get_top_songs_with_artists(similar_playlists, playlist_tracks, top_k=10):
    song_counter = Counter()
    for pid, _ in tqdm(similar_playlists, desc="Counting songs", unit="playlist"):
        # Convert integer pid to string to match keys in playlist_tracks
        pid_str = str(pid)
        if pid_str in playlist_tracks:
            for track_metadata in playlist_tracks[pid_str]:
                song_counter[(track_metadata["track_name"], track_metadata["artist_name"])] += 1
    return song_counter.most_common(top_k)


########################################
# 7) Evaluation metrics
########################################

def compute_metrics(recommended_songs, relevant_songs, top_n=10):
    relevant_hits = [song for song in recommended_songs[:top_n] if song in relevant_songs]
    hits = len(relevant_hits)
    hit_score = hits / min(top_n, len(relevant_songs)) if len(relevant_songs) > 0 else 0.0

    precision = hits / len(recommended_songs[:top_n]) if len(recommended_songs[:top_n]) > 0 else 0.0
    recall = hits / len(relevant_songs) if len(relevant_songs) > 0 else 0.0

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


########################################
# 8) Main Function
########################################

def main():
    # Paths
    model_dir = "/home/vellard/playlist_continuation/finetuning/fine_tuned_model"
    playlist_embeddings_file = "/home/vellard/playlist_continuation/embeddings/new-model/playlists_embeddings.pkl"
    items_csv = "/data/playlist_continuation_data/csvs/items.csv"
    tracks_csv = "/data/playlist_continuation_data/csvs/tracks.csv"
    playlists_csv = "/data/playlist_continuation_data/csvs/playlists.csv"

    # 1. Load the tokenizer & HF model
    tokenizer, model = load_fine_tuned_model(model_dir)
    print("Loaded tokenizer & fine-tuned model successfully.")

    # 2. Load precomputed embeddings
    playlist_embeddings = load_playlist_embeddings(playlist_embeddings_file)

    # 3. Load track metadata for the playlist->tracks mapping
    playlist_tracks = load_playlist_tracks_with_artists(items_csv, tracks_csv)

    # Prompt user for a playlist PID (as a string)
    pid = input("Enter the playlist PID: ").strip()
    print(f"Generating recommendations for playlist PID: '{pid}'...")

    # Build a dictionary: pid->playlist_title, also as strings
    playlist_titles = {}
    with open(playlists_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading playlist titles", unit="playlist"):
            pid_str = row["pid"].strip()
            playlist_titles[pid_str] = row["name"]

    # Attempt to get the playlist name from the dictionary
    playlist_name = playlist_titles.get(pid, "Unknown Playlist Title")
    if playlist_name == "Unknown Playlist Title":
        print("Invalid PID entered.")
        return

    # 4. Find similar playlists (will yield integer PIDs)
    top_playlists = find_similar_playlists(playlist_name, playlist_embeddings, tokenizer, model, top_k=50)

    print("\nTop Similar Playlists:")
    for i, (similar_pid, similarity) in enumerate(top_playlists, start=1):
        # Convert integer -> string for dictionary lookup
        title = playlist_titles.get(str(similar_pid), "Unknown Playlist Title")
        print(f"{i}. Playlist Title: {title}, Similarity: {similarity:.4f}")

    # 5. Retrieve top songs (convert each pid to str)
    top_songs = get_top_songs_with_artists(top_playlists, playlist_tracks, top_k=66)

    print("\nTop Recommended Songs:")
    for i, ((song, artist), count) in enumerate(top_songs, start=1):
        print(f"{i}. Song: {song}, Artist: {artist}, Occurrences: {count}")

    # 6. Evaluate metrics
    relevant_songs_info = playlist_tracks.get(pid, [])
    relevant_songs = list(set((trk["track_name"], trk["artist_name"]) for trk in relevant_songs_info))

    recommended_songs = [song_artist for song_artist, _ in top_songs]
    metrics = compute_metrics(recommended_songs, relevant_songs, top_n=66)

    print(f"\nRecommended Songs:", recommended_songs)
    print("Relevant Songs:", relevant_songs)


if __name__ == "__main__":
    main()
