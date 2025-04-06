import os
import csv
import math
import pickle
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModel

# Load the model (choose model directory in the main)
def load_fine_tuned_model(model_dir, base_model_name='sentence-transformers/all-MiniLM-L6-v2'):
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected.")
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModel.from_pretrained(model_dir)
    model.eval()
    model.to(device)

    return tokenizer, model

# Create the input playlist embedding
def get_text_embedding(text, tokenizer, model):
    device = next(model.parameters()).device
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True
        ).to(device)
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state
        embedding = last_hidden.mean(dim=1).squeeze()
    
    # Go back to CPU to store it later
    return embedding.cpu().numpy()

def load_playlist_embeddings(embeddings_file):
    # Load the precomputed playlists embeddings
    with open(embeddings_file, 'rb') as f:
        playlist_embeddings = pickle.load(f)
    return playlist_embeddings

def load_playlist_tracks_with_artists(items_csv, tracks_csv):
    # Link each song to their information (title and artist)
    track_metadata = {}
    with open(tracks_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading track metadata", unit="track"):
            track_uri = row["track_uri"]
            track_metadata[track_uri] = {
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

# Calculate cosine similarity scores and find the K closest playlists
def find_similar_playlists_batch(query_emb_np, playlist_embeddings, top_k=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_emb_torch = torch.from_numpy(query_emb_np).unsqueeze(0).to(device)

    pids = list(playlist_embeddings.keys())
    all_embs_np = [playlist_embeddings[pid]["embedding"] for pid in pids]
    all_embs_np = np.stack(all_embs_np, axis=0)
    all_embs_torch = torch.from_numpy(all_embs_np).to(device)
    
    cos_sims = F.cosine_similarity(query_emb_torch, all_embs_torch, dim=1)
    
    # use cpu for the sort
    cos_sims_np = cos_sims.cpu().numpy()
    similarities = list(zip(pids, cos_sims_np))
    # Sort and find k highest scores
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]

# Among the k closest playlists, find the most occuring ones
def get_top_songs_with_artists(similar_playlists, playlist_tracks, top_k=66):
    song_counter = Counter()
    for pid, _ in similar_playlists:
        pid_str = str(pid)
        if pid_str in playlist_tracks:
            for track_info in playlist_tracks[pid_str]:
                pair = (track_info["track_name"], track_info["artist_name"])
                song_counter[pair] += 1
    return song_counter.most_common(top_k)

def compute_metrics(recommended_songs, relevant_songs, top_n=66):
    # Sets
    G_T = set(relevant_songs)
    G_A = set(artist for _, artist in relevant_songs)

    R = len(G_T)
    top_r = recommended_songs[:R]
    S_T = set(top_r)
    S_A = set(artist for _, artist in top_r)
    
    # R-Precision with artist bonus
    exact_matches = S_T & G_T
    matched_artists = S_A & G_A
    track_score = len(exact_matches)
    artist_score = len(matched_artists) * 0.25
    r_precision = (track_score + artist_score) / R if R > 0 else 0.0

    # HIT@N, Precision, Recall, MRR
    hits = sum(1 for song in recommended_songs[:top_n] if song in G_T)
    hit_score = hits / min(top_n, len(G_T)) if len(G_T) > 0 else 0.0
    precision = hits / len(recommended_songs[:top_n]) if len(recommended_songs[:top_n]) > 0 else 0.0
    recall = hits / len(G_T) if len(G_T) > 0 else 0.0

    mrr = 0.0
    for i, song in enumerate(recommended_songs[:top_n]):
        if song in G_T:
            mrr = 1 / (i + 1)
            break

    # NDCG
    relevance_list = [1 if song in G_T else 0 for song in recommended_songs[:top_n]]

    def dcg(rel):
        return sum(rel_i / math.log2(idx + 2) for idx, rel_i in enumerate(rel))

    dcg_val = dcg(relevance_list)
    ideal_rel = sorted(relevance_list, reverse=True)
    idcg_val = dcg(ideal_rel)
    ndcg = dcg_val / idcg_val if idcg_val > 0 else 0.0

    return {
        "HIT@N": hit_score,
        "Precision@N": precision,
        "Recall@N": recall,
        "MRR@N": mrr,
        "R-Precision": r_precision,
        "NDCG": ndcg
    }

########
# Main #
########
def main():

    # Choose the model directory
    model_dir = "/home/vellard/playlist_continuation/fine_tuned_model_no_scheduler_2"
    playlist_embeddings_file = "/home/vellard/playlist_continuation/playlists_embeddings/final_embeddings/playlists_embeddings_scheduler.pkl"
    items_csv = "/data/csvs/items.csv"
    tracks_csv = "/data/csvs/tracks.csv"
    playlists_csv = "/data/csvs/playlists.csv"
    clusters_test_csv = "/home/vellard/playlist_continuation/clusters/clusters_test.csv"

    results_csv = "evaluation_results_scheduler.csv"

    # Load models and playlists
    tokenizer, model = load_fine_tuned_model(model_dir)
    print("Modèle chargé sur GPU avec succès.")

    playlist_embeddings = load_playlist_embeddings(playlist_embeddings_file)
    print("Embeddings de playlists chargés.")

    playlist_tracks = load_playlist_tracks_with_artists(items_csv, tracks_csv)
    print("playlist_tracks chargé.")

    playlist_titles = {}
    with open(playlists_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid_str = row["pid"].strip()
            playlist_titles[pid_str] = row["name"]

    # Batch evaluation
    all_results = []
    with open(clusters_test_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Evaluating test playlists", unit="playlist"):
            cluster_id = row["Cluster ID"]
            test_pid = row["Playlist ID"].strip()
            playlist_name = row["Playlist Title"].strip()

            if not playlist_name or playlist_name == "Unknown Playlist Title":
                continue

            first_track_info = playlist_tracks.get(test_pid, [])
            if not first_track_info:
                continue
            first_track_name = first_track_info[0]["track_name"]
            first_track_artist = first_track_info[0]["artist_name"]

            emb_title = get_text_embedding(playlist_name, tokenizer, model)
            emb_track = get_text_embedding(f"{first_track_name} by {first_track_artist}", tokenizer, model)
            query_emb_np = (emb_title + emb_track) / 2.0

            top_playlists = find_similar_playlists_batch(
                query_emb_np,
                playlist_embeddings,
                top_k=50
            )
            top_songs = get_top_songs_with_artists(
                top_playlists,
                playlist_tracks,
                top_k=10
            )

            relevant_songs_info = playlist_tracks.get(test_pid, [])
            relevant_songs = list({
                (trk["track_name"], trk["artist_name"]) for trk in relevant_songs_info
            })

            first_pair = (first_track_name, first_track_artist)
            relevant_songs = [s for s in relevant_songs if s != first_pair]

            recommended_songs = [song_artist for song_artist, _ in top_songs]
            metrics = compute_metrics(recommended_songs, relevant_songs, top_n=10)

            result_row = {
                "Cluster ID": cluster_id,
                "Playlist ID": test_pid,
                "Playlist Title": playlist_name,
                "HIT@10": metrics["HIT@N"],
                "Precision@10": metrics["Precision@N"],
                "Recall@10": metrics["Recall@N"],
                "MRR@10": metrics["MRR@N"],
                "R-Precision": metrics["R-Precision"],
                "NDCG@10": metrics["NDCG"]
            }
            all_results.append(result_row)

    fieldnames = [
        "Cluster ID", "Playlist ID", "Playlist Title",
        "HIT@10", "Precision@10", "Recall@10", "MRR@10",
        "R-Precision", "NDCG@10"
    ]
    with open(results_csv, 'w', encoding='utf8', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"\nResults saved in '{results_csv}'.")


if __name__ == "__main__":
    main()
