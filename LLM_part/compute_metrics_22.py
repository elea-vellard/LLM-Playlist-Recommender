# This script computes various metrics for playlist recommendations based on a generated playlist and relevant songs.
# It loads data from CSV files and a JSON file, processes the data, and calculates metrics such as HIT@N, Precision@N, Recall@N, MRR@N, R-Precision, and NDCG.

###########
# Imports #
###########

import csv
import json
import os
import statistics
import math
from tqdm import tqdm  # type: ignore

########################
# Functions definition #
########################


def normalize_song(song):
    #Normalize a the tuple (track_name, artist_name) by stripping spaces and converting to lowercase.
    #Returns a tuple (track_name, artist_name).

    return (song[0].strip().lower(), song[1].strip().lower())


def compute_metrics(recommended_songs, relevant_songs, top_n=10):
    # Convert lists to sets and prepare R-Precision metrics

    G_T = set(relevant_songs)
    G_A = set(artist for _, artist in relevant_songs)

    R = len(G_T)
    top_r = recommended_songs[:R]
    S_T = set(top_r)
    S_A = set(artist for _, artist in top_r)

    exact_matches = S_T & G_T
    matched_artists = S_A & G_A
    track_score = len(exact_matches)
    artist_score = len(matched_artists) * 0.25
    r_precision = (track_score + artist_score) / R if R > 0 else 0.0

    hits = sum(1 for song in recommended_songs[:top_n] if song in G_T)
    hit_score = hits / min(top_n, len(G_T)) if len(G_T) > 0 else 0.0
    precision = hits / len(recommended_songs[:top_n]) if recommended_songs[:top_n] else 0.0
    recall = hits / len(G_T) if len(G_T) > 0 else 0.0

    mrr = 0.0
    for i, song in enumerate(recommended_songs[:top_n]):
        if song in G_T:
            mrr = 1 / (i + 1)
            break

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

def load_all_playlist_data(items_csv, tracks_csv):
    #Loads track metadata and the mapping between PID and track once.
    #Returns a dictionary: {pid: [(track_name, artist_name), ...]}.
    #This is done to avoid loading the CSV files multiple times.
    
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
            current_pid = row["pid"]
            track_uri = row["track_uri"]
            if current_pid not in playlist_tracks:
                playlist_tracks[current_pid] = []
            if track_uri in track_metadata:
                playlist_tracks[current_pid].append(
                    (track_metadata[track_uri]["track_name"], track_metadata[track_uri]["artist_name"])
                )
    return playlist_tracks

def load_generated_data(json_file):
    # Loads the JSON file containing the 22 generated playlists.
    # Assume the JSON is a dictionary with PIDs as keys.

    with open(json_file, 'r', encoding='utf8') as f:
        data = json.load(f)
    return data

def main():
    # File paths
    items_csv = "/data/csvs/items.csv"
    tracks_csv = "/data/csvs/tracks.csv"
    json_file = "Json_file/Five_shot_22_song.json"

    generated_data = load_generated_data(json_file)
    print("Loading all playlist data")
    playlist_tracks = load_all_playlist_data(items_csv, tracks_csv)

    metrics_list = []

    # For each playlist generated in the JSON (five-shot mode)
    for pid, pl_data in generated_data.items():
        playlist_title = pl_data["playlist_title"]
        # For five-shot, the seed song is song1 and artist1
        first_song = pl_data["song1"]
        artist = pl_data["artist1"]
        # Convert the generated_playlist (list of dictionaries) to a list of tuples
        generated_songs = [(item["song"], item["artist"]) for item in pl_data.get("generated_playlist", [])]

        seed_song = (first_song, artist)
        relevant_songs = playlist_tracks.get(pid, [])
        if seed_song:
            relevant_songs = [song for song in relevant_songs if normalize_song(song) != normalize_song(seed_song)]
        relevant_songs = list(set(relevant_songs))

        # Compute metrics
        metrics = compute_metrics(generated_songs, relevant_songs, top_n=10)
        metrics_list.append({
            "pid": pid,
            "playlist_title": playlist_title,
            "HIT@N": metrics["HIT@N"],
            "Precision@N": metrics["Precision@N"],
            "Recall@N": metrics["Recall@N"],
            "MRR@N": metrics["MRR@N"],
            "R-Precision": metrics["R-Precision"],
            "NDCG": metrics["NDCG"]
        })
        print(f"PID {pid} - Metrics: {metrics}")

    # Calculate average metrics
    avg_hit = statistics.mean([m["HIT@N"] for m in metrics_list]) if metrics_list else 0
    avg_precision = statistics.mean([m["Precision@N"] for m in metrics_list]) if metrics_list else 0
    avg_recall = statistics.mean([m["Recall@N"] for m in metrics_list]) if metrics_list else 0
    avg_mrr = statistics.mean([m["MRR@N"] for m in metrics_list]) if metrics_list else 0
    avg_rprec = statistics.mean([m["R-Precision"] for m in metrics_list]) if metrics_list else 0
    avg_ndcg = statistics.mean([m["NDCG"] for m in metrics_list]) if metrics_list else 0

    average_metrics = {
        "pid": "average",
        "playlist_title": "average",
        "HIT@N": avg_hit,
        "Precision@N": avg_precision,
        "Recall@N": avg_recall,
        "MRR@N": avg_mrr,
        "R-Precision": avg_rprec,
        "NDCG": avg_ndcg
    }
    metrics_list.append(average_metrics)

    output_csv = "Five_shot_22_metrics.csv"
    with open(output_csv, "w", newline='', encoding="utf8") as csvfile:
        fieldnames = ["pid", "playlist_title", "HIT@N", "Precision@N", "Recall@N", "MRR@N", "R-Precision", "NDCG"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow(m)

    print(f"\nThe metrics for each playlist were written in : {output_csv}")

if __name__ == "__main__":
    main()
