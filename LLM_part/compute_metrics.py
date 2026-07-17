# This script computes various metrics for playlist recommendations based on a generated playlist and relevant songs.
# It loads data from CSV files and a JSON file, processes the data, and calculates metrics such as HIT@N, Precision@N, Recall@N, MRR@N, R-Precision, and NDCG.

###########
# Imports #
###########

import csv
import json
import argparse
import statistics
import math
from tqdm import tqdm  # type: ignore
from data_loader import load_all_playlist_data

# File paths
items_csv = "./data/items.csv"
tracks_csv = "./data/tracks.csv"

########################
# Functions definition #
########################


def normalize_song(song):
    #Normalize a the tuple (track_name, artist_name) by stripping spaces and converting to lowercase.
    #Returns a tuple (track_name, artist_name).

    return (song[0].strip().lower(), song[1].strip().lower())

SONG = 0
ARTIST = 1
def compute_metrics(recommended_songs, relevant_songs, top_n=10):
    # Convert lists to sets and prepare R-Precision metrics
    G_T = set([song[SONG] for song in relevant_songs])
    G_A = set([song[ARTIST] for song in relevant_songs])

    R = len(G_T)
    top_r = recommended_songs[:R]
    S_T = set([song['song'] for song in top_r])
    S_A = set([song['artist'] for song in top_r])

    exact_matches = S_T & G_T
    matched_artists = S_A & G_A
    track_score = len(exact_matches)
    artist_score = len(matched_artists) * 0.25
    r_precision = (track_score + artist_score) / R if R > 0 else 0.0

    hits = sum(1 for song in recommended_songs[:top_n] if song['song'] in G_T)
    hit_score = hits / min(top_n, len(G_T)) if len(G_T) > 0 else 0.0
    precision = hits / len(recommended_songs[:top_n]) if recommended_songs[:top_n] else 0.0
    recall = hits / len(G_T) if len(G_T) > 0 else 0.0

    mrr = 0.0
    for i, song in enumerate(recommended_songs[:top_n]):
        if song['song'] in G_T:
            mrr = 1 / (i + 1)
            break

    relevance_list = [1 if song['song'] in G_T else 0 for song in recommended_songs[:top_n]]
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

def load_generated_data(json_file):
    # Loads the JSON file containing the 22 generated playlists.
    # Assume the JSON is a dictionary with PIDs as keys.

    with open(json_file, 'r', encoding='utf8') as f:
        data = json.load(f)
    return data

def main(json_file):
    generated_data = load_generated_data(json_file)
    print(f"Loaded {len(generated_data)} generated playlists from {json_file}")
    pids = [int(pl["pid"]) for pl in generated_data]

    print("Loading playlist data")
    playlist_tracks = load_all_playlist_data(items_csv, tracks_csv, wanted_playlist_ids=pids)
    print(f"Loaded {len(playlist_tracks.keys())} playlists")

    metrics_list = []

    print('Compute metrics')
    # For each playlist generated in the JSON (five-shot mode)
    for pl in tqdm(generated_data):
        pid = int(pl["pid"])
        playlist_title = pl["playlist_title"]
        generated_songs = pl.get("tracks", [])

        relevant_songs = playlist_tracks.get(pid, [])
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
        # print(f"PID {pid} - Metrics: {metrics}")

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
    print(average_metrics)

    output_csv = json_file.replace('.json', '_metrics.csv')
    with open(output_csv, "w", newline='', encoding="utf8") as csvfile:
        fieldnames = ["pid", "playlist_title", "HIT@N", "Precision@N", "Recall@N", "MRR@N", "R-Precision", "NDCG"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow(m)

    print(f"\nThe metrics for each playlist were written in : {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute recommendation metrics from generated playlists.")
    parser.add_argument(
        "--json-file",
        default="out_llm_prediction/subset22.json",
        help="Path to the JSON file containing generated playlists."
    )
    args = parser.parse_args()
    main(args.json_file)
