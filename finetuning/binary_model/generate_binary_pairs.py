import csv
import random
import pandas as pd
from itertools import combinations
from sentence_transformers import InputExample
from tqdm import tqdm

########################
# Functions defintions #
########################
def load_clusters(csv_file):
    # Outputs 2 dictionaries: {pid, title} and {pid, cluster ID}
    playlist_title = {}
    playlist_cluster = {}
    with open(csv_file, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['Playlist ID'].strip()
            playlist_title[pid] = row['Playlist Title'].strip()
            playlist_cluster[pid] = row['Cluster ID'].strip()
    print("Clusters loaded")
    return playlist_title, playlist_cluster


def generate_pairs(playlist_title, playlist_cluster, max_pairs=2000):
    # Generate random positive and negative pairs of playlists
    # Positive pairs are playlists in the same cluster (label 1)
    # Negative pairs are playlists not in the same cluster (label 0)
    # max_pairs to avoid an unbalance in the number of pairs between big and small clusters (bias)
    grouped_pids = {}
    for pid, cluster_id in tqdm(playlist_cluster.items()):
        grouped_pids.setdefault(cluster_id, []).append(pid)

    all_pids = list(playlist_title.keys())
    pairs = []

    for cluster_id, pids in tqdm(grouped_pids.items(), desc="Generating pairs"):
        if len(pids) < 2:
            continue

        positive_pairs = list(combinations(pids, 2))[:max_pairs]

        for pid1, pid2 in positive_pairs:
            pairs.append(InputExample(texts=[playlist_title[pid1],
                                             playlist_title[pid2]],
                                      label=1.0))
            # For each positive pair, we create a negative pair
            # to obtain the same number of positive/negative pairs
            other_clusters = [pid for pid in all_pids if playlist_cluster[pid] != cluster_id]
            pid_negative = random.choice(other_clusters)
            pairs.append(InputExample(texts=[playlist_title[pid1],
                                             playlist_title[pid_negative]],
                                      label=0.0))
    return pairs


def pairs_to_dataframe(pairs):
    # Transform in dataframes to output csv files (easier)
    rows = []
    for ex in pairs:
        rows.append({
            "text1": ex.texts[0],
            "text2": ex.texts[1],
            "label": int(ex.label)
        })
    return pd.DataFrame(rows)

########
# Main #
########
if __name__ == "__main__":

    train_csv_path = "/home/vellard/playlist_continuation/clusters/clusters_train.csv"
    val_csv_path   = "/home/vellard/playlist_continuation/clusters/clusters_val.csv"

    train_pairs_output = "/home/vellard/playlist_continuation/binary/train_pairs.csv"
    val_pairs_output   = "/home/vellard/playlist_continuation/binary/val_pairs.csv"

    train_titles, train_clusters = load_clusters(train_csv_path)
    val_titles,   val_clusters   = load_clusters(val_csv_path)

    train_pairs = generate_pairs(train_titles, train_clusters, max_pairs=2000)
    val_pairs   = generate_pairs(val_titles,   val_clusters,   max_pairs=2000)

    df_train = pairs_to_dataframe(train_pairs)
    df_val   = pairs_to_dataframe(val_pairs)

    df_train.to_csv(train_pairs_output, index=False, encoding="utf-8")
    df_val.to_csv(val_pairs_output,   index=False, encoding="utf-8")

    print(f"Saved training pairs in {train_pairs_output}")
    print(f"Savec validation pairs in {val_pairs_output}")
