# This script computes various metrics for playlist recommendations based on a generated playlist and relevant songs.
# It loads data from CSV files and a JSON file, processes the data, and calculates metrics such as HIT@N, Precision@N, Recall@N, MRR@N, R-Precision, and NDCG.

###########
# Imports #
###########

import csv
import polars as pl
import yaml
import statistics
from tqdm import tqdm  # type: ignore
from collections import defaultdict

########################
# Functions definition #
########################


items_csv = "/data/items.csv"
tracks_csv = "/data/tracks.csv"
test_set = "/data/cluster_test.csv"

def normalize_song(song):
    #Normalize a the tuple (track_name, artist_name) by stripping spaces and converting to lowercase.
    #Returns a tuple (track_name, artist_name).

    return (song[0].strip().lower(), song[1].strip().lower())

def load_all_playlist_data(items_csv=items_csv, tracks_csv=tracks_csv, wanted_playlist_ids=None):
    #Loads track metadata and the mapping between PID and track once.
    #Returns a dictionary: {pid: [(track_name, artist_name), ...]}.
    #This is done to avoid loading the CSV files multiple times.
    
    items_lf = pl.scan_csv(items_csv).select(["pid", "track_uri"])

    if wanted_playlist_ids is not None:
        items_lf = items_lf.filter(
            pl.col("pid").is_in(wanted_playlist_ids)
        )

    tracks_lf = pl.scan_csv(tracks_csv).select(
        ["track_uri", "track_name", "artist_name"]
    )

    df = (
        items_lf
        .join(
            tracks_lf,
            on="track_uri",
            how="inner",
        )
        .select(
            ["pid", "track_name", "artist_name"]
        )
        .collect(engine="streaming")
    )

    playlist_tracks = defaultdict(list)

    for current_pid, track_name, artist_name in df.iter_rows():
        playlist_tracks[current_pid].append(
            (track_name, artist_name)
        )

    return playlist_tracks

def load_playlists_yaml(yaml_path: str) -> dict:
    """Load playlists from YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['playlists']


def load_playlists_csv(test_set=test_set) -> list:
    """Load playlists from CSV file."""
    playlists = []
    with open(test_set, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading playlists from CSV", unit=" playlists"):
            playlist = {
                "pid": row["Playlist ID"],
                "playlist_title": row["Playlist Title"],
            }
            playlists.append(playlist)
    return playlists