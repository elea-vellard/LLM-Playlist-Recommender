# This script computes various metrics for playlist recommendations based on a generated playlist and relevant songs.
# It loads data from CSV files and a JSON file, processes the data, and calculates metrics such as HIT@N, Precision@N, Recall@N, MRR@N, R-Precision, and NDCG.

###########
# Imports #
###########

import csv
import yaml
import statistics
from tqdm import tqdm  # type: ignore

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



def load_all_playlist_data(items_csv=items_csv, tracks_csv=tracks_csv):
    #Loads track metadata and the mapping between PID and track once.
    #Returns a dictionary: {pid: [(track_name, artist_name), ...]}.
    #This is done to avoid loading the CSV files multiple times.
    
    track_metadata = {}
    with open(tracks_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading track metadata", unit=" tracks"):
            track_metadata[row["track_uri"]] = {
                "track_name": row["track_name"],
                "artist_name": row["artist_name"],
            }

    playlist_tracks = {}
    with open(items_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading playlist tracks", unit=" playlists"):
            current_pid = row["pid"]
            track_uri = row["track_uri"]
            if current_pid not in playlist_tracks:
                playlist_tracks[current_pid] = []
            if track_uri in track_metadata:
                playlist_tracks[current_pid].append(
                    (track_metadata[track_uri]["track_name"], track_metadata[track_uri]["artist_name"])
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