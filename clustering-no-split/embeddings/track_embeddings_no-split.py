##############################################################
# Code to precompute the tracks embeddings before clustering #
##############################################################

import os
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

#-------------------------------------#
# Load the playlists (titles and tracks)

def load_playlist_titles(file_path):
    titles = {}
    #Dictionary to store (pid, playlist title)
    with open(file_path, 'r', encoding='utf8') as f:#Read the csv file
        for line in f:
            pid, title = line.strip().split(',')[0], line.strip().split(',')[1]#The separator is a ,
            titles[pid] = title
    return titles

def load_playlist_track_titles(items_file_path, tracks_file_path):#Two dictionaries to store pids, playlists titles and tracks
    track_titles = {} #Act like a look-up table (faster computation)
    #First dictionary: store (track uri, track title)
    with open(tracks_file_path, 'r', encoding='utf8') as f:
        for line in f:
            track_uri, track_name, *_ = line.strip().split(',')
            track_titles[track_uri] = track_name

    playlist_tracks = {}
    #Second dictionary: (pid, [title1, title2...])
    with open(items_file_path, 'r', encoding='utf8') as f:
        for line in f:
            pid, _, track_uri = line.strip().split(',')#Chat GPT's tip: use "_" in a "for" loop when we don't need the incrementer
            if pid not in playlist_tracks:
                playlist_tracks[pid] = []
            if track_uri in track_titles:
                playlist_tracks[pid].append(track_titles[track_uri])   
    return playlist_tracks

#----------------------------------------------#
#Compute the embeddings (for tracks first, then use them to represent a playlist)

def compute_track_embeddings(model, playlist_tracks):
    unique_tracks = list(set(title for tracks in playlist_tracks.values() for title in tracks))
    # We use the encode() function (pre-created for sentenceBERT)
    track_embeddings_array = model.encode(unique_tracks, show_progress_bar=True, convert_to_numpy=True)
    track_embeddings = {track_title: emb for track_title, emb in zip(unique_tracks, track_embeddings_array)}
    return track_embeddings

def compute_playlist_embeddings(playlist_tracks, track_embeddings):
    playlist_embeddings = {}
    for pid, tracks in tqdm(playlist_tracks.items(), desc="Processing playlists", unit="playlist"):
        final_embedding = np.mean(track_embeddings[track], axis=0)
        playlist_embeddings[pid] = final_embedding
    return playlist_embeddings

def main():
    base_dir = "/data/playlist_continuation_data/csvs"
    tracks_file = "/data/playlist_continuation_data/csvs/tracks.csv"
    output_dir = "/home/vellard/playlist_continuation/embeddings"
    playlist_file = os.path.join(base_dir, f"playlists.csv")
    output_file = os.path.join(output_dir, f"embeddings.pkl")
    items_file = os.path.join(base_dir, f"items.csv")
    os.makedirs(output_dir, exist_ok=True)

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    playlist_titles = load_playlist_titles(playlist_file)
    playlist_tracks = load_playlist_track_titles(items_file, tracks_file)
    
    track_embeddings = compute_track_embeddings(model, playlist_tracks)

    playlist_embeddings = compute_playlist_embeddings(playlist_tracks, track_embeddings)
    
    embeddings_data = {
            "playlist_embeddings": playlist_embeddings,
            "playlist_titles": playlist_titles,
            "playlist_tracks": playlist_tracks
    }

    with open(output_file, 'wb') as f:
        pickle.dump(embeddings_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Embeddings (no split) saved to {output_file}")
    
if __name__ == "__main__":
    main()
