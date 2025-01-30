######################################################
# adapted from previous project (mdp2csv.py) and gpt #
######################################################

import csv
import os
import json
from os import listdir, path

# Definit input and output paths
input_dir = '/data/million_playlist_dataset'
output_dir = '/data/playlist_continuation_data/csvs'
os.makedirs(output_dir, exist_ok=True)#create a new directory to store csv files

# Open files in writing mode
# For now, we will only use these two
items_file = open(path.join(output_dir, 'items.csv'), 'w', newline='', encoding='utf8')#Saves the position of songs in each playlist.
playlists_file = open(path.join(output_dir, 'playlists.csv'), 'w', newline='', encoding='utf8')#Contains the main information of each playlist.

# Additionnal information that can be used later
tracks_file = open(path.join(output_dir, 'tracks.csv'), 'w', newline='', encoding='utf8')#Contains unique information about each song in the database.
playlists_descr_file = open(path.join(output_dir, 'playlists_descr.csv'), 'w', newline='', encoding='utf8')#Only saves playlist descriptions, for those that have them.

# Create csv writers
items_writer = csv.writer(items_file)
playlists_writer = csv.writer(playlists_file)
tracks_writer = csv.writer(tracks_file)
playlists_descr_writer = csv.writer(playlists_descr_file)

# Write CSV file headers
items_writer.writerow(['pid', 'track_position', 'track_uri'])
playlists_writer.writerow(['pid', 'name', 'collaborative', 'num_tracks', 'num_artists',
                           'num_albums', 'num_followers', 'num_edits', 'modified_at', 'duration_ms'])
playlists_descr_writer.writerow(['pid', 'description'])
tracks_writer.writerow(['track_uri', 'track_name', 'artist_uri', 'artist_name', 'album_uri', 'album_name', 'duration_ms'])

# Store unique tracks and avoid repetitions when storing into tracks.csv
unique_tracks = set()

# Browse JSON files in directory
for mpd_slice in listdir(input_dir):
    if mpd_slice.endswith('.json'):
        with open(path.join(input_dir, mpd_slice), encoding='utf8') as json_file:
            print(f"Lecture du fichier {mpd_slice}...")
            json_slice = json.load(json_file)

            # For each playlist in the json file
            for playlist in json_slice['playlists']:
                playlists_writer.writerow([
                    playlist['pid'], playlist['name'], playlist['collaborative'],
                    playlist.get('num_tracks', 0), playlist.get('num_artists', 0),
                    playlist.get('num_albums', 0), playlist.get('num_followers', 0),
                    playlist.get('num_edits', 0), playlist['modified_at'],
                    playlist.get('duration_ms', 0)
                ])

                # If a description exists, write it in playlists_descr.csv
                if 'description' in playlist:
                    playlists_descr_writer.writerow([playlist['pid'], playlist['description']])

                # For each song in the playlist
                for track in playlist['tracks']:
                    items_writer.writerow([playlist['pid'], track['pos'], track['track_uri']])

                    # If the song is unique, add it to tracks.csv
                    if track['track_uri'] not in unique_tracks:
                        unique_tracks.add(track['track_uri'])
                        tracks_writer.writerow([
                            track['track_uri'], track['track_name'],
                            track['artist_uri'], track['artist_name'],
                            track['album_uri'], track['album_name'], track['duration_ms']
                        ])

# Fermer tous les fichiers
items_file.close()
playlists_file.close()
tracks_file.close()
playlists_descr_file.close()

print("Conversion complete! CSV files are available in the output folder.")
