import csv
import random
import os
from os import path

# Sets the seed for reproducibility of results
random.seed(1)

# I/O paths
input_dir = '/data/playlist_continuation_data/csvs'
output_dir = '/data/playlist_continuation_data/split_csvs'
os.makedirs(output_dir, exist_ok=True)

# I/O files
input_playlists = path.join(input_dir, 'playlists.csv')
input_items = path.join(input_dir, 'items.csv')

output_playlists_train = path.join(output_dir, 'playlists_train.csv')
output_playlists_val = path.join(output_dir, 'playlists_val.csv')
output_playlists_test = path.join(output_dir, 'playlists_test.csv')
output_items_train = path.join(output_dir, 'items_train.csv')
output_items_val = path.join(output_dir, 'items_val.csv')
output_items_test = path.join(output_dir, 'items_test.csv')

# Loading playlist and track data
playlists = {}
items = {}
playlists_pid = []

# Reading playlists.csv to load playlists into `playlists` and `playlists_pid`
with open(input_playlists, 'r', newline='', encoding='utf8') as playlists_file:
    playlists_reader = csv.reader(playlists_file)
    next(playlists_reader)  # Ignore the header
    for playlist in playlists_reader:
        pid = playlist[0]
        playlists[pid] = playlist
        playlists_pid.append(pid)

# Reading items.csv to load the songs associated with each playlist into `items`
with open(input_items, 'r', newline='', encoding='utf8') as items_file:
    items_reader = csv.reader(items_file)
    next(items_reader)  # Ignore the header
    for item in items_reader:
        pid = item[0]
        # Adding the track to the list
        if pid in items:
            items[pid].append(item)
        else:
            items[pid] = [item]

# Splitting the data into training (80%), validation (10%), and test (10%) sets
num_validation = int(0.1 * len(playlists_pid))
num_test = int(0.1 * len(playlists_pid))
num_train = len(playlists_pid) - num_validation - num_test

# Shuffle the playlist IDs and split
random.shuffle(playlists_pid)
validation_playlists = playlists_pid[:num_validation]
test_playlists = playlists_pid[num_validation:num_validation + num_test]
train_playlists = playlists_pid[num_validation + num_test:]

# Writing training, validation, and test sets into output files
with open(output_playlists_train, 'w', newline='', encoding='utf8') as train_file, \
     open(output_playlists_val, 'w', newline='', encoding='utf8') as val_file, \
     open(output_playlists_test, 'w', newline='', encoding='utf8') as test_file, \
     open(output_items_train, 'w', newline='', encoding='utf8') as items_train_file, \
     open(output_items_val, 'w', newline='', encoding='utf8') as items_val_file, \
     open(output_items_test, 'w', newline='', encoding='utf8') as items_test_file:

    train_writer = csv.writer(train_file)
    val_writer = csv.writer(val_file)
    test_writer = csv.writer(test_file)
    items_train_writer = csv.writer(items_train_file)
    items_val_writer = csv.writer(items_val_file)
    items_test_writer = csv.writer(items_test_file)

    # Headers
    headers = ['pid', 'name', 'collaborative', 'num_tracks', 'num_artists', 'num_albums', 'num_followers', 'num_edits', 'modified_at', 'duration_ms']
    train_writer.writerow(headers)
    val_writer.writerow(headers)
    test_writer.writerow(headers)
    items_train_writer.writerow(['pid', 'track_position', 'track_uri'])
    items_val_writer.writerow(['pid', 'track_position', 'track_uri'])
    items_test_writer.writerow(['pid', 'track_position', 'track_uri'])

    for pid, playlist in playlists.items():
        if pid in validation_playlists:
            val_writer.writerow(playlist)
            for item in items.get(pid, []):
                items_val_writer.writerow(item)
        elif pid in test_playlists:
            test_writer.writerow(playlist)
            for item in items.get(pid, []):
                items_test_writer.writerow(item)
        else:
            train_writer.writerow(playlist)
            for item in items.get(pid, []):
                items_train_writer.writerow(item)

print("The training, validation, and test sets were generated and stored in the output folder.")
