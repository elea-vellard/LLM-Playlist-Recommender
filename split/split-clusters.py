import os
import csv
import random
from os import path
from tqdm import tqdm

# Sets the seed for reproducibility of results
random.seed(1)

# I/O paths
input_dir = '/home/vellard/playlist_continuation/clustering-no-split/clean/test/'
output_dir = '/home/vellard/playlist_continuation/clustering-no-split/split/'
os.makedirs(output_dir, exist_ok=True)

# I/O files
input_clusters_file = path.join(input_dir, 'small.csv')

output_clusters_train = path.join(output_dir, 'clusters_train.csv')
output_clusters_val = path.join(output_dir, 'clusters_val.csv')
output_clusters_test = path.join(output_dir, 'clusters_test.csv')

# Load cluster data
clusters = {}

# Reading clusters.csv to load playlists into `clusters`
with open(input_clusters_file, 'r', newline='', encoding='utf8') as clusters_file:
    clusters_reader = csv.DictReader(clusters_file)
    headers = clusters_reader.fieldnames

    for row in tqdm(clusters_reader, desc="Reading clusters.csv", unit="row"):
        cluster_id = row["Cluster ID"]
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(row)

# Splitting clusters into training (80%), validation (10%), and test (10%) sets
cluster_ids = list(clusters.keys())
num_validation = int(0.1 * len(cluster_ids))
num_test = int(0.1 * len(cluster_ids))
num_train = len(cluster_ids) - num_validation - num_test

# Shuffle the cluster IDs and split
random.shuffle(cluster_ids)
validation_clusters = cluster_ids[:num_validation]
test_clusters = cluster_ids[num_validation:num_validation + num_test]
train_clusters = cluster_ids[num_validation + num_test:]

# Writing training, validation, and test sets into output files
with open(output_clusters_train, 'w', newline='', encoding='utf8') as train_file, \
     open(output_clusters_val, 'w', newline='', encoding='utf8') as val_file, \
     open(output_clusters_test, 'w', newline='', encoding='utf8') as test_file:

    train_writer = csv.DictWriter(train_file, fieldnames=headers)
    val_writer = csv.DictWriter(val_file, fieldnames=headers)
    test_writer = csv.DictWriter(test_file, fieldnames=headers)

    train_writer.writeheader()
    val_writer.writeheader()
    test_writer.writeheader()

    for cluster_id in tqdm(train_clusters, desc="Writing train clusters", unit="cluster"):
        for row in clusters[cluster_id]:
            train_writer.writerow(row)

    for cluster_id in tqdm(validation_clusters, desc="Writing validation clusters", unit="cluster"):
        for row in clusters[cluster_id]:
            val_writer.writerow(row)

    for cluster_id in tqdm(test_clusters, desc="Writing test clusters", unit="cluster"):
        for row in clusters[cluster_id]:
            test_writer.writerow(row)

print("The training, validation, and test cluster files were generated and stored in the output folder.")
