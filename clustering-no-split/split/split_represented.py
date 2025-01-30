import os
import csv
import random
from os import path
from tqdm import tqdm

# Sets the seed for reproducibility of results
random.seed(1)

# Paths and files
input_dir = '/home/vellard/playlist_continuation/clustering-no-split/clean/200/'
output_dir = '/home/vellard/playlist_continuation/clustering-no-split/split/represented/'
os.makedirs(output_dir, exist_ok=True)
input_clusters_file = path.join(input_dir, 'clusters_with_exact_matches.csv')
output_clusters_train = path.join(output_dir, 'clusters_train.csv')
output_clusters_val = path.join(output_dir, 'clusters_val.csv')
output_clusters_test = path.join(output_dir, 'clusters_test.csv')

# Load cluster data
clusters = {}

print(f"Reading clusters from: {input_clusters_file}")
with open(input_clusters_file, 'r', newline='', encoding='utf8') as clusters_file:
    clusters_reader = csv.DictReader(clusters_file)
    headers = clusters_reader.fieldnames

    for row in tqdm(clusters_reader, desc="Reading clusters.csv", unit="row"):
        cluster_id = row["Cluster ID"]
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(row)

print(f"Total clusters found: {len(clusters)}")

#Open output files + DictWriter
train_file = open(output_clusters_train, 'w', newline='', encoding='utf8')
val_file   = open(output_clusters_val,   'w', newline='', encoding='utf8')
test_file  = open(output_clusters_test,  'w', newline='', encoding='utf8')

train_writer = csv.DictWriter(train_file, fieldnames=headers)
val_writer   = csv.DictWriter(val_file,   fieldnames=headers)
test_writer  = csv.DictWriter(test_file,  fieldnames=headers)

# headers
train_writer.writeheader()
val_writer.writeheader()
test_writer.writeheader()

#For each cluster, we do a local 80/10/10 split
for cluster_id, rows in tqdm(clusters.items(), desc="Splitting each cluster", unit="cluster"):
    random.shuffle(rows)
    nb_total = len(rows)

    # Calculate how many go to val/test
    nb_val  = int(0.1 * nb_total)
    nb_test = int(0.1 * nb_total)
    nb_train = nb_total - nb_val - nb_test

    train_rows = rows[:nb_train]
    val_rows   = rows[nb_train : nb_train + nb_val]
    test_rows  = rows[nb_train + nb_val : ]

    # Write them
    for r in train_rows:
        train_writer.writerow(r)
    for r in val_rows:
        val_writer.writerow(r)
    for r in test_rows:
        test_writer.writerow(r)

# Close files
train_file.close()
val_file.close()
test_file.close()

print(f"Train CSV : {output_clusters_train}")
print(f"Val CSV   : {output_clusters_val}")
print(f"Test CSV  : {output_clusters_test}")

