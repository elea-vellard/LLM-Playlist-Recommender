#############################################
# Code to add the exact matches percentages #
#############################################

import os
import csv
from collections import Counter

def analyze_clusters_with_exact_matches(input_file, output_file):

    cluster_titles = {}
    exact_match_percentages = {}

    #Add the playlists' titles for each cluster
    with open(input_file, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cluster_id = row["Cluster ID"]
            title = row["Playlist Title"]
            if cluster_id not in cluster_titles:
                cluster_titles[cluster_id] = []
            cluster_titles[cluster_id].append(title)
            
    #Compute exact match percentages (for the most occuring title in the cluster)
    for cluster_id, titles in cluster_titles.items():
        title_counts = Counter(titles)  #Count occurrences of each title
        most_frequent_count = max(title_counts.values())  #Get the max count
        total_titles = len(titles)  #Total number of playlists in the cluster
        exact_match_percentages[cluster_id] = (most_frequent_count / total_titles) * 100

    #ChatGPT generated: write the exact matches in the csv file
    with open(input_file, 'r', encoding='utf8') as f_in, \
         open(output_file, 'w', newline='', encoding='utf8') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        # Write the header
        header = next(reader)
        new_header = header[:3] + ["Exact Match Percentage"] + header[3:]
        writer.writerow(new_header)

        # Write the rows with the new column
        for row in reader:
            cluster_id = row[0]  # Assuming "Cluster ID" is the first column
            exact_match = exact_match_percentages[cluster_id]
            new_row = row[:3] + [f"{exact_match:.2f}"] + row[3:]
            writer.writerow(new_row)

# Main function
def main():
    input_file = "/home/vellard/malis/clustering-no-split/clusters/200/clusters.csv"
    output_file = "/home/vellard/malis/clustering-no-split/clusters/200/clusters_with_exact_matches.csv"

    analyze_clusters_with_exact_matches(input_file, output_file )
    
    print("Clusters with percentages saved to {output_file}")

# Main function
def main():
    input_dir = "/home/vellard/playlist_continuation/clustering-no-split/clusters/200/"
    output_dir = "/home/vellard/playlist_continuation/clustering-no-split/analysis/200/"
    os.makedirs(output_dir, exist_ok=True)

    analyze_clusters_with_exact_matches(input_dir, output_dir)
    
    print("Analysis completed. Enriched CSV files saved to:", output_dir)
    '''

    for split in ["train", "val", "test"]:
        input_csv = os.path.join(input_dir, f"clusters_{split}.csv")
        output_csv = os.path.join(output_dir, f"clusters_{split}_percent.csv")

        print(f"Processing {split} split...")
        compute_exact_match_percentage(input_csv, output_csv)'''

if __name__ == "__main__":
    main()


