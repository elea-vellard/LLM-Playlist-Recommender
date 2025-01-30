import csv
from tqdm import tqdm

def normalize_song(song):
    """
    Normalize a song tuple (track_name, artist_name) by stripping spaces and converting to lowercase.
    """
    return (song[0].strip().lower(), song[1].strip().lower())

def compute_metrics(recommended_songs, relevant_songs, top_n=10):
    """
    Compute evaluation metrics: HIT@N, Precision@N, Recall@N, MRR@N.
    """
    # Normalize both lists of songs for accurate comparison
    normalized_recommended = [normalize_song(song) for song in recommended_songs]
    normalized_relevant = [normalize_song(song) for song in relevant_songs]

    # Compute HIT@N
    relevant_hits = [song for song in normalized_recommended[:top_n] if song in normalized_relevant]

    hits = len(relevant_hits)
    print(hits)
    hit_score = hits / min(top_n, len(normalized_relevant))  # Normalized

    # Compute Precision@N
    precision = hits / len(normalized_recommended[:top_n])

    # Compute Recall@N
    recall = hits / len(normalized_relevant)

    # Compute MRR@N
    mrr = 0.0
    for i, song in enumerate(normalized_recommended[:top_n]):
        if song in normalized_relevant:
            mrr = 1 / (i + 1)
            break

    print(f"HIT@{top_n}: {hit_score:.4f}")
    print(f"Precision@{top_n}: {precision:.4f}")
    print(f"Recall@{top_n}: {recall:.4f}")
    print(f"MRR@{top_n}: {mrr:.4f}")

    return {"HIT@N": hit_score, "Precision@N": precision, "Recall@N": recall, "MRR@N": mrr}

def load_relevant_songs(items_csv, tracks_csv):
    """
    Load the relevant songs (actual tracks in the playlist) from the dataset.
    """
    track_metadata = {}
    with open(tracks_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading track metadata", unit="track"):
            track_metadata[row["track_uri"]] = {
                "track_name": row["track_name"],
                "artist_name": row["artist_name"],
            }

    playlist_tracks = {}
    with open(items_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading playlist tracks", unit="playlist"):
            pid = row["pid"]
            track_uri = row["track_uri"]
            if pid not in playlist_tracks:
                playlist_tracks[pid] = []
            if track_uri in track_metadata:
                playlist_tracks[pid].append((track_metadata[track_uri]["track_name"], track_metadata[track_uri]["artist_name"]))

    return playlist_tracks

def load_generated_songs(txt_file):
    """
    Load the generated songs from the TXT file.
    """
    with open(txt_file, 'r', encoding='utf8') as f:
        lines = f.readlines()

    pid = lines[0].strip()  # First line is the PID
    generated_songs = []

    for line in lines[1:]:
        if line.strip():  # Ignore empty lines
            parts = line.split('. ', 1)  # Split to get song and artist
            if len(parts) == 2:
                song_artist = parts[1].strip().split(', ', 1)
                if len(song_artist) == 2:
                    song, artist = song_artist
                    generated_songs.append((song.strip(), artist.strip()))

    return pid, generated_songs

def main():
    items_csv = "/data/playlist_continuation_data/csvs/items.csv"
    tracks_csv = "/data/playlist_continuation_data/csvs/tracks.csv"
    txt_file = "/home/vellard/playlist_continuation/similarity/songs-llama2.txt"

    # Load relevant songs from the dataset
    playlist_tracks = load_relevant_songs(items_csv, tracks_csv)

    # Load generated songs from the TXT file
    pid, generated_songs = load_generated_songs(txt_file)

    # Get relevant songs for the given PID
    relevant_songs = playlist_tracks.get(pid, [])
    relevant_songs = list(set(relevant_songs))  # Remove duplicates

    # Compute metrics
    metrics = compute_metrics(generated_songs, relevant_songs, top_n=66)

    print("\nGenerated Songs:", generated_songs)
    print("\nRelevant Songs:", relevant_songs)
    print("\nMetrics:", metrics)

if __name__ == "__main__":
    main()
