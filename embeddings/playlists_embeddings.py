######################################################
# Code tp precompute the playlists titles embeddings #
######################################################

#Using sentence bert inteas of bert
import os
import csv
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def load_sentence_bert_model(finetuned_model_dir):
    model = SentenceTransformer(finetuned_model_dir)
    model.eval()
    return model

def get_embedding(text, model):
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding

def compute_and_save_playlist_embeddings(playlists_csv, output_file, model):
    playlist_embeddings = {}
    with open(playlists_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        total_playlists = sum(1 for _ in reader)
        f.seek(0)
        next(reader)

        for row in tqdm(reader, total=total_playlists, desc="Computing Playlist Embeddings", unit="playlist"):
            pid = row["pid"]
            title = row["name"]
            if pid not in playlist_embeddings:
                embedding = get_embedding(title, model)
                playlist_embeddings[pid] = {
                    "embedding": embedding,
                    "title": title,
                }

    with open(output_file, 'wb') as f:
        pickle.dump(playlist_embeddings, f)

    print(f"Playlist embeddings saved successfully to {output_file}.")

def main():
    playlists_csv = "/data/playlist_continuation_data/csvs/playlists.csv"
    output_file = "/home/vellard/playlist_continuation/embeddings/playlists_embeddings.pkl"
    finetuned_model_dir = "/home/vellard/playlist_continuation/finetuned-model"

    model = load_sentence_bert_model(finetuned_model_dir)
    compute_and_save_playlist_embeddings(playlists_csv, output_file, model)

if __name__ == "__main__":
    main()
