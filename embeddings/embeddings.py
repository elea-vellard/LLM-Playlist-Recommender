import os
import csv
import torch
import pickle
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

def load_bert_model(finetuned_model_dir):
    tokenizer = BertTokenizer.from_pretrained(finetuned_model_dir)
    model = BertModel.from_pretrained(finetuned_model_dir)
    model.eval()
    return tokenizer, model

def get_embedding(text, tokenizer, model, max_length=128):
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()
    return embedding

def compute_and_save_embeddings_with_metadata(tracks_csv, output_file, tokenizer, model):
    song_embeddings = {}
    with open(tracks_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        total_tracks = sum(1 for _ in reader)  
        f.seek(0)  
        next(reader) 

        for row in tqdm(reader, total=total_tracks, desc="Computing Embeddings with Metadata", unit="track"):
            track_name = row["track_name"]  
            artist_name = row["artist_name"]  
            track_uri = row["track_uri"]  
            if track_name not in song_embeddings:  
                embedding = get_embedding(track_name, tokenizer, model)
                song_embeddings[track_name] = {
                    "embedding": embedding,
                    "artist": artist_name,
                    "uri": track_uri,
                }

    with open(output_file, 'wb') as f:
        pickle.dump(song_embeddings, f)

#########################
# Main Function         #
#########################
def main():

    tracks_csv = "/data/playlist_continuation_data/csvs/tracks.csv"
    output_file = "/home/vellard/playlist_continuation/embeddings/embeddings.pkl"
    finetuned_model_dir = "/home/vellard/playlist_continuation/finetuned-model"

    tokenizer, model = load_bert_model(finetuned_model_dir)

    compute_and_save_embeddings_with_metadata(tracks_csv, output_file, tokenizer, model)

    print(f"Embeddings saved with success to {output_file}.")

if __name__ == "__main__":
    main()
