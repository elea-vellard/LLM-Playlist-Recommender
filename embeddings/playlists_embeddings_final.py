# Code to compute the embeddigns of playlists using the fine-tuned model (adapted from transformers library)

import os
import csv
import pickle
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

########################
# Functions definition #
########################

def load_fine_tuned_model(model_dir, base_model_name='sentence-transformers/all-MiniLM-L6-v2'):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)#load the tokenizer from the pre-trained model

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

def get_embedding(text, tokenizer, model):#from the transformers library

    if not isinstance(text, str):
        if pd.isna(text) or text is None:
            text = ""
        else:
            text = str(text)

    with torch.no_grad():
        # Tokenize
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        # outputs.hidden_states[-1] is shape: (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs.hidden_states[-1]

        #mean pooling
        embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    return embedding

def load_playlist_titles(playlists_csv):
    if not os.path.exists(playlists_csv):
        raise FileNotFoundError(f"CSV not found: {playlists_csv}")
    df = pd.read_csv(playlists_csv)
    df['name'] = df['name'].fillna('')
    pid_to_title = dict(zip(df['pid'], df['name']))
    return pid_to_title

def compute_and_save_playlist_embeddings(playlists_csv, output_file, tokenizer, model):
    playlist_embeddings = {}
    pid_to_title = load_playlist_titles(playlists_csv)

    problematic_pids = []
    for pid, title in tqdm(pid_to_title.items(), desc="Computing Playlist Embeddings", unit="playlist"):
        try:
            embedding = get_embedding(title, tokenizer, model)
            playlist_embeddings[pid] = {
                "embedding": embedding,
                "title": title,
            }
        except Exception as e:
            problematic_pids.append(pid)
            print(f"Erreur pour pid {pid}: {e}")

    with open(output_file, 'wb') as f:
        pickle.dump(playlist_embeddings, f)
    print(f"Playlist embeddings saved successfully to {output_file}.")

    if problematic_pids:
        problem_file = os.path.splitext(output_file)[0] + "_problematic_pids.pkl"
        with open(problem_file, 'wb') as f:
            pickle.dump(problematic_pids, f)
        print(f"Problematic pids saved in {problem_file}")

########
# Main #
########

def main():
    playlists_csv = "/data/playlist_continuation_data/csvs/playlists.csv"
    output_file = "/home/vellard/playlist_continuation/embeddings/new-model/playlists_embeddings.pkl"
    finetuned_model_dir = "/home/vellard/playlist_continuation/finetuning/fine_tuned_model/checkpoint-295830"
    base_model_name = 'sentence-transformers/all-MiniLM-L6-v2'

    tokenizer, model = load_fine_tuned_model(finetuned_model_dir, base_model_name)
    print("Loaded fine-tuned classification model (with updated weights).")

    compute_and_save_playlist_embeddings(playlists_csv, output_file, tokenizer, model)

if __name__ == "__main__":
    main()
