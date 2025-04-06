import os
import torch
import pickle
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

###########################
# Functions Definitions   #
###########################

def load_fine_tuned_model(model_dir, base_model_name='sentence-transformers/all-MiniLM-L6-v2'):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, output_hidden_states=True)
    model.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    return tokenizer, model, device

def get_embedding(text, tokenizer, model, device):
    if not isinstance(text, str) or pd.isna(text):
        text = ""

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden_state = outputs.hidden_states[-1]
        embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    return embedding

def load_playlist_titles(playlists_csv):
    if not os.path.exists(playlists_csv):
        raise FileNotFoundError(f"CSV not found: {playlists_csv}")
    df = pd.read_csv(playlists_csv)
    df['name'] = df['name'].fillna('')
    pid_to_title = dict(zip(df['pid'], df['name']))
    return pid_to_title

def compute_and_save_playlist_embeddings(playlists_csv, output_file, tokenizer, model, device):
    playlist_embeddings = {}
    pid_to_title = load_playlist_titles(playlists_csv)

    problematic_pids = []
    for pid, title in tqdm(pid_to_title.items(), desc="Computing embeddings", unit="playlist"):
        try:
            embedding = get_embedding(title, tokenizer, model, device)
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

##################
# Main Function  #
##################

def main():
    playlists_csv = "/data/csvs/playlists.csv"
    output_file = "/home/vellard/playlist_continuation/playlists_embeddings/final_embeddings/playlists_embeddings_scheduler.pkl"
    # Choose the  model directory
    finetuned_model_dir = "/home/vellard/playlist_continuation/fine_tuned_model_no_scheduler_2"

    tokenizer, model, device = load_fine_tuned_model(finetuned_model_dir)
    print("Loaded fine-tuned classification model (with updated weights).")

    compute_and_save_playlist_embeddings(playlists_csv, output_file, tokenizer, model, device)

if __name__ == "__main__":
    main()
