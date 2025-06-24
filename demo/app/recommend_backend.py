import sys
import csv
import torch
import pickle
from flask import Flask, request, jsonify, render_template
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from gensim.models import KeyedVectors
import numpy as np

csv.field_size_limit(sys.maxsize)
app = Flask(__name__)

# -------------------------------------------------------------------
# 1 · Paths to model, embeddings, and CSV files
# -------------------------------------------------------------------
MODEL_DIR        = "/app/data/fine_tuned_model_no_scheduler_2"
EMBEDDINGS_FILE  = "/app/data/playlists_embeddings_scheduler.pkl"

TRACKS_CSV       = "/app/data/tracks.csv"
ITEMS_CSV        = "/app/data/items.csv"
PLAYLISTS_CSV    = "/app/data/playlists.csv"

# -------------------------------------------------------------------
# 2 · Global variables (loaded once)
# -------------------------------------------------------------------
_loaded = None
track_meta = {}
playlist_tracks = {}

# -------------------------------------------------------------------
# 3 · Static CSV loading (tracks + items)
# -------------------------------------------------------------------
print("Loading track metadata...")
with open(TRACKS_CSV, "r", encoding="utf8") as f:
    reader = csv.DictReader(f)
    for r in tqdm(reader, desc="tracks"):
        track_meta[r["track_uri"]] = {
            "track_name":  r["track_name"],
            "artist_name": r["artist_name"],
            "track_uri":   r["track_uri"]
        }

print("Loading playlist → track mapping...")
with open(ITEMS_CSV, "r", encoding="utf8") as f:
    reader = csv.DictReader(f)
    for r in tqdm(reader, desc="items"):
        pid_str = r["pid"].strip()
        uri = r["track_uri"]
        if uri in track_meta:
            playlist_tracks.setdefault(pid_str, []).append(track_meta[uri])

print("Static data loaded.\n")

# -------------------------------------------------------------------
# 4 · Load model, tokenizer, and embeddings (once)
# -------------------------------------------------------------------
def load():
    global _loaded
    if _loaded:
        return _loaded

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModel.from_pretrained(MODEL_DIR)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with open(EMBEDDINGS_FILE, "rb") as f:
        embdict = pickle.load(f)

    kv = KeyedVectors(vector_size=384)
    keys = list(embdict.keys())
    vectors = [embdict[k]["embedding"].astype(np.float32, copy=False) for k in keys]
    kv.add_vectors(keys, vectors)

    _loaded = (tokenizer, model, kv)
    return _loaded

# Force preloading at startup
print("Preloading tokenizer and model...")
load()
print("Model ready.\n")

# -------------------------------------------------------------------
# 5 · Encode playlist name → 384-D embedding
# -------------------------------------------------------------------
def embed_name(name: str, tokenizer, model) -> np.ndarray:
    with torch.no_grad():
        inputs = tokenizer(name, return_tensors="pt", truncation=True, padding=True).to(model.device)
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb

# -------------------------------------------------------------------
# 6 · Retrieve top-50 similar playlists
# -------------------------------------------------------------------
def find_similar(name: str, kv: KeyedVectors, tokenizer, model, topk: int = 50):
    emb = embed_name(name, tokenizer, model)
    return kv.similar_by_vector(emb, topn=topk)

# -------------------------------------------------------------------
# 7 · Aggregate top tracks by frequency
# -------------------------------------------------------------------
def top_tracks(similar_playlists, topk: int = 10):
    counter = Counter()
    for pid, _ in similar_playlists:
        pid_str = str(pid)
        for track in playlist_tracks.get(pid_str, []):
            uri = track["track_uri"]
            counter[uri] += 1
    return counter.most_common(topk)

# -------------------------------------------------------------------
# 8 · Flask routes
# -------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend")
def recommend():
    name = request.args.get("playlist_name", "").strip()
    if not name:
        return jsonify({"error": "playlist_name required"}), 400

    tokenizer, model, kv = load()
    sim = find_similar(name, kv, tokenizer, model)
    topk_list = top_tracks(sim, topk=10)

    recommendations = []
    for uri, count in topk_list:
        meta = track_meta.get(uri, {})
        recommendations.append({
            "song":   meta.get("track_name", "Unknown"),
            "artist": meta.get("artist_name", "Unknown"),
            "uri":    uri,
            "count":  count
        })

    return jsonify({"recommendations": recommendations})

# -------------------------------------------------------------------
# 9 · Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
