# This script generates playlists based on a five-shot prompt using various LLMs.
# The same code has been adusted to do the zero and one-shot prompts. 

###########
# Imports #
###########

import argparse
import argparse
from html import parser
import logging
import os
import re
import yaml # type: ignore
from openai import OpenAI # type: ignore
import time
from tqdm import tqdm # type: ignore
import json
# from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.prompts.prompt import PromptTemplate # type: ignore
from data_loader import load_playlists_yaml, load_playlists_csv, load_all_playlist_data

items_csv = "./data/items.csv"
tracks_csv = "./data/tracks.csv"
test_set = "./data/clusters_test.csv"

vllm_config = {
    "base_url": "http://helix01:8000",
    "max_tokens": 16384,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "enable_thinking": True,
    "thinking_budget_tokens": 8192
}


# GPU environment and cache configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/huggingface/"

# All LLMs available for use
# available_llms = {
#     "zephyr": "TheBloke/zephyr-7B-beta-AWQ",
#     "dpo": "yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B",
#     "una": "fblgit/UNA-TheBeagle-7b-v1",
#     "solar": "bhavinjawade/SOLAR-10B-OrcaDPO-Jawade",
#     "gpt4": "OpenAI-GPT4",
#     "llama3.1": "Ollama/llama3.1"  
# }

available_llms = {
    "gemma": "google/gemma-4-31B-it",
    "qwen": "Qwen/Qwen3.6-35B-A3B"
}

# Set up logging
loggingFormatString = "%(asctime)s:%(levelname)s:%(threadName)s:%(funcName)s:%(message)s"
logging.basicConfig(format=loggingFormatString, level=logging.INFO)

########################
# Functions definition #
########################

# Load the prompt template from a YAML file
def load_prompt_template(template_path):
    with open(template_path, 'r') as file:
        template_data = yaml.safe_load(file)
    return PromptTemplate(input_variables=template_data['input'], template=template_data['template'])

## Call the GPT-4 API with retries
def call_gpt4_api(prompt, model, max_retries=3, timeout=1200):
    client = OpenAI(
        base_url=vllm_config["base_url"] + "/v1",
        api_key=os.getenv("API_KEY"))
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=available_llms[model],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16_384,
                temperature=1.0,
                top_p=0.95,
                extra_body={
                    "top_k": 64,
                    "chat_template_kwargs": {
                        "enable_thinking": True,
                    },
                    "thinking_token_budget": 8_192,
                },
            )
            return str(response.choices[0].message.content).strip()

        except Exception as e:
            print(f"An error occurred: {e}")
            retries += 1
    return None

# Call the Hugging Face model with GPU support
# def call_huggingface_model(prompt, tokenizer, model):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)
#     inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
#     # Increase max_new_tokens to ensure you get the whole answer
#     outputs = model.generate(**inputs, max_new_tokens=600, eos_token_id=tokenizer.eos_token_id, temperature=0.8)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#     return response

# Generate a playlist using the five-shot prompt
def generate_playlist_five_shot(seed_playlist, llm_model, template_path, verbose, model_name):
    PROMPT_TEMPLATE = load_prompt_template(template_path)
    tracks = seed_playlist["tracks"]
    input_data = {
        "playlist_title": seed_playlist["playlist_title"],
    }

    print(tracks)
    for i in range(5):
        track = tracks[i] if len(tracks) > i else {"song": "Unknown Song", "artist": "Unknown Artist"}
        if 'song' not in track or 'artist' not in track:
            track = {"song": track[0], "artist": track[1]}
        input_data[f"song{i+1}"] = track['song']
        input_data[f"artist{i+1}"] = track["artist"]

    formatted_prompt = PROMPT_TEMPLATE.format(**input_data)
    if verbose:
        print("Full prompt:", formatted_prompt)
    res = str(call_gpt4_api(formatted_prompt, llm_model))
    res = res.replace("Example:\s", "").strip()
    if verbose:
        print("Raw Model Output:", res)

    try:
        json_block = re.search(r'\[\s*{[\s\S]+?}\s*\]', res)
        if json_block:
            extracted = json_block.group()
        else:
            print("/!\ No JSON blocks found in the response.")
            return []
        if not extracted.endswith(']'):
            print("Incomplete JSON detected, closing block...")
            extracted = extracted.rstrip() + "]"
        generated_playlist = json.loads(extracted)
        return generated_playlist
    except json.JSONDecodeError as e:
        print("JSON parsing error:", e)
        print("Captured content :", extracted if 'extracted' in locals() else "N/A")
        return []


def main(llm_model, yaml_path=None, template_path='./subset_22.yml', verbose = True):
    # List of 22 playlists to generate
    # Each playlist contains a title and 5 songs with their respective artists

    if yaml_path is not None and os.path.exists(yaml_path):  
        print(f"Loading playlists from {yaml_path}...")
        playlists = load_playlists_yaml(yaml_path)
    else:
        print(f"YAML file {args.yaml} not available. Processing the entire test set.")
        playlist_tracks = load_all_playlist_data(items_csv, tracks_csv)

        playlists = load_playlists_csv(test_set)
        for playlist in playlists:
            playlist['tracks'] = playlist_tracks.get(playlist['pid'], [])[0:5]

    output_dir = "out_llm_prediction"
    os.makedirs(output_dir, exist_ok=True)

    for pl in tqdm(playlists, desc="Generating playlists"):
        output_path = os.path.join(output_dir, pl["pid"] + ".yml")
        if os.path.exists(output_path):
            if verbose:
                print(f"{pl['pid']} | {pl['playlist_title']} | SKIPPING")
            continue  # Skip if the output file already exists
        elif verbose:
                print(f"{pl['pid']} | {pl['playlist_title']} | GENERATING")

        tracks = generate_playlist_five_shot(
            pl,
            llm_model,
            template_path,
            verbose,
            llm_model
        )
        generated_playlist = {
            "pid": pl["pid"],
            "playlist_title": pl["playlist_title"],
            "tracks": tracks
        }
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(generated_playlist, f, allow_unicode=True, sort_keys=False)

    output_filename = "total.json"
    output_path = os.path.join(output_dir, output_filename)

    generated_data = []
    for filename in sorted([x for x in os.listdir(output_dir) if x.endswith(".yml")]):
        with open(os.path.join(output_dir, filename), "r", encoding="utf-8") as f:
            parsed_playlist = yaml.safe_load(f) or []
            generated_data.append(parsed_playlist)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=4)

    print(f"\nThe JSON file was generated in : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate playlists using a five-shot prompt with various LLMs.")
    parser.add_argument("--template_path", type=str, default="LLM_part/prompt_template_five_shot.yml", help="Path to the prompt template YAML file.")
    parser.add_argument("--llm_model", type=str, default="gemma", choices=available_llms.keys())
    parser.add_argument(
            '--yaml', type=str,
            help="Path to playlists YAML file (for playlist titles and PIDs)"
        )
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output for debugging.")
    args = parser.parse_args()
    main(args.llm_model, args.yaml, args.template_path, args.verbose)
