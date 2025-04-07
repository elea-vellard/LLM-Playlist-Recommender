# This script generates playlists based on a five-shot prompt using various LLMs.
# The same code has been adusted to do the zero and one-shot prompts. 

###########
# Imports #
###########

import logging
import os
import re
import yaml # type: ignore
import openai # type: ignore
import time
import torch # type: ignore
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.prompts.prompt import PromptTemplate # type: ignore

# GPU environment and cache configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/huggingface/"

# All LLMs available for use
available_llms = {
    "zephyr": "TheBloke/zephyr-7B-beta-AWQ",
    "dpo": "yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B",
    "una": "fblgit/UNA-TheBeagle-7b-v1",
    "solar": "bhavinjawade/SOLAR-10B-OrcaDPO-Jawade",
    "gpt4": "OpenAI-GPT4",
    "llama3.1": "Ollama/llama3.1"  
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
def call_gpt4_api(prompt, api_key, max_retries=3, timeout=1200):
    client = openai.OpenAI(api_key=api_key)
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                timeout=timeout
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"An error occurred: {e}")
            retries += 1
    return None

# Call the Hugging Face model with GPU support
def call_huggingface_model(prompt, tokenizer, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    # Increase max_new_tokens to ensure you get the whole answer
    outputs = model.generate(**inputs, max_new_tokens=600, eos_token_id=tokenizer.eos_token_id, temperature=0.8)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return response

# Generate a playlist using the five-shot prompt
def generate_playlist_five_shot(playlist_title, song1, artist1, song2, artist2, song3, artist3, song4, artist4, song5, artist5, llm_model, template_path, api_key, verbose, model_name):
    PROMPT_TEMPLATE = load_prompt_template(template_path)
    input_data = {
        "playlist_title": playlist_title,
        "song1": song1,
        "artist1": artist1,
        "song2": song2,
        "artist2": artist2,
        "song3": song3,
        "artist3": artist3,
        "song4": song4,
        "artist4": artist4,
        "song5": song5,
        "artist5": artist5
    }
    formatted_prompt = PROMPT_TEMPLATE.format(**input_data)
    if verbose:
        print("Full prompt:", formatted_prompt)
    if llm_model == "gpt4":
        res = call_gpt4_api(formatted_prompt, api_key)
    else:
        tokenizer = AutoTokenizer.from_pretrained(available_llms[model_name])
        model = AutoModelForCausalLM.from_pretrained(available_llms[model_name], device_map='cuda')
        res = call_huggingface_model(formatted_prompt, tokenizer, model)
    print("Raw Model Output:", res)
    try:
        if "Example:" in res:
            json_block = re.search(r'Example:\s*(\[[\s\S]+)', res)
            if json_block:
                extracted = json_block.group(1).strip()
            else:
                print("/!\ No JSON block found after 'Example:'.")
                return []
        else:
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

def main():
    # Configuration
    template_path = "prompt_template_five_shot.yml"
    llm_model = "llama3.1"
    api_key = None
    verbose = True

    # List of 22 playlists to generate
    # Each playlist contains a title and 5 songs with their respective artists
    playlists = [
        {
            "pid": "673925",
            "playlist_title": "K-pop",
            "song1": "Ma city", "artist1": "BTS",
            "song2": "BTS Cypher 4", "artist2": "BTS",
            "song3": "Run", "artist3": "BTS",
            "song4": "Epilogue: Young Forever", "artist4": "BTS",
            "song5": "Converse High", "artist5": "BTS"
        },
        {
            "pid": "677580",
            "playlist_title": "workout music",
            "song1": "S&M", "artist1": "Rihanna",
            "song2": "I Know You Want Me (Calle Ocho)", "artist2": "Pitbull",
            "song3": "Miami 2 Ibiza - (Swedish House Mafia vs. Tinie Tempah)", "artist3": "Swedish House Mafia vs. Tinie Tempah",
            "song4": "Like A G6 - RedOne Remix", "artist4": "Far East Movement",
            "song5": "Stamp on the Ground - Radio Edit", "artist5": "Hardwell"
        },
        {
            "pid": "321143",
            "playlist_title": "Dance",
            "song1": "Runaway (U & I)", "artist1": "Galantis",
            "song2": "Addicted To A Memory", "artist2": "Zedd",
            "song3": "Done With Love", "artist3": "Unknown",
            "song4": "Follow You Down", "artist4": "Unknown",
            "song5": "Don't Let Me Down", "artist5": "The Chainsmokers"
        },
        {
            "pid": "923247",
            "playlist_title": "Rock",
            "song1": "Smells Like Teen Spirit", "artist1": "Nirvana",
            "song2": "Bright Lights", "artist2": "Unknown",
            "song3": "Born To Be Wild", "artist3": "Steppenwolf",
            "song4": "Paint It Black", "artist4": "The Rolling Stones",
            "song5": "Sympathy For The Devil", "artist5": "The Rolling Stones"
        },
        {
            "pid": "301195",
            "playlist_title": "Summer",
            "song1": "Stronger (What Doesn't Kill You)", "artist1": "Kelly Clarkson",
            "song2": "Part Of Me", "artist2": "Katy Perry",
            "song3": "International Love", "artist3": "Pitbull feat. Chris Brown",
            "song4": "Stereo Hearts (feat. Adam Levine)", "artist4": "Gym Class Heroes",
            "song5": "Party Rock Anthem", "artist5": "LMFAO"
        },
        {
            "pid": "490485",
            "playlist_title": "Hawaii",
            "song1": "Jamaica Mistaica", "artist1": "Jimmy Buffett",
            "song2": "Why Don't We Get Drunk", "artist2": "Jimmy Buffett",
            "song3": "Volcano", "artist3": "Jimmy Buffett",
            "song4": "Mele Kalikimaka", "artist4": "Jimmy Buffett",
            "song5": "Christmas Island", "artist5": "Jimmy Buffett"
        },
        {
            "pid": "575612",
            "playlist_title": "Classic Country",
            "song1": "Coat of Many Colors", "artist1": "Dolly Parton",
            "song2": "Jolene", "artist2": "Dolly Parton",
            "song3": "I Will Always Love You", "artist3": "Dolly Parton",
            "song4": "Here You Come Again", "artist4": "Dolly Parton",
            "song5": "It's Too Late to Love Me Now", "artist5": "Dolly Parton"
        },
        {
            "pid": "269088",
            "playlist_title": "older songs",
            "song1": "Billie Jean", "artist1": "Michael Jackson",
            "song2": "Smile Jamaica", "artist2": "Bob Marley & The Wailers",
            "song3": "One Drop", "artist3": "Bob Marley & The Wailers",
            "song4": "California Dreamin'", "artist4": "The Mamas & the Papas",
            "song5": "California Dreamin'", "artist5": "The Mamas & the Papas"
        },
        {
            "pid": "606436",
            "playlist_title": "2016",
            "song1": "Wine And Chocolates", "artist1": "Theophilus London",
            "song2": "Stressed Out", "artist2": "Twenty One Pilots",
            "song3": "Same Old Love", "artist3": "Selena Gomez",
            "song4": "Hands To Myself", "artist4": "Selena Gomez",
            "song5": "Don't Let Me Down", "artist5": "The Chainsmokers"
        },
        {
            "pid": "701866",
            "playlist_title": "Dance",
            "song1": "Telephone", "artist1": "Lady Gaga",
            "song2": "It's Tricky", "artist2": "Run-D.M.C.",
            "song3": "Time Warp", "artist3": "The Rocky Horror Picture Show",
            "song4": "Singing In The Rain / Umbrella (Glee Cast Version featuring Gwyneth Paltrow)", "artist4": "Glee Cast",
            "song5": "Singing In The Rain", "artist5": "Glee Cast"
        },
        {
            "pid": "608829",
            "playlist_title": "FINESSE",
            "song1": "Lose You", "artist1": "Sam Smith",
            "song2": "Tunnel Vision", "artist2": "Kodak Black",
            "song3": "Too Many Years", "artist3": "Unknown",
            "song4": "Selfish", "artist4": "Madison Beer",
            "song5": "Look At Me!", "artist5": "XXXTentacion"
        },
        {
            "pid": "273344",
            "playlist_title": "Oldies",
            "song1": "September", "artist1": "Earth Wind & Fire",
            "song2": "After the Love Has Gone", "artist2": "Earth Wind & Fire",
            "song3": "Sing a Song", "artist3": "Earth Wind & Fire",
            "song4": "That's the Way of the World", "artist4": "Earth Wind & Fire",
            "song5": "Reasons", "artist5": "Earth Wind & Fire"
        },
        {
            "pid": "501054",
            "playlist_title": "Rock",
            "song1": "The Sky Is A Neighborhood", "artist1": "Foo Fighters",
            "song2": "DOA", "artist2": "The Offspring",
            "song3": "Californication", "artist3": "Red Hot Chili Peppers",
            "song4": "Dani California", "artist4": "Red Hot Chili Peppers",
            "song5": "Wake Me Up When September Ends", "artist5": "Green Day"
        },
        {
            "pid": "750528",
            "playlist_title": "sports",
            "song1": "See You Again (feat. Charlie Puth)", "artist1": "Wiz Khalifa",
            "song2": "Sail", "artist2": "AWOLNATION",
            "song3": "Whistle", "artist3": "Flo Rida",
            "song4": "Centuries", "artist4": "Fall Out Boy",
            "song5": "Hall of Fame", "artist5": "The Script"
        },
        {
            "pid": "684261",
            "playlist_title": "Christian",
            "song1": "Sing to the King", "artist1": "Billy Foote",
            "song2": "Heart Won't Stop", "artist2": "TobyMac",
            "song3": "Good Good Father", "artist3": "Chris Tomlin",
            "song4": "O Come to the Altar", "artist4": "Elevation Worship",
            "song5": "Grace Alone", "artist5": "Rend Collective"
        },
        {
            "pid": "44648",
            "playlist_title": "gaming",
            "song1": "Heartbeat", "artist1": "Kelly Clarkson",
            "song2": "All About That Bass", "artist2": "Meghan Trainor",
            "song3": "Kanye", "artist3": "Kanye West",
            "song4": "Girl", "artist4": "Unknown",
            "song5": "Collarbone", "artist5": "Unknown"
        },
        {
            "pid": "837665",
            "playlist_title": "classics",
            "song1": "Ms. Jackson", "artist1": "OutKast",
            "song2": "Hey Ya! - Radio Mix / Club Mix", "artist2": "OutKast",
            "song3": "Hey Ma", "artist3": "OutKast",
            "song4": "Replay", "artist4": "Iyaz",
            "song5": "Angel", "artist5": "Shaggy"
        },
        {
            "pid": "786219",
            "playlist_title": "Party",
            "song1": "Red Nose", "artist1": "Sage the Gemini",
            "song2": "Ain't No Fun (If the Homies Cant Have None) (feat. Nate Dogg", "artist2": "Snoop Dogg",
            "song3": "Uber Everywhere", "artist3": "MadeinTYO",
            "song4": "Best Friend", "artist4": "Saweetie",
            "song5": "With That (feat. Duke)", "artist5": "Unknown"
        },
        {
            "pid": "47214",
            "playlist_title": "workout",
            "song1": "That's My Bitch", "artist1": "Jay-Z and Kanye West",
            "song2": "Saturday", "artist2": "Unknown",
            "song3": "Schoolin' Life", "artist3": "Unknown",
            "song4": "I'm In It", "artist4": "Unknown",
            "song5": "Fight Night", "artist5": "Migos"
        },
        {
            "pid": "889395",
            "playlist_title": "work",
            "song1": "Cool for the Summer", "artist1": "Demi Lovato",
            "song2": "Trap Queen", "artist2": "Fetty Wap",
            "song3": "23", "artist3": "Unknown",
            "song4": "Work It", "artist4": "Missy Elliott",
            "song5": "Omen - Radio Edit", "artist5": "Disclosure"
        },
        {
            "pid": "497427",
            "playlist_title": "Love songs",
            "song1": "Thinking out Loud", "artist1": "Ed Sheeran",
            "song2": "Keep Your Head Up", "artist2": "Andy Grammer",
            "song3": "Terrible Things", "artist3": "Unknown",
            "song4": "Miserable At Best", "artist4": "Mayday Parade",
            "song5": "Molly (feat. Brendon Urie of Panic at the Disco)", "artist5": "Unknown"
        },
        {
            "pid": "677006",
            "playlist_title": "Summer",
            "song1": "T-Shirt", "artist1": "Shontelle",
            "song2": "HUMBLE.", "artist2": "Kendrick Lamar",
            "song3": "1-800-273-8255", "artist3": "Logic",
            "song4": "Despacito - Remix", "artist4": "Luis Fonsi",
            "song5": "Strip That Down", "artist5": "Liam Payne"
        }
    ]

    generated_data = {}

    for pl in playlists:
        print(f"\nGeneration for the playlist '{pl['playlist_title']}' (PID {pl['pid']}) in five-shot...")
        generated_playlist = generate_playlist_five_shot(
            pl["playlist_title"],
            pl["song1"], pl["artist1"],
            pl["song2"], pl["artist2"],
            pl["song3"], pl["artist3"],
            pl["song4"], pl["artist4"],
            pl["song5"], pl["artist5"],
            llm_model,
            template_path,
            api_key,
            verbose,
            llm_model
        )
        generated_data[pl["pid"]] = {
            "playlist_title": pl["playlist_title"],
            "song1": pl["song1"],
            "artist1": pl["artist1"],
            "song2": pl["song2"],
            "artist2": pl["artist2"],
            "song3": pl["song3"],
            "artist3": pl["artist3"],
            "song4": pl["song4"],
            "artist4": pl["artist4"],
            "song5": pl["song5"],
            "artist5": pl["artist5"],
            "generated_playlist": generated_playlist
        }

    output_dir = "Json_file"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = "Five_shot_22_song.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=2)

    print(f"\nThe JSON file was generated in : {output_path}")

if __name__ == "__main__":
    main()
