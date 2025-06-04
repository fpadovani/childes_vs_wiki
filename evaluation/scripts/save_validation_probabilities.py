import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from minicons import scorer
import pandas as pd
from utils_1 import load_sentences_from_txt
from sympy import plot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from utils.variables import *

for dir_path in SCORE_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)
        
MODELS = {
    'en': {
        'childes_best_clm': [f"{TRAINED_MODELS}/en_child_clm_{seed}_new/checkpoint-48000" for seed in [42, 30, 13]],
        'wikipedia_best_clm': [f"{TRAINED_MODELS}/en_wiki_clm_{seed}_new/checkpoint-64000" for seed in [42, 30, 13]],
        'childes_convergence_clm': [f"{TRAINED_MODELS}/en_child_clm_{seed}_new/checkpoint-100000" for seed in [42, 30, 13]],
        'wikipedia_convergence_clm': [f"{TRAINED_MODELS}/en_wiki_clm_{seed}_new/checkpoint-100000" for seed in [42, 30, 13]],
        'childes_best_mlm': [f"{TRAINED_MODELS}/en_child_mlm_{seed}_new/checkpoint-100000" for seed in [42, 30, 13]],
        'wikipedia_best_mlm': [f"{TRAINED_MODELS}/en_wiki_mlm_{seed}_new/checkpoint-100000" for seed in [42, 30, 13]],
        'evalset': {
            'clams': 'en_evalset_ok',
            'zorro': 'zorro',
            'blimp': 'blimp'},
    },
    'de': {
        'childes_best_clm': [f"{TRAINED_MODELS}/de_child_clm_{seed}_new/checkpoint-48000" for seed in [42, 30, 13]],
        'wikipedia_best_clm': [f"{TRAINED_MODELS}/de_wiki_clm_{seed}_new/checkpoint-64000" for seed in [42, 30, 13]],
        'childes_convergence_clm': [f"{TRAINED_MODELS}/de_child_clm_{seed}_new/checkpoint-100000" for seed in [42, 30, 13]],
        'wikipedia_convergence_clm': [f"{TRAINED_MODELS}/de_wiki_clm_{seed}_new/checkpoint-100000" for seed in [42, 30, 13]],
        'childes_best_mlm': [f"{TRAINED_MODELS}/de_child_mlm_{seed}_new/checkpoint-100000" for seed in [42, 30, 13]],
        'wikipedia_best_mlm': [f"{TRAINED_MODELS}/de_wiki_mlm_{seed}_new/checkpoint-100000" for seed in [42, 30, 13]],
        'evalset': {'clams':'de_evalset_ok'},
    },
    
    'fr': {
        'childes_best_clm': [f"{TRAINED_MODELS}/fr_child_clm_{seed}_new/checkpoint-36000" for seed in [42, 30, 13]],
        'wikipedia_best_clm': [f"{TRAINED_MODELS}/fr_wiki_clm_{seed}_new/checkpoint-44000" for seed in [42, 30, 13]],
        'childes_convergence_clm': [f"{TRAINED_MODELS}/fr_child_clm_{seed}_new/checkpoint-100000" for seed in [42, 30, 13]],
        'wikipedia_convergence_clm': [f"{TRAINED_MODELS}/fr_wiki_clm_{seed}_new/checkpoint-100000" for seed in [42, 30, 13]],
        'wikipedia_best_mlm': [f"{TRAINED_MODELS}/fr_wiki_mlm_{seed}_new/checkpoint-100000" for seed in [42,30,13]],   
        'childes_best_mlm': [f"{TRAINED_MODELS}/fr_child_mlm_{seed}_new/checkpoint-100000" for seed in [42, 30, 13]],
        'evalset': {'clams':'fr_evalset_ok'}

    }
    }


    
# MAIN scoring function: updated to work for all evalsets
def score_and_save_models_for_language(lang, lang_config):
    print(f"\nScoring language: {lang}")

    model_configs = {k: v for k, v in lang_config.items() if k != 'evalset'}

    for evalset_key, evalset_subpath in lang_config["evalset"].items():
        print(f"\n> Dataset: {evalset_key.upper()}")
        if evalset_key == "clams":
            evalset_path = CLAMS_DIR + '/' + evalset_subpath
        elif evalset_key == "blimp":
            evalset_path = BLIMP_DIR
        elif evalset_key == "zorro":
            evalset_path = ZORRO_DIR
        else:
            raise ValueError(f"Unknown evalset key: {evalset_key}")

        out_lang_dir = os.path.join(SCORE_DIRS[evalset_key], lang)
        os.makedirs(out_lang_dir, exist_ok=True)

        for paradigm in os.listdir(evalset_path):
            if ".DS_Store" in paradigm:
                continue
            file_path = os.path.join(evalset_path, paradigm)
            sentences = load_sentences_from_txt(file_path)
            df = pd.DataFrame(sentences, columns=["sentence"])

            for model_type, model_paths in model_configs.items():
                for path in model_paths:
                    seed = path.split("_")[-2]
                    is_clm = "clm" in path
                    model_id = f"{lang}_{model_type}_{seed}"

                    print(f"Scoring: {model_id} on {paradigm}")

                    # Load model once
                    if is_clm:
                        model = scorer.IncrementalLMScorer(path, 'cpu')
                    else:
                        model = scorer.MaskedLMScorer(path, 'cpu')

                    scores = []
                    for sent in sentences:
                        if is_clm:
                            score = model.sequence_score([sent], reduction=lambda x: x.sum(0).item(), bow_correction=True)[0]
                        else:
                            score = model.sequence_score([sent], reduction=lambda x: x.sum(0).item(), PLL_metric='within_word_l2r')[0]
                        scores.append(score)

                    df[model_id] = scores

            out_file = os.path.join(out_lang_dir, f"{paradigm.replace('.txt','')}.csv")
            df.to_csv(out_file, index=False)
            print(f"Saved: {out_file}")



# Run for all languages
for lang, config in MODELS.items():
    score_and_save_models_for_language(lang, config)
