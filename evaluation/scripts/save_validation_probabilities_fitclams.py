from curses.ascii import isdigit
import os
import sys
import numpy as np
import pandas as pd
from minicons import scorer
from transformers import AutoTokenizer
from utils_1 import load_sentences_from_csv
# Add project root for relative imports if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from utils.variables import *

# Ensure output dirs exist
for lang in ['en', 'fr', 'de']: 
    for subfolder in ['childes', 'wiki']:
        os.makedirs(os.path.join(SCORE_DIR_FITCLAMS, lang, subfolder), exist_ok=True)

MODELS = {
    'en': {
        'childes_best_clm': [f"{TRAINED_MODELS}/en_child_clm_{seed}_new/checkpoint-48000" for seed in [42, 30, 13]],
        'wikipedia_best_clm': [f"{TRAINED_MODELS}/en_wiki_clm_{seed}_new/checkpoint-64000" for seed in [42, 30, 13]],
        'new_clams': ['en_/childes', 'en_/wiki']
    },
    'fr': {
        'childes_best_clm': [f"{TRAINED_MODELS}/fr_child_clm_{seed}_new/checkpoint-36000" for seed in [42, 30, 13]],
        'wikipedia_best_clm': [f"{TRAINED_MODELS}/fr_wiki_clm_{seed}_new/checkpoint-44000" for seed in [42, 30, 13]],
        'new_clams': ['fr_/childes', 'fr_/wiki']
    },
    'de': {
        'childes_best_clm': [f"{TRAINED_MODELS}/de_child_clm_{seed}_new/checkpoint-48000" for seed in [42, 30, 13]],
        'wikipedia_best_clm': [f"{TRAINED_MODELS}/de_wiki_clm_{seed}_new/checkpoint-64000" for seed in [42, 30, 13]],
        'new_clams': ['de_/childes', 'de_/wiki']
    }

}


def score_new_clams(lang, config):
    print(f"\n🔍 Scoring FIT-CLAMS for: {lang.upper()}")

    for eval_path in config['new_clams']:
        subset = 'childes' if 'childes' in eval_path else 'wiki'
        full_path = os.path.join(FITCLAMS_DIR, eval_path)
        out_dir = os.path.join(SCORE_DIR_FITCLAMS, lang, subset)
        os.makedirs(out_dir, exist_ok=True)

        paradigms = sorted([f for f in os.listdir(full_path) if f.endswith(".csv") and 'scored' not in f])
        print(f"\n> Subset: {subset.upper()} - {len(paradigms)} paradigms")

        for paradigm in paradigms:
            file_path = os.path.join(full_path, paradigm)
            sentences, df = load_sentences_from_csv(file_path)

            # Collect model configs together for unified processing
            model_configs = {
                'childes_best_clm': config['childes_best_clm'],
                'wikipedia_best_clm': config['wikipedia_best_clm']
            }

            for model_type, model_paths in model_configs.items():
                for path in model_paths:
                    elements = path.split("_")[-4:-1]
                    seed = [ele for ele in elements if ele.isdigit()][0]
                    model_id = f"{lang}_{model_type}_{seed}"

                    print(f"Scoring: {model_id} on {paradigm}")

                    model = scorer.IncrementalLMScorer(path, 'cpu')
                    tokenizer = AutoTokenizer.from_pretrained(path)

                    # Score sentence by sentence
                    scores = []
                    for sent in sentences:

                        if 'long_vp_coord' in paradigm:
                            if lang == 'en':
                                try:
                                    tokens = sent.split()
                                    verb_token = tokens.index("and") + 1

                                    and_idx = sent.index("and")
                                    verb_idx = and_idx + 4
                                    verb = tokens[verb_token]
                
                                except ValueError:
                                    print(f"Warning: 'and' not found in sentence: {sent}")
                                    scores.append(None)
                                    continue
                            elif lang == 'fr':
                                try:
                                    tokens = sent.split()
                                    verb_token = tokens.index("et") + 1

                                    and_idx = sent.index("et")
                                    verb_idx = and_idx + 3
                                    verb = tokens[verb_token]
                
                                except ValueError:
                                    print(f"Warning: 'et' not found in sentence: {sent}")
                                    scores.append(None)
                                    continue
                            
                            elif lang == 'de':
                                try:
                                    tokens = sent.split()
                                    verb_token = tokens.index("und") + 1

                                    and_idx = sent.index("und")
                                    verb_idx = and_idx + 4
                                    verb = tokens[verb_token]
                
                                except ValueError:
                                    print(f"Warning: 'und' not found in sentence: {sent}")
                                    scores.append(None)
                                    continue

                        elif 'obj_rel_within_anim' in paradigm:
                            if lang == 'en' or lang == 'fr':
                                try:
                                    tokens = sent.split()
                                    verb = tokens[-2]
                                    verb_idx = sent.index(verb)
                                
                                except ValueError:
                                    print(f"Warning: 'and' not found in sentence: {sent}")
                                    scores.append(None)
                                    continue
                            elif lang == 'de':
                                try:
                                    tokens = sent.split()
                                    verb = tokens[-3]
                                    verb_idx = sent.index(verb)
                                
                                except ValueError:
                                    print(f"Warning: 'und' not found in sentence: {sent}")
                                    scores.append(None)
                                    continue

                        else:
                            tokens = sent.split()
                            verb = tokens[-1]
                            verb_idx = sent.index(verb)
                            
                        context = sent[:verb_idx].strip()
                        context_len = len(tokenizer.tokenize(context))
                        verb_len = len(tokenizer.tokenize(" " + verb))
                        verb_span = slice(context_len, context_len + verb_len)

                        tokens_prob = model.token_score(sent)
                        tokens_prob = tokens_prob[0][verb_span]
                            
                        score = sum([x[1] for x in tokens_prob])
                        scores.append(score)

                    # Add to DataFrame
                    df[model_id] = scores

            out_file = os.path.join(out_dir, paradigm)
            df.to_csv(out_file, index=False)
            print(f"Saved: {out_file}")

# Run for all languages
for lang, config in MODELS.items():
    score_new_clams(lang, config)