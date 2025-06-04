import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_fc import *
import nltk
from nltk.corpus import wordnet as wn

nltk.download('wordnet')
nltk.download('omw-1.4')  # For multilingual support


def process_and_save_distributions(freq_dicts, output_lang):
    """Sort, bin, and save frequency distributions, and generate corresponding plots."""
    output_plots_dir = os.path.join(BASE_DIR, "fitclams_generation", "distribution_plots", output_lang)
    os.makedirs(output_plots_dir, exist_ok=True)

    hist_labels = {
        'childes_noun': 'childes_nouns',
        'childes_verb': 'childes_verbs',
        'wiki_noun': 'wiki_nouns',
        'wiki_verb': 'wiki_verbs'
    }

    # Sort, bin, and filter frequencies
    binned_data = {}
    # Filter out words that don't have both singular and plural frequency counts
    for label, freq in freq_dicts.items():
        sorted_freq = sort_by_total_freq(freq)
        filtered = [(w, f) for w, f in sorted_freq if f["singular"][1] > 0 and f["plural"][1] > 0]

        # Assign frequency bins using logarithmic scale
        binned_data[label] = assign_bins_log(filtered, language=output_lang)

    # Plotting
    for label, df in binned_data.items():
        color = 'skyblue' if 'noun' in label else 'lightcoral'
        plot_bin_histogram_freq(df, data_type='nouns' if 'noun' in label else 'verbs',
                                save_path=os.path.join(output_plots_dir, f"{hist_labels[label]}_histogram.png"))

    # Saving to bins
    for label, df in binned_data.items():
        is_noun = 'noun' in label
        source = 'childes' if 'childes' in label else 'wiki'
        noun_or_verb = 'nouns' if is_noun else 'verbs'
        folder = f"{noun_or_verb}_per_bin_{source}"
        save_dir = os.path.join(EXTRACTED_WORDS, output_lang, folder)
        save_all_bins(df, save_dir, language=output_lang, is_noun=is_noun)
        print(f"Sampled words saved to {os.path.join(save_dir, 'sampled_words.csv')}")


def compute_frequencies(tokens, lang, nlp, childes_freq, wiki_freq):
    """Identify and assign noun/verb frequencies for each token."""
    noun_freqs = {
        'childes': create_frequency_dict(),
        'wiki': create_frequency_dict()
    }
    verb_freqs = {
        'childes': create_frequency_dict(),
        'wiki': create_frequency_dict()
    }

    for token in tokens:
        if not token or not isinstance(token, str):
            continue

        token = token.strip()
        # For German, capitalize and re-evaluate part-of-speech
        if lang == 'de':
            token_cap = token.capitalize()
            doc_cap = nlp(token_cap)
            if doc_cap and doc_cap[0].pos_ == 'NOUN':
                process_noun(token_cap, doc_cap[0].lemma_.lower(), doc_cap, lang, childes_freq, wiki_freq,
                             noun_freqs['childes'], noun_freqs['wiki'])

        doc = nlp(token.lower())
        if not doc:
            continue

        lemma = doc[0].lemma_.lower()
        pos = doc[0].pos_

        if pos == 'NOUN':
            process_noun(token, lemma, doc, lang, childes_freq, wiki_freq, noun_freqs['childes'], noun_freqs['wiki'])
        elif pos == 'VERB':
            process_verb(token, lemma, doc, lang, childes_freq, wiki_freq, verb_freqs['childes'], verb_freqs['wiki'])

    return noun_freqs['childes'], noun_freqs['wiki'], verb_freqs['childes'], verb_freqs['wiki']



def main():
    """Entry point for frequency analysis pipeline."""
    lang_settings = select_language()
    if not lang_settings:
        return

    # Unpack language settings
    _, lang_code, vocab_file, childes_file, wiki_file, _, _, nlp, childes_corpus, wiki_corpus = lang_settings

    # Load data and preprocess
    intersected_vocab, freq_childes, freq_wiki = load_data(vocab_file, childes_file, wiki_file, wiki_corpus, childes_corpus, nlp)

    # Compute noun and verb frequencies
    childes_noun, wiki_noun, childes_verb, wiki_verb = compute_frequencies(
        intersected_vocab, lang_code, nlp, freq_childes, freq_wiki
    )

    freq_dicts = {
        'childes_noun': childes_noun,
        'wiki_noun': wiki_noun,
        'childes_verb': childes_verb,
        'wiki_verb': wiki_verb
    }

    # Process and save plots and binned data
    process_and_save_distributions(freq_dicts, lang_code)


if __name__ == "__main__":
    main()