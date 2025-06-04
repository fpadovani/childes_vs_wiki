from collections import defaultdict
import pandas as pd
import os 
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.variables import *

# ------------------------------
# Loading and Tokenization Utils
# ------------------------------

def load_sentences_from_csv(file_path, column_name="sentences"):
    """Load sentences from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    if column_name not in df:
        raise ValueError(f"Column '{column_name}' not found in {file_path}")
    
    return df[column_name].dropna().astype(str).tolist()


def tokenize_sentences_spacy(sentences, nlp):
    """Tokenize list of sentences using SpaCy."""
    return [token.text for sentence in sentences for token in nlp(sentence)]


def get_intersection(tokens1, tokens2):
    """Return intersection of two token lists."""
    return set(tokens1).intersection(tokens2)


# ------------------------------
# Frequency Dictionary Utilities
# ------------------------------

def sort_by_total_freq(distribution):
    """Sort frequency distribution by total frequency descending."""
    return sorted(distribution.items(), key=lambda x: x[1]['total'], reverse=True)

def create_frequency_dict():
    """Default frequency dictionary structure for nouns/verbs."""
    return defaultdict(lambda: {"singular": ("", 0), "plural": ("", 0), "total": 0})


def remove_duplicates(distribution):
    """Keep the entry with the highest frequency for each word."""
    seen = {}
    for word, freq_data in distribution:
        if word not in seen or seen[word]["total"] < freq_data["total"]:
            seen[word] = freq_data
    return list(seen.items())



# ------------------------------
# Bin Management
# ------------------------------

def assign_bins_log(distribution, language, num_bins=10):
    """Assign words to logarithmic frequency bins."""
    if not distribution:
        return {}
    
    freq_values = [freq['total'] for _, freq in distribution]
    log_min, log_max = np.log10(min(freq_values)), np.log10(max(freq_values))
    bin_edges = np.logspace(log_min, log_max, num_bins + 1)

    binned_data = {}
    for word, freq in distribution:
        total_freq = freq['total']
        bin_index = np.clip(np.digitize(total_freq, bin_edges, right=True) - 1, 0, num_bins - 1)
        singular, plural = freq['singular'][0], freq['plural'][0]
        gender = freq.get('gender')

        key = (singular, plural, gender) if language in ["fr", "de"] and gender else (singular, plural)
        binned_data[key] = (bin_index, total_freq)
    
    return binned_data



def get_words_in_bin(binned_data, bin_index):
    """Return words and frequency data in the specified bin."""
    return [(word[0], word[1], freq) for word, (bin_idx, freq) in binned_data.items() if bin_idx == bin_index]



def save_bin_to_csv(binned_data, bin_index, output_path, language, is_noun):
    """Save words in a specific bin to a CSV file."""
    words = []
    for word_info, (bin_idx, freq) in binned_data.items():
        if bin_idx != bin_index:
            continue
        if language in ["fr", "de"] and is_noun:
            words.append((*word_info, freq))  # (singular, plural, gender, freq)
        else:
            singular, plural = word_info
            words.append((singular, plural, freq))

    headers = ['singular', 'plural', 'gender', 'total_frequency'] if language in ["fr", "de"] and is_noun else ['singular', 'plural', 'total_frequency']
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(words)
        

def save_all_bins(binned_data, output_dir, language, is_noun, prefix="bin_"):
    """Save each bin's contents to a separate CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    max_bin = max(bin_idx for _, (bin_idx, _) in binned_data.items())
    for bin_index in range(max_bin + 1):
        filename = os.path.join(output_dir, f"{prefix}{bin_index:02}.csv")
        save_bin_to_csv(binned_data, bin_index, filename, language, is_noun)



# ------------------------------
# Frequency Processing Helpers
# ------------------------------


def get_token_frequency(df, token):
    """Retrieve token frequency from DataFrame."""
    row = df[df['word'] == token]
    return row['count'].iloc[0] if not row.empty else 0


def process_noun(token, lemma, doc, lang, childes_df, wiki_df, childes_freq, wiki_freq):
    """Process and store noun frequency info."""
    morph = doc[0].morph

    # Check grammatical number
    is_plural = morph.get("Number") == ["Plur"]
    is_singular = morph.get("Number") == ["Sing"]

    # Extract case and gender if available (useful for German and French)
    case = morph.get("Case")
    gender = morph.get("Gender")[0] if lang in ('fr', 'de') and morph.get("Gender") else None

    # For German, only include nominative nouns
    if lang == 'de' and case != ["Nom"]:
        return

    # Preprocessing: lowercase for German, remove possessive in English
    token = token.lower() if lang == 'de' else token.replace("'s", "") if lang == 'eng' else token

    # Loop over both datasets (CHILDES and Wiki)
    for df, freq_dict in [(childes_df, childes_freq), (wiki_df, wiki_freq)]:

        # Store frequency based on number (plural or singular)
        freq = get_token_frequency(df, token)
        if freq > 0:
            freq_dict[lemma]["plural" if is_plural else "singular"] = (token, freq)
            freq_dict[lemma]["total"] += freq
            if gender:
                freq_dict[lemma]["gender"] = gender


def process_verb(token, lemma, doc, lang, unigram_freq_childes, unigram_freq_wiki, childes_verb_freq, wiki_verb_freq):

    """Process verb frequencies based on language-specific rules."""
    
    # Extract POS and grammatical features
    pos_tag, tag = doc[0].pos_, doc[0].tag_
    is_plural = doc[0].morph.get("Number") == ["Plur"]
    is_singular = doc[0].morph.get("Number") == ["Sing"]
    is_present = doc[0].morph.get("Tense") == ['Pres']
    mood_subj = doc[0].morph.get("Mood") == ['Sub']
    mood_cond = doc[0].morph.get("Mood") == ['Cnd']
    is_third_person = doc[0].morph.get("Person") == ['3']
    is_infinitive = doc[0].morph.get("VerbForm") == ["Inf"]
    
    # Flag to control whether this token should be processed
    process = False

    # For English: only certain present-tense verb tags
    if lang == 'eng' and pos_tag == 'VERB' and tag in {'VB', "VBP", "VBZ"}:
        process = True

    # For French and German: present, third-person, not subjunctive or conditional
    elif lang == 'fr' or lang == 'de':
        if pos_tag == 'VERB' and is_third_person and is_present and not mood_subj and not mood_cond:
            process = True


    if process:
        for dataset, freq_dict in [(unigram_freq_childes, childes_verb_freq), (unigram_freq_wiki, wiki_verb_freq)]:
            freq = get_token_frequency(dataset, token.lower())

            # Store frequency based on number
            if freq > 0:
                if is_singular:
                    freq_dict[lemma]["singular"] = (token.lower(), freq)
                    freq_dict[lemma]["total"] += freq
                else:
                    freq_dict[lemma]["plural"] = (token.lower(), freq)
                    freq_dict[lemma]["total"] += freq



# ------------------------------
# Language + Data Setup
# ------------------------------

def select_language():
    """Let user select a language and return related variables."""
    options = {
        "1": ("English", "eng", INTERSECTED_VOCAB_EN, UNIGRAM_CHILDES_ENG, UNIGRAM_WIKI_ENG, UNIGRAM_DEP_CHILDES_ENG, UNIGRAM_DEP_WIKI_ENG, nlp_eng, TRAINING_CHILDES_ENG, TRAINING_WIKI_ENG),
        "2": ("French", "fr", INTERSECTED_VOCAB_FR, UNIGRAM_CHILDES_FR, UNIGRAM_WIKI_FR, UNIGRAM_DEP_CHILDES_FR, UNIGRAM_DEP_WIKI_FR, nlp_fr, TRAINING_CHILDES_FR, TRAINING_WIKI_FR),
        "3": ("German", "de", INTERSECTED_VOCAB_DE, UNIGRAM_CHILDES_DE, UNIGRAM_WIKI_DE, UNIGRAM_DEP_CHILDES_DE, UNIGRAM_DEP_WIKI_DE, nlp_de, TRAINING_CHILDES_DE, TRAINING_WIKI_DE),
    }
    choice = input("Choose a language: \n1. English\n2. French\n3. German\nEnter 1, 2, or 3: ").strip()
    return options.get(choice, (None,))


def load_or_compute_intersection(wiki_df, childes_df, intersected_file, nlp):
    """Load or compute intersected vocabulary."""
    if os.path.exists(intersected_file):
        with open(intersected_file, 'r') as f:
            return set(f.read().splitlines())

    intersected = get_intersection(wiki_df['word'].tolist(), childes_df['word'].tolist())

    with open(intersected_file, 'w') as f:
        f.write("\n".join(intersected))
    
    return intersected


def load_data(intersected_file, childes_path, wiki_path, wiki_corpus, childes_corpus, nlp):
    """Load frequency data and intersected vocabulary."""
    childes_df = pd.read_csv(childes_path)
    wiki_df = pd.read_csv(wiki_path)

    intersected_vocab = load_or_compute_intersection(wiki_df, childes_df, intersected_file, nlp)
    print(f"Loaded {len(intersected_vocab)} intersected tokens.")
    
    return intersected_vocab, childes_df, wiki_df




#--------------------------------------------------------------------
## HELPERS FOR THE MAIN FUNCTION
#--------------------------------------------------------------------

def select_language():
    """Prompt the user to choose a language and return corresponding settings."""
    languages = {
        "1": ("English", "eng", INTERSECTED_VOCAB_EN, UNIGRAM_CHILDES_ENG, UNIGRAM_WIKI_ENG,
              UNIGRAM_DEP_CHILDES_ENG, UNIGRAM_DEP_WIKI_ENG, nlp_eng, TRAINING_CHILDES_ENG, TRAINING_WIKI_ENG),
        "2": ("French", "fr", INTERSECTED_VOCAB_FR, UNIGRAM_CHILDES_FR, UNIGRAM_WIKI_FR,
              UNIGRAM_DEP_CHILDES_FR, UNIGRAM_DEP_WIKI_FR, nlp_fr, TRAINING_CHILDES_FR, TRAINING_WIKI_FR),
        "3": ("German", "de", INTERSECTED_VOCAB_DE, UNIGRAM_CHILDES_DE, UNIGRAM_WIKI_DE,
              UNIGRAM_DEP_CHILDES_DE, UNIGRAM_DEP_WIKI_DE, nlp_de, TRAINING_CHILDES_DE, TRAINING_WIKI_DE)
    }

    choice = input("Choose a language: \n1. English\n2. French\n3. German\nEnter 1, 2, or 3: ").strip()
    if choice in languages:
        print(f"You selected {languages[choice][0]}.")
        return languages[choice]
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
        return None
    


def load_or_compute_intersection(wiki_file, childes_file,intersected_vocab_file, nlp):
    """Load intersected vocabulary from file or compute if not available."""
    if os.path.exists(intersected_vocab_file):
        with open(intersected_vocab_file, 'r') as f:
            return set(f.read().splitlines())

    childes_tokens = childes_file['word'].tolist()
    wiki_tokens = wiki_file['word'].tolist()
    intersected_vocab = get_intersection(childes_tokens, wiki_tokens)

    # Save for future use
    with open(intersected_vocab_file, 'w') as f:
        for word in intersected_vocab:
            f.write(f"{word}\n")

    return intersected_vocab


def load_data(intersected_vocabulary_file, childes_file, wiki_file, wiki_corpus, childes_corpus, nlp):
    """Load intersected vocabulary and frequency data."""
    unigram_childes = pd.read_csv(childes_file)
    unigram_wiki = pd.read_csv(wiki_file)

    intersected_vocabulary = load_or_compute_intersection(
        unigram_wiki, unigram_childes,
        intersected_vocab_file=intersected_vocabulary_file, nlp=nlp
    )
    print(f"Number of intersected tokens: {len(intersected_vocabulary)}")

    

    return intersected_vocabulary, unigram_childes, unigram_wiki



# ------------------------------
# Plotting Utility
# ------------------------------

def plot_bin_histogram_freq(binned_data, data_type="words", save_path=None):
    """Plot histogram of number of words in each frequency bin."""
    bin_counts = defaultdict(int)
    bin_ranges = {}

    for _, (bin_index, freq) in binned_data.items():
        bin_counts[bin_index] += 1
        bin_ranges.setdefault(bin_index, [freq, freq])
        bin_ranges[bin_index][0] = min(bin_ranges[bin_index][0], freq)
        bin_ranges[bin_index][1] = max(bin_ranges[bin_index][1], freq)

    sorted_bins = sorted(bin_counts.keys())
    counts = [bin_counts[b] for b in sorted_bins]
    labels = [f"[{bin_ranges[b][0]} - {bin_ranges[b][1]}]" for b in sorted_bins]

    cmap = cm.Blues if data_type == "nouns" else cm.Reds if data_type == "verbs" else cm.Purples
    colors = [cmap(plt.Normalize(min(counts), max(counts))(c)) for c in counts]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(sorted_bins)), counts, color=colors, edgecolor='black')
    plt.xticks(range(len(sorted_bins)), labels, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Frequency Range per Bin", fontsize=14)
    plt.ylabel("Total Words", fontsize=14)
    plt.title(f"Distribution of {data_type.capitalize()}", fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha='center', va='bottom')

    if data_type == "nouns":
        plt.ylim(0, 250)
    elif data_type == "verbs":
        plt.ylim(0, 40)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Histogram saved to {save_path}")
    
    plt.show()