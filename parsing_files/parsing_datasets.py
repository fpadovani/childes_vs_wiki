import spacy
import pandas as pd
import pickle
from typing import *
from tqdm import tqdm, trange
from conllu import parse
from spacy_conll import ConllFormatter
from supar import Parser
from diaparser.parsers import Parser as DiaParser
from conllu.exceptions import ParseException
import os
import math
import re
import logging
from nltk import map_tag
from spacy_conll import init_parser
from spacy_conll.parser import ConllParser
from spacy.displacy import render
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.variables import PARSED_DATASETS,DATASET_DIR


# Language configurations
LANG_CONFIGS = {
    "en": {
        "spacy_model": "en_core_web_trf",
        "supar_model": "biaffine-dep-en",
        "childes_file": "english/random/train.csv",
        "wiki_file": "english/wikipedia/train.csv",
        "pickle_dir": f'{PARSED_DATASETS}/training_datasets/pickle/eng_deps'
    },
    "fr": {
        "spacy_model": "fr_dep_news_trf",
        "supar_model": None,
        "childes_file": "french/random/train.csv",
        "wiki_file": "french/wikipedia/train.csv",
        "pickle_dir": f'{PARSED_DATASETS}/training_datasets/pickle/fr_deps'
    },
    "de": {
        "spacy_model": None,
        "supar_model": "de_hdt.dbmdz-bert-base",
        "childes_file": "german/random/train.csv",
        "wiki_file": "german/wikipedia/train.csv",
        "pickle_dir": f'{PARSED_DATASETS}/training_datasets/pickle/de_deps'
    }
}


for config in LANG_CONFIGS.values():
    os.makedirs(config["pickle_dir"], exist_ok=True)


def conllu2sen(tokenlist) -> str:
    tokens = [x[i]['form'] for x in tokenlist for i in range(len(x))]
    
    return ' '.join(tokens)



conll_parser = ConllParser(init_parser("en_core_web_trf", "spacy"))

def conll_doc_to_tree(doc):
    tree = doc.to_tree().serialize().strip()
    spacy_doc = conll_parser.parse_conll_text_as_spacy(tree)

    options = {"collapse_punct": False}
    render(spacy_doc, style='dep', options=options)


def rewrite_dep(match):
    pos = match.group(1)
    ud = map_tag('en-ptb', 'universal', pos)

    if ud in '.,':
        ud = 'PUNCT'
    elif ud == 'PRT':
        ud = 'PART'

    return f"\t_\t{ud}\t{pos}\t"


#the following functions are used to parse the wikipedia and childes datasets using supar only and save it into conllu files
##FUNCTION TO CONVERT A PARSED SENTENCE INTO CONLLU FORMAT
def convert_to_conllu(parsed, sentence):
    """
    Converts a parsed sentence (from parser.predict) into CoNLL-U format.
    """
    conllu_lines = []
    sentence = preprocess_raw_dep(str(parsed))
    tokens = [line.split("\t")[1] for line in sentence.strip().split("\n")]


    for i, (token, arc, rel) in enumerate(zip(tokens, parsed.arcs, parsed.rels)):
        # ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, (EXTRA UNUSED)
        conllu_line = [
            str(i + 1),       # ID (1-based index)
            token,            # FORM (word)
            "_",              # LEMMA
            "_",              # UPOS
            "_",              # XPOS
            "_",              # FEATS
            str(arc),         # HEAD (dependency head)
            rel,              # DEPREL (relation)
            "_"               # MISC (ignored)
        ]
        conllu_lines.append("\t".join(conllu_line))

    return "\n".join(conllu_lines)



def preprocess_raw_dep(raw_dep):
    return re.sub(r'\t_\t_\t([A-Z,.$]+)\t', rewrite_dep, raw_dep)  


def get_batch(data: list, batch_size: int, max_sen_len=200):
    sindex = 0
    eindex = batch_size

    while eindex < len(data):
        batch = data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp

        batch = [
            item if (len(item.split()) < max_sen_len) else None
            for item in batch
        ]

        yield batch

    if eindex >= len(data):
        batch = data[sindex:]
        
        batch = [
            item if (len(item.split()) < max_sen_len) else None
            for item in batch
        ]
        
        yield batch

def fix_spacy_contractions(doc_lines: List[str], supar_tokens: List[str]) -> List[str]:
    """
    Re-align spaCy CoNLL lines to match SuPar token count by merging contractions.
    """
    fixed_lines = []

    i = 0
    while i < len(doc_lines):
        line = doc_lines[i]
        parts = line.split("\t")
        if len(parts) <= 1:
            i += 1
            continue
        token = parts[1]

        # Example contraction logic
        if i + 1 < len(doc_lines):
            next_parts = doc_lines[i + 1].split("\t")
            if len(next_parts) > 1:
                next_token = next_parts[1]
                if token == "let" and next_token == "'s":
                    combined = "let's"
                    # Do something with combined...
                    i += 2
                    continue

        # Default handling
        # Do something with `token`
        i += 1

    return fixed_lines


def modify_conllu_string(dataset_tok, doc, arcs, rels):
    """Modify the CoNLL-U string with new arcs and relations."""
    conllu_lines = doc._.conll_str.strip().split("\n")
    modified_lines = []

    if len(conllu_lines) != len(arcs):
        conllu_lines = fix_spacy_contractions(conllu_lines,dataset_tok)

    for i, line in enumerate(conllu_lines):
        columns = line.split("\t")
        if len(columns) >= 8:
            columns[6] = str(arcs[i])   # HEAD
            columns[7] = rels[i]        # RELATION
        modified_lines.append("\t".join(columns))

    return "\n".join(modified_lines)


def parse_corpus(nlp, corpus, conllu_file, batch_size=64):
    """Parse corpus using Spacy and save in CoNLL-U format."""
    if not os.path.exists(conllu_file):
        os.makedirs(os.path.dirname(conllu_file), exist_ok=True)
    with open(conllu_file, 'w', encoding="utf-8") as f:
        for batch in tqdm(get_batch(corpus, batch_size), total=len(corpus) // batch_size):
            docs = list(nlp.pipe([item for item in batch if item is not None], disable=["ner", "textcat"], batch_size=batch_size))
            for doc in docs:
                f.write(doc._.conll_str + '\n\n')


##FUNCTION TO PARSE A CORPUS USING SUPAR AND SAVE IT AS A CONLLU FILE
def parse_corpus_supar_childes_wiki(parser, corpus: List[str], conllu_file: str, batch_size=64, empty_file=False):
    """
    Parses a corpus using only `parser.predict()` and saves it as a CoNLL-U file.
    """
    iterator = get_batch(corpus, batch_size)
    if not os.path.exists(conllu_file):
        os.makedirs(os.path.dirname(conllu_file), exist_ok=True)

    with open(conllu_file, 'a', encoding="utf-8") as f:
        if empty_file:
            f.truncate(0)  # Clear the file if empty_file=True

        for batch in tqdm(iterator, total=len(corpus) // batch_size):
            true_batch = [item for item in batch if item is not None]

            # Use only `parser.predict()` (no spaCy)
            dataset = list(parser.predict(true_batch, lang='en', prob=True, verbose=False))

            for idx, sentence in enumerate(true_batch):
                if sentence is None:
                    f.write('None\n\n')
                else:
                    parsed = dataset[idx]  # Parsed object
                    conllu = convert_to_conllu(parsed, sentence)  # Convert to CoNLL-U format
                    f.write(conllu + '\n\n')


def parse_corpus_supar(nlp, parser, corpus, conllu_file, batch_size=64):
    """Parse corpus using Spacy and Supar, replacing dependency structure."""
    if not os.path.exists(conllu_file):
        os.makedirs(os.path.dirname(conllu_file), exist_ok=True)
    with open(conllu_file, 'w', encoding="utf-8") as f:
        for batch in tqdm(get_batch(corpus, batch_size), total=len(corpus) // batch_size):
            docs = list(nlp.pipe([item for item in batch if item is not None], disable=["ner", "textcat"], batch_size=batch_size))
            dataset = list(parser.predict([doc.text for doc in docs], lang='en', prob=True, verbose=False))
            for i, doc in enumerate(docs):
                f.write(modify_conllu_string(doc, dataset[i].arcs, dataset[i].rels) + '\n\n')


##FUNCTION TO PARSE GERMAN CLAMS with diaparser
def parse_corpus_diaper_de(parser, corpus: List[str], conllu_file: str, batch_size=64, empty_file=False):
    """
    Parses a corpus using only `parser.predict()` and saves it as a CoNLL-U file.
    """
    iterator = get_batch(corpus, batch_size)

    with open(conllu_file, 'a', encoding="utf-8") as f:
        if empty_file:
            f.truncate(0)  # Clear the file if empty_file=True

        for batch in tqdm(iterator, total=len(corpus) // batch_size):
            true_batch = [item for item in batch if item is not None]
            tokens_batch = [[token for token in item.split()] for item in true_batch]
            

            # Use only `parser.predict()` (no spaCy)
            dataset = parser.predict(tokens_batch, prob=True)

            for idx, sentence in enumerate(true_batch):
                if sentence is None:
                    f.write('None\n\n')
                else:
                    parsed = dataset.sentences[idx]  # Parsed object
                    conllu = convert_to_conllu(parsed, sentence)  # Convert to CoNLL-U format
                    f.write(conllu + '\n\n')



def parse_corpus_diaper_de_safe(parser, corpus: List[str], conllu_file: str, batch_size=64, empty_file=False):
    """
    Parses a corpus using only `parser.predict()`, skipping problematic batches.
    """
    nlp = spacy.load("de_dep_news_trf")
    iterator = get_batch(corpus, batch_size)
    if not os.path.exists(conllu_file):
        os.makedirs(os.path.dirname(conllu_file), exist_ok=True)
    with open(conllu_file, 'a', encoding="utf-8") as f:
        if empty_file:
            f.truncate(0)  # Clear the file if empty_file=True

        for batch in tqdm(iterator, total=len(corpus) // batch_size):
            true_batch = [item for item in batch if item is not None]
            tokens_batch = [[token.text for token in nlp(item).doc] for item in true_batch]
            
            try:
                # Attempt to parse the batch
                dataset = parser.predict(tokens_batch, prob=True)

                for idx, sentence in enumerate(true_batch):
                    if sentence is None:
                        f.write('None\n\n')
                    else:
                        parsed = dataset.sentences[idx]  # Parsed object
                        conllu = convert_to_conllu(parsed, sentence)  # Convert to CoNLL-U format
                        f.write(conllu + '\n\n')

            except IndexError as e:
                logging.error(f"Skipping batch due to error: {e}")  # Log the issue
                print(f"Skipping batch due to error: {e}")  # Print to console for debugging
                continue  # Skip to the next batch


def save_pickle_data(parsed_data, pickle_dir, corpus_name, num_splits=1):
    """Save parsed dependencies as pickle files."""
    split_size = math.ceil(len(parsed_data) / num_splits)

    for i in trange(num_splits, desc="Saving pickle files"):
        split = parsed_data[split_size * i:split_size * (i + 1)]
        pickle_file = os.path.join(pickle_dir, corpus_name, f"conllu_deps_{i}.pickle")
        
        with open(pickle_file, "wb") as f:
            pickle.dump(split, f)


def load_pickle_data(pickle_dir, corpus_name):
    """Load parsed dependency data from pickle files."""
    pickle_files = [f for f in os.listdir(pickle_dir) if corpus_name in f and f.endswith(".pickle")]
    
    parsed_data = []
    for pfile in pickle_files:
        with open(os.path.join(pickle_dir, pfile), "rb") as f:
            parsed_data.extend(pickle.load(f))
    
    return parsed_data
    

def process_corpus(lang, corpus_type):
    """General function to process Wikipedia and CHILDES datasets for a given language."""
    config = LANG_CONFIGS[lang]
    corpus_file = os.path.join(DATASET_DIR, config[f"{corpus_type}_file"])
    conllu_file = os.path.join(PARSED_DATASETS,'training_datasets','conllu', f'corpus_{corpus_type}_{lang}_deps.conllu')
    

    # Load corpus
    corpus = pd.read_csv(corpus_file)['text'].dropna().tolist()

    
    # Load models
    if config["spacy_model"]:
        nlp = spacy.load(config["spacy_model"])
        nlp.add_pipe("conll_formatter", last=True)
    
    if config["supar_model"] and lang != "de":
        parser = Parser.load(config["supar_model"])
    
    elif lang == "de":
        parser = DiaParser.load(config["supar_model"])
    
    # Parse corpus
    if lang == "de":
        parse_corpus_diaper_de_safe(parser, corpus, conllu_file)
    elif lang == 'en':
        parse_corpus_supar(nlp, parser, corpus, conllu_file)
    else:
        parse_corpus(nlp, corpus, conllu_file)
    
    

    with open(conllu_file, encoding="utf-8") as f:
        raw_deps = f.read().strip().split('\n\n')

    new_docs = []

    for raw_dep in tqdm(raw_deps):
        if raw_dep == 'None':
            continue
        try:
            clean_raw_dep = preprocess_raw_dep(raw_dep)
            doc = parse(clean_raw_dep)[0]
            new_docs.append(doc)
        except ParseException as e:
            print(f"Skipping malformed sentence due to ParseException: {e}")
    
    # Store pickled dependencies to file, split into multiple files to reduce RAM overhead when loading it in
    num_splits = 20
    split_size = math.ceil(len(new_docs) / num_splits)


    for i in trange(num_splits):
        split = new_docs[split_size*i:split_size*(i+1)]

        dir = config['pickle_dir']
        if not os.path.exists(f'{dir}/{corpus_type}'):
            os.makedirs(f'{dir}/{corpus_type}')
        with open(f'{dir}/{corpus_type}/conllu_deps_{i}.pickle', 'wb') as f:
            pickle.dump(split, f)

# Process all languages for Wikipedia and CHILDES
for lang in ['en', 'fr', 'de']:
    for corpus_type in ['childes','wiki']:
        process_corpus(lang, corpus_type)
