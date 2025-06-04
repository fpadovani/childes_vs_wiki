import spacy
import pandas as pd
import pickle
from typing import List
from tqdm import tqdm, trange
import math
import os
from conllu import parse
from spacy_conll import ConllFormatter
from supar import Parser
from diaparser.parsers import Parser as DiaParser
import sys
import re
from nltk import map_tag
from spacy_conll import init_parser
from spacy_conll.parser import ConllParser
from spacy.displacy import render
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.variables import CLAMS_DIR, PARSED_DATASETS


CLAMS_DIR = '/Users/frapadovani/Desktop/CLAMS_analysis/CLAMS/'

# Language-specific configurations
LANG_CONFIGS = {
    "eng": {
        "spacy_model": "en_core_web_trf",
        "supar_model": "biaffine-dep-en",
        "clams_folder": "en_evalset_ok/",
        'clams_new_folder': 'en_new_clams/',
        "pickle_folder": "clams_eng_deps",
        "conllu_folder": "clams_conllu_files_eng"
    },
    "fr": {
        "spacy_model": "fr_dep_news_trf",
        "supar_model": None,  # Uses Spacy only
        "clams_folder": "fr_evalset_ok/",
        'clams_new_folder': 'fr_new_clams/',
        "pickle_folder": "clams_fr_deps",
        "conllu_folder": "clams_conllu_files_fr"
    },
    "de": {
        "spacy_model": None,  # Uses Diaparser only
        "supar_model": "de_hdt.dbmdz-bert-base",
        "clams_folder": "de_evalset_ok/",
        'clams_new_folder': 'de_new_clams/',
        "pickle_folder": "clams_de_deps",
        "conllu_folder": "clams_conllu_files_de"
    }
}

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


def modify_conllu_string(doc, arcs, rels):
    """
    Modify the CoNLL-U string for the doc using arcs and rels.
    """
    conllu_lines = [line for line in doc._.conll_str.strip().split("\n") if line.strip()]
    modified_lines = []
    arc_idx = 0  # Separate index for arcs and rels

    for idx, line in enumerate(conllu_lines):

        columns = line.split("\t")

        if len(columns) >= 8:
            columns[6] = str(arcs[arc_idx])   # Update HEAD (column 7)
            columns[7] = rels[arc_idx]        # Update RELATION (column 8)
            arc_idx += 1  # Only increment if a valid arc is processed
        else:
            print(f"Skipping line due to index mismatch: {line}")

        modified_lines.append("\t".join(columns))
        
    
    return "\n".join(modified_lines)


def parse_corpus(nlp, corpus: List[str], conllu_file: str, batch_size=64, empty_file=False):
    iterator = get_batch(corpus, batch_size)

    # Ensure directory exists
    conllu_dir = os.path.dirname(conllu_file)
    if not os.path.exists(conllu_dir):
        os.makedirs(conllu_dir, exist_ok=True)

    
    with open(conllu_file, 'a', encoding="utf-8") as f:
        if empty_file:
            f.truncate(0)
    
        for batch in tqdm(iterator, total=len(corpus)//batch_size):
            true_batch = [item.capitalize() + '.' for item in batch if item is not None]
    
            docs = list(nlp.pipe(
                true_batch, 
                disable=["tok2vec", "ner", "attribute_ruler", "lemmatizer", "textcat"],
                batch_size=batch_size,
            ))
    
            doc_idx = 0
            for item in batch:
                if item is None:
                    f.write('None\n\n')
                else:
                    doc = docs[doc_idx]
                    f.write(doc._.conll_str + '\n')    
                    doc_idx += 1


def parse_corpus_supar(nlp, parser, corpus: List[str], conllu_file: str, batch_size=64, empty_file=False):
    iterator = get_batch(corpus, batch_size)
    
    with open(conllu_file, 'a', encoding="utf-8") as f:
        if empty_file:
            f.truncate(0)
    
        for batch in tqdm(iterator, total=len(corpus)//batch_size):
            true_batch = [item.strip() for item in batch if item is not None]
    
            docs = list(nlp.pipe(
                true_batch, 
                disable=["tok2vec", "ner", "attribute_ruler", "lemmatizer", "textcat"],
                batch_size=batch_size,
            ))

            dataset = list(parser.predict(true_batch, lang='en', prob=True, verbose=False))
            
            doc_idx = 0
            for item in batch:
                if item is None:
                    f.write('None\n\n')
                else:
                    doc = docs[doc_idx]
                    arcs = dataset[doc_idx].arcs
                    rels = dataset[doc_idx].rels
                    modified_conllu = modify_conllu_string(doc, arcs, rels)
                    f.write(modified_conllu + '\n\n')
                    doc_idx += 1

def parse_corpus_diaparser(parser, corpus: List[str], conllu_file: str, batch_size=64, empty_file=False):
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

def process_language(lang: str):
    """General function to process CLAMS datasets for a given language."""
    config = LANG_CONFIGS[lang]

    # Load NLP Model
    if config["spacy_model"]:
        nlp = spacy.load(config["spacy_model"])
        nlp.add_pipe("conll_formatter", last=True)

    # Load parser
    parser = None
    if config["supar_model"] and lang == "eng":
        parser = Parser.load(config["supar_model"])
    elif lang == "de":
        parser = DiaParser.load(config["supar_model"])

    corpus_names = os.listdir(os.path.join(CLAMS_DIR, config["clams_folder"]))
    corpus_names_ = [corpus.split('.txt')[0] for corpus in corpus_names]
    conllu_files = [f'{PARSED_DATASET}clams/conllu/{config["conllu_folder"]}/{corpus}_{lang}_deps.conllu' for corpus in corpus_names_]

    for corpus_file, conllu_file in zip(corpus_names, conllu_files):
        with open(os.path.join(CLAMS_DIR, config["clams_folder"], corpus_file), encoding="utf-8") as f:
            corpus = f.read().strip().split('\n')

        if lang == "de":
            parse_corpus_diaparser(parser, corpus, conllu_file)
        elif parser:
            parse_corpus_supar(nlp, parser, corpus, conllu_file)
        else:
            parse_corpus(nlp, corpus, conllu_file)

    for deps_file, conllu_file in zip(corpus_names_, conllu_files):
        with open(conllu_file, encoding="utf-8") as f:
            raw_deps = f.read().strip().split('\n\n')

        new_docs = [parse(preprocess_raw_dep(dep))[0] for dep in raw_deps if dep != 'None']

        for i in trange(1):
            split = new_docs
            print(new_docs)
            with open(f'{PARSED_DATASET}clams/pickle/{config["pickle_folder"]}/clams_{lang}_{deps_file}_deps_{i}.pickle', 'wb') as f:
                pickle.dump(split, f)

def parse_extra_files_single(minimal_pair_csv, lang, clams_type):
    """Parse the minimal pairs CSV file."""

    config = LANG_CONFIGS[lang]

    # Load NLP Model
    if config["spacy_model"]:
        nlp = spacy.load(config["spacy_model"])
        nlp.add_pipe("conll_formatter", last=True)

    if config["supar_model"] and lang == "eng":
        parser = Parser.load(config["supar_model"])

    minimal_pairs = pd.read_csv(minimal_pair_csv)
    corpus = minimal_pairs['sentence'].tolist()
    corpus_name = os.path.basename(minimal_pair_csv).split('.csv')[0]
    conllu_file = f'{PARSED_DATASET}/{clams_type}/conllu/{config["conllu_folder"]}/{corpus_name}_{lang}_deps.conllu'
    parse_corpus_supar(nlp, parser, corpus, conllu_file)

    with open(conllu_file, encoding="utf-8") as f:
        raw_deps = f.read().strip().split('\n\n')

    new_docs = [parse(preprocess_raw_dep(dep))[0] for dep in raw_deps if dep != 'None']
    for i in trange(1):
        split = new_docs
        with open(f'{PARSED_DATASET}{clams_type}/pickle/{config["pickle_folder"]}/clams_{lang}_{corpus_name}_deps_{i}.pickle', 'wb') as f:
            pickle.dump(split, f)


def parse_extra_files(clams_type, lang):
    """Parse the minimal pairs CSV file."""

    config = LANG_CONFIGS[lang]

    # Load NLP Model
    if config["spacy_model"]:
        nlp = spacy.load(config["spacy_model"])
        nlp.add_pipe("conll_formatter", last=True)

    if config["supar_model"] and lang == "eng":
        parser = Parser.load(config["supar_model"])
    
    elif lang == "de":
        parser = DiaParser.load(config["supar_model"])
    
    elif lang == "fr":
        parser = None

    for sub in ['childes', 'wiki']: 
        minimal_pairs_path = os.path.join(CLAMS_DIR, config[f'{clams_type}_folder'], sub)
        for minimal_pair_file in os.listdir(minimal_pairs_path):
            minimal_pair_df = pd.read_csv(os.path.join(minimal_pairs_path, minimal_pair_file),header=None)
            corpus = minimal_pair_df.iloc[:, 0].tolist()
            corpus_name = os.path.basename(minimal_pair_file).split('.csv')[0]
            conllu_file = f'{PARSED_DATASET}{clams_type}/{sub}/conllu/{config["conllu_folder"]}/{corpus_name}_{lang}_deps.conllu'
            if not os.path.exists(os.path.dirname(conllu_file)):
                os.makedirs(os.path.dirname(conllu_file), exist_ok=True)

            if lang == "de":
                parse_corpus_diaparser(parser, corpus, conllu_file)
            elif lang == "eng":
                parse_corpus_supar(nlp, parser, corpus, conllu_file)
            else:
                parse_corpus(nlp, corpus, conllu_file)


            with open(conllu_file, encoding="utf-8") as f:
                raw_deps = f.read().strip().split('\n\n')

            new_docs = [parse(preprocess_raw_dep(dep))[0] for dep in raw_deps if dep != 'None']
            for i in trange(1):
                split = new_docs
                pickle_file = f'{PARSED_DATASET}{clams_type}/{sub}/pickle/{config["pickle_folder"]}/clams_{lang}_{corpus_name}_deps_{i}.pickle'
                if not os.path.exists(os.path.dirname(pickle_file)):
                    os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
                with open(pickle_file, 'wb') as f:
                    pickle.dump(split, f)



# Process all three languages
for lang in ['fr', 'eng','de']:
    #let the user select between clams and clams_new
    clams_type = input("Which dataset do you want to parse? (clams/clams_new): ")
    if clams_type == 'clams':
        process_language(lang)
    else:
        parse_extra_files(clams_type,lang)
