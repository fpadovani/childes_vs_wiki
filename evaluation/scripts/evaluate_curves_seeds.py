import os
import sys
import json
import numpy as np
from pathlib import Path
from utils_1 import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.variables import *  # Assuming this module contains paths like TRAINED_MODELS, etc.
from minicons import scorer
import math

# Seeds to process
SEEDS = [30, 13, 42]

test_suite_folders = {'zorro_clm': ZORRO_DIR,
                      'blimp_clm': BLIMP_DIR,
                      'en_evalset_clm': CLAMS_FOLDER_ENG,
                      'fr_evalset_clm': CLAMS_FOLDER_FR,
                      'de_evalset_clm': CLAMS_FOLDER_DE,
                      'en_fitclams_clm_childes': FITCLAMS_ENG_childes,
                      'fr_fitclams_clm_childes': FITCLAMS_FR_childes,
                      'de_fitclams_clm_childes': FITCLAMS_DE_childes,
                      'en_fitclams_clm_wiki': FITCLAMS_ENG_wikipedia,
                      'fr_fitclams_clm_wiki': FITCLAMS_FR_wikipedia,
                      'de_fitclams_clm_wiki': FITCLAMS_DE_wikipedia}

output_folders = {'zorro_clm': JSON_RESULT_ZORRO_clm,
                    'blimp_clm': JSON_RESULT_BLIMP_clm,
                    'en_evalset_clm': JSON_RESULT_clams_ENG_clm,
                    'fr_evalset_clm': JSON_RESULT_clams_FR_clm,
                    'de_evalset_clm': JSON_RESULT_clams_DE_clm,
                    'en_fitclams_clm_childes': JSON_RESULT_fitclams_ENG_childes_clm,
                    'fr_fitclams_clm_childes': JSON_RESULT_fitclams_FR_childes_clm,
                    'de_fitclams_clm_childes': JSON_RESULT_fitclams_DE_childes_clm,
                    'en_fitclams_clm_wiki': JSON_RESULT_fitclams_ENG_wiki_clm,
                    'fr_fitclams_clm_wiki': JSON_RESULT_fitclams_DE_wiki_clm,
                    'de_fitclams_clm_wiki': JSON_RESULT_fitclams_FR_wiki_clm}





def evaluate_checkpoint(checkpoint_path, test_suite_folder, lower_case=True):
    """
    Evaluate a single checkpoint on the test suite and return paradigm accuracies.

    Parameters:
        checkpoint_path (str): Path to the checkpoint.
        test_suite_name (str): Name of the test suite.
        lower_case (bool): Whether to lowercase the sentences for evaluation.

    Returns:
        dict: Paradigm accuracies.
    """
    # Check for .safetensors file
    safetensors_file = os.path.join(checkpoint_path, "model.safetensors")
    if not os.path.exists(safetensors_file):
        print(f"Skipping checkpoint {checkpoint_path}: .safetensors file not found.")
        return None  # Skip evaluation if the file is missing

    try:
        if 'clm' in checkpoint_path:
            lm = scorer.IncrementalLMScorer(checkpoint_path, 'cpu')
        else:
            lm = scorer.MaskedLMScorer(checkpoint_path, 'cpu')
    except Exception as e:
        print(f"Error loading safetensors for {checkpoint_path}: {e}")
        return None
    
    paradigm_accuracies = {}

    test_suite_folder = Path(test_suite_folder)

    for path_paradigm in test_suite_folder.glob('*.txt'):
        print(f"Processing: {path_paradigm}")

        # Load sentences
        sentences_ = path_paradigm.read_text().strip().split('\n')
        assert len(sentences_) % 2 == 0, f"File {path_paradigm} does not have an even number of lines!"
        # Lowercase if needed
        sentences = [s.lower() for s in sentences_] if lower_case else sentences_

        # Process pairs of sentences
        correct_count = 0
        total_count = 0
        for i in range(0, len(sentences), 2):
            grammatical = sentences[i]
            ungrammatical = sentences[i + 1]

            # Score both sentences
            stimuli = [grammatical, ungrammatical]
            if 'clm' in checkpoint_path:
                scores = lm.sequence_score(stimuli, reduction=lambda x: x.sum(0).item(), bow_correction=True)
            else:
                scores = lm.sequence_score(stimuli, reduction = lambda x: x.sum(0).item(), PLL_metric='within_word_l2r')

            # Lower surprisal is better
            if scores[0] > scores[1]:
                correct_count += 1

            total_count += 1

        # Calculate paradigm accuracy
        paradigm_accuracy = correct_count / total_count
        paradigm_accuracies[path_paradigm.name] = paradigm_accuracy
        print(f"Accuracy for {path_paradigm.name}: {paradigm_accuracy:.2%}")

    return paradigm_accuracies


def process_folder(trained_model_to_use, test_suite_name, step_intervals):
    """
    Process all checkpoints in a folder and evaluate them.

    Parameters:
        trained_model_to_use (str): Base path for the trained model.
        test_suite_name (str): Name of the test suite.
        step_intervals (list[int]): List of step intervals for checkpoint evaluation.

    Returns:
        dict: Results for all steps.
    """
    results = {}
    results_per_paradigm = {}

    # Evaluate each checkpoint matching step intervals
    for interval in step_intervals:
        checkpoint_name = f"checkpoint-{interval}"
        print(f"Evaluating {checkpoint_name}...")
        checkpoint_path = os.path.join(trained_model_to_use, checkpoint_name)
        paradigm_accuracies = evaluate_checkpoint(checkpoint_path, test_suite_name)
        if paradigm_accuracies is None:
            print(f"Skipping {checkpoint_name} due to missing or invalid safetensors file.")
            continue

        # Aggregate results
        overall_accuracy = np.mean(list(paradigm_accuracies.values()))
        results[interval] = round(overall_accuracy, 3)
        results_per_paradigm[interval] = paradigm_accuracies

    return results, results_per_paradigm


def evaluate_model_with_seeds(trained_model_base_path, test_suite_folder, seeds, output_folder):
    """
    Evaluate the model with multiple seeds and save results for each seed.

    Parameters:
        trained_model_base_path (str): Base path of the trained model.
        test_suite_folder (str): Path to the test suite folder.
        seeds (list[int]): List of seed values.
        output_folder (str): Folder to save results.

    Returns:
        dict: Aggregated results (averaged across seeds) for plotting.
    """
    aggregated_results = {}
    aggregated_results_per_paradigm = {}

    for seed in seeds:
        print(f"Processing seed: {seed}")
        seed_model_path = trained_model_base_path + f"_{seed}_new"
        if not os.path.exists(seed_model_path):
            print(f"Model path does not exist for seed {seed}: {seed_model_path}")
            continue
        
        early_checkpoints = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        late_checkpoints = [20000, 32000, 40000, 52000, 60000, 72000, 80000, 92000, 100000]
        checkpoints = early_checkpoints + late_checkpoints
        results, results_per_paradigm = process_folder(seed_model_path, test_suite_folder, checkpoints)

        # Save results for this seed
        seed_output_path = os.path.join(output_folder, f"results_seed_{seed}.json")
        with open(seed_output_path, 'w') as f:
            json.dump(results, f)
        
        seed_paradigm_output_path = os.path.join(output_folder, f"results_paradigm_seed_{seed}.json")
        with open(seed_paradigm_output_path, 'w') as f:
            json.dump(results_per_paradigm, f)

        # Aggregate results
        for step, accuracy in results.items():
            aggregated_results.setdefault(step, []).append(accuracy)
        for step, paradigms in results_per_paradigm.items():
            for paradigm, accuracy in paradigms.items():
                aggregated_results_per_paradigm.setdefault(step, {}).setdefault(paradigm, []).append(accuracy)

    # Average and variance across seeds
    averaged_results = {step: np.mean(acc_list) for step, acc_list in aggregated_results.items()}
    variance_results = {step: np.var(acc_list) for step, acc_list in aggregated_results.items()}

    averaged_results_per_paradigm = {}
    variance_results_per_paradigm = {}
    for step, paradigms in aggregated_results_per_paradigm.items():
        averaged_results_per_paradigm[step] = {paradigm: np.mean(acc_list) for paradigm, acc_list in paradigms.items()}
        variance_results_per_paradigm[step] = {paradigm: np.var(acc_list) for paradigm, acc_list in paradigms.items()}

    # Save aggregated results
    with open(os.path.join(output_folder, 'averaged_results.json'), 'w') as f:
        json.dump(averaged_results, f)
    with open(os.path.join(output_folder, 'variance_results.json'), 'w') as f:
        json.dump(variance_results, f)

    with open(os.path.join(output_folder, 'averaged_results_per_paradigm.json'), 'w') as f:
        json.dump(averaged_results_per_paradigm, f)
    with open(os.path.join(output_folder, 'variance_results_per_paradigm.json'), 'w') as f:
        json.dump(variance_results_per_paradigm, f)

    return averaged_results, variance_results, averaged_results_per_paradigm, variance_results_per_paradigm


def main():
    language = input("Enter the language ('en','fr','de'): ").strip()
    model_type = input("Enter the model type ('mlm' or 'clm'): ").strip()


    if language not in {'en', 'fr', 'de'} or model_type not in ['mlm', 'clm']:
        print("Invalid language input.")
        return

    if language == 'en':
        test_suite_options = [f'blimp_{model_type}', f'zorro_{model_type}', f'en_evalset_{model_type}']
    elif language == 'fr':
        test_suite_options = [f'fr_evalset_{model_type}']
    elif language == 'de':
        test_suite_options = [f'de_evalset_{model_type}']
    else:
        print("Invalid language input.")
        return

    for suite in test_suite_options:
        print(f"Processing test suite: {suite}")
        suite_folder = test_suite_folders[suite]
        output_json_folder = output_folders[suite]
        if model_type == 'mlm':
            output_json_folder = output_json_folder.replace('clm', 'mlm')


        model_childes = os.path.join(TRAINED_MODELS, f'{language}_child_{model_type}')
        model_wikipedia = os.path.join(TRAINED_MODELS, f'{language}_wiki_{model_type}')

        models = {'childes': model_childes,'wikipedia': model_wikipedia}

        for _, model_path in models.items():
            print(f"Processing model: {model_path.split('/')[-1]}")
            model_output_folder = os.path.join(output_json_folder, f"{model_type}_results")
            os.makedirs(model_output_folder, exist_ok=True)

            # Evaluate model with seeds
            evaluate_model_with_seeds(model_path, suite_folder, SEEDS, model_output_folder)

if __name__ == "__main__":
    main()