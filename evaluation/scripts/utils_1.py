import json
import os
import pandas as pd

def get_colors_curves(dataset_dir, language):
    if "blimp" in dataset_dir.lower():
        return {
            "childes": "#1F77B4",
            "wiki": "#D85C5C"
        }
    
    elif "zorro" in dataset_dir.lower():
        return {
        "childes": "#FFD700",  # Yellow
        "wiki": "#FF0000"      # Red
    }
    elif "clams" in dataset_dir.lower():

        if language == "en":
            return {
                "childes": "#1f77b4",
                "wiki": "#d62728" 
            }
        elif language == "fr":
            return {
                "childes": "#A5D6A7", 
                "wiki": "#FFCC80"
            }
        elif language == "de":
            return {
            "childes": "#D1A7E0",
            "wiki": "#D5A39D" 
    }
    else:

        return {
            "childes": "#B1C9E6",
            "wiki": "#F4A6A0" 
        }

def get_colors_barplots(dataset_dir, language):
    if "blimp" in dataset_dir.lower():
        return {
            "childes_best_clm": "#1F77B4",
            "wikipedia_best_clm": "#D85C5C"
        }
    
    elif "zorro" in dataset_dir.lower():
        return {
            "childes_best_clm": "#A0E0A1",
            "wikipedia_best_clm": "#FF9E9E"
        }
    elif "clams" in dataset_dir.lower():

        if language == "en":
            return {
                "childes_best_clm": "#1f77b4",
                "wikipedia_best_clm": "#d62728",
                "childes_best_mlm": "#1f77b4",
                "wikipedia_best_mlm": "#d62728" 
                
            }
        elif language == "fr":
            return {
                "childes_best_clm": "#A5D6A7", 
                "wikipedia_best_clm": "#FFCC80",
                "childes_best_mlm": "#A5D6A7", 
                "wikipedia_best_mlm": "#FFCC80"
            }
        elif language == "de":
            return {
            "childes_best_clm": "#D1A7E0",
            "wikipedia_best_clm": "#D5A39D",
            "childes_best_mlm": "#D1A7E0",
            "wikipedia_best_mlm": "#D5A39D" 
    }
    else:

        return {
            "childes_best_clm": "#B1C9E6",
            "wikipedia_best_clm": "#F4A6A0" 
        }
    

def load_json(filepath):
    """Loads JSON data from a given file path."""
    if not os.path.exists(filepath):
        print(f"Warning: Missing file {filepath}")
        return {}
    with open(filepath, "r") as f:
        return json.load(f)
    


def load_all_data(json_folders):
    """
    Loads all required data for Childes and Wikipedia models:
    - Averaged results per paradigm
    - Variance results per paradigm
    - Overall averaged results
    - Overall variance results
    """
    results = {}
    for model_type, folder_path in json_folders.items():
        results[model_type] = {
            "avg_per_paradigm": load_json(os.path.join(folder_path, "averaged_results_per_paradigm.json")),
            "var_per_paradigm": load_json(os.path.join(folder_path, "variance_results_per_paradigm.json")),
            "overall_avg": load_json(os.path.join(folder_path, "averaged_results.json")),
            "overall_var": load_json(os.path.join(folder_path, "variance_results.json"))
        }
    return results


def get_step_intervals(model):
    """
    Retrieve checkpoints divisible by 4000 dynamically for evaluation.

    Args:
        model (str): Path to the model's checkpoint directory.

    Returns:
        List[int]: List of checkpoint step numbers divisible by 4000, sorted in ascending order.
    """
    # Extract checkpoint step numbers from folder names
    checkpoints = [
        int(cp.split('-')[1])  # Extract the step number from 'checkpoint-XXXX'
        for cp in os.listdir(model)
        if 'checkpoint' in cp and cp.split('-')[1].isdigit()
    ]

    filtered_checkpoints = [step for step in checkpoints if step % 4000 == 0]
    return sorted(filtered_checkpoints)


def save_results(results, results_per_paradigm, output_path):
    """
    Save results to JSON files.
    """
    with open(os.path.join(output_path, 'results_overall.json'), 'w') as f:
        json.dump(results, f)

    with open(os.path.join(output_path, 'results_per_paradigm.json'), 'w') as f:
        json.dump(results_per_paradigm, f)


def load_sentences_from_txt(path):
    if '.DS_Store' not in path:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip().lower() for line in f.readlines()]


def load_sentences_from_csv(path):
    df = pd.read_csv(path, header=None, names=["sentence"])
    sentences = df["sentence"].astype(str).str.strip().str.lower().tolist()
    return sentences, df


# === HELPERS === #
def extract_model_columns(df, model_type_prefix):
    return [col for col in df.columns if col.startswith(model_type_prefix)]

def compute_accuracy_from_rows(scores):
    assert len(scores) % 2 == 0, "Expected even number of rows (sentence pairs)"
    correct = sum(1 for i in range(0, len(scores), 2) if scores[i] > scores[i + 1])
    return correct / (len(scores) // 2)