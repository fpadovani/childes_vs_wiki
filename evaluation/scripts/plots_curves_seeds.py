import os
from tkinter import font
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import math 
from utils_1 import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.variables import *
from evaluation.scripts.evaluate_curves_seeds import output_folders

def extract_checkpoints_and_paradigms(results):
    """
    Extracts unique checkpoints and paradigms from the loaded data.
    """
    checkpoints = sorted(set(
        int(ckpt) for model_data in results.values() for ckpt in model_data["overall_avg"].keys()
    ))
    
    sample_ckpt = next(iter(results['childes']['avg_per_paradigm']))
    paradigms = list(results['childes']['avg_per_paradigm'][sample_ckpt].keys())
    
    return checkpoints, paradigms

def prepare_accuracy_variance_data(results, checkpoints, paradigms):
    """
    Organizes accuracy and variance data into structured lists for plotting.
    """
    sorted_paradigms = sorted(paradigms)
    accuracies, variances = [], []
    
    for model_type in ["childes", "wikipedia"]:
        model_accuracies, model_variances = [], []
        
        for checkpoint in checkpoints:
            checkpoint_str = str(checkpoint)
            paradigm_accuracies = []
            paradigm_variances = []

            for paradigm in sorted_paradigms:
                avg_results = results[model_type]["avg_per_paradigm"].get(checkpoint_str, {}).get(paradigm, 0)
                var_results = results[model_type]["var_per_paradigm"].get(checkpoint_str, {}).get(paradigm, 0)
                
                paradigm_accuracies.append(avg_results)
                paradigm_variances.append(var_results)

            model_accuracies.append(paradigm_accuracies)
            model_variances.append(paradigm_variances)

        accuracies.append(model_accuracies)
        variances.append(model_variances)

    return accuracies, variances, sorted_paradigms


def plot_avg_accuracy_with_variance(results, language, save_folder):
    """
    Plots the overall average accuracy with variance for Wikipedia and Childes models.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_type, color in [("childes", "#FF0000"), ("wikipedia", "#CCAA00")]:
        model_data = results[model_type]
        checkpoints = sorted(map(int, model_data["overall_avg"].keys()))
        avg_values = [model_data["overall_avg"][str(ckpt)] for ckpt in checkpoints]
        var_values = [model_data["overall_var"][str(ckpt)] for ckpt in checkpoints]

        label = "CDL" if model_type == "childes" else model_type.capitalize()

        # Scale accuracy to percentage
        avg_values = [val * 100 for val in avg_values]
        var_values = [val * 100 for val in var_values]

        ax.plot(checkpoints, avg_values, label=label, linewidth = 2,color=color)
        ax.fill_between(checkpoints, 
                        np.array(avg_values) - np.array(var_values), 
                        np.array(avg_values) + np.array(var_values), 
                        color=color, alpha=0.2)

    ax.set_title(f'Overall AVG Accuracy for {language.upper()}', fontsize=17)
    ax.set_xlabel('Training Steps', fontsize = 14)
    ax.set_ylabel('Accuracy', fontsize = 14)
    ax.legend(title="Model")

    ax.set_ylim(0, 100)
    ax.set_xscale('linear')
    ax.set_xticks([1000, 20000, 40000, 60000, 80000, 100000])
    ax.set_xticklabels([1000, 20000, 40000, 60000, 80000, 100000], rotation=45)
    ax.set_xlim(1000, 100000)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f'{language}_overall_accuracy.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_accuracy_per_paradigm(checkpoints, accuracies, variances, paradigms, language, save_folder, col1, col2):
    """
    Plots the accuracy per paradigm for Childes and Wikipedia models across checkpoints.
    """


    num_paradigms = len(paradigms)
    rows = math.ceil(num_paradigms / 3)
    cols = min(num_paradigms, 3)
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4), squeeze=False)
    axes = axes.flatten()

    for i, paradigm in enumerate(paradigms):
        ax = axes[i]

        for model_idx, (model_name, color) in enumerate([("childes", f"{col1}"), ("wikipedia", f"{col2}")]):
            model_checkpoints = checkpoints
            model_accuracies = [acc[i] * 100 for acc in accuracies[model_idx]]
            model_variances = [var[i] * 100 for var in variances[model_idx]]

            label = "CHILDES" if model_name == "childes" else model_name.capitalize()

            ax.plot(model_checkpoints, model_accuracies, label=label, linewidth = 2, color=color)
            ax.fill_between(model_checkpoints, 
                            np.array(model_accuracies) - np.array(model_variances), 
                            np.array(model_accuracies) + np.array(model_variances), 
                            color=color, alpha=0.2)

        title = paradigms_name.get(paradigm, paradigm)
        ax.set_title(title, fontsize=13)         
        #ax.set_xlabel('Training Steps', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_ylim(0, 100)
        ax.set_xscale('linear')
        ax.set_xticks([1000, 20000, 40000, 60000, 80000, 100000])
        ax.set_xticklabels([1000, 20000, 40000, 60000, 80000, 100000], rotation=45)
        ax.set_xlim(1000, 100000)
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Model", loc='lower right', bbox_to_anchor=(0.98, 0.02), prop={'size': 15}, title_fontsize=15)

        plt.tight_layout(rect=[0, 0.05, 1, 1])

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    os.makedirs(save_folder, exist_ok=True)
    save_path = Path(save_folder) / f'{language}_accuracy_per_paradigm.png'
    plt.tight_layout(h_pad=3.0)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved accuracy per paradigm plots: {save_path}")


def plot_accuracy_per_paradigm_blimp(checkpoints, accuracies, variances, paradigms, language, save_folder):
    """
    Plots the accuracy per paradigm in two separate figures if too many paradigms are present.
    """
    def create_plot(paradigm_subset, suffix,add_legend=False):
        num_paradigms = len(paradigm_subset)
        rows = math.ceil(num_paradigms / 3)
        cols = min(num_paradigms, 3)
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4), squeeze=False)
        axes = axes.flatten()

        for i, paradigm in enumerate(paradigm_subset):
            ax = axes[i]
            for model_idx, (model_name, color) in enumerate([("childes", "#FF0000"), ("wikipedia", "#CCAA00")]):
                model_checkpoints = checkpoints
                model_accuracies = [acc[paradigms.index(paradigm)] * 100 for acc in accuracies[model_idx]]
                model_variances = [var[paradigms.index(paradigm)] * 100 for var in variances[model_idx]]

                label = "CDL" if model_name == "childes" else model_name.capitalize()

                ax.plot(model_checkpoints, model_accuracies, label=label, linewidth=2, color=color)
                ax.fill_between(model_checkpoints,
                                np.array(model_accuracies) - np.array(model_variances),
                                np.array(model_accuracies) + np.array(model_variances),
                                color=color, alpha=0.2)

            title = paradigms_name.get(paradigm, paradigm)
            ax.set_title(title, fontsize=17)
            ax.set_ylabel('Accuracy', fontsize=17)
            ax.set_ylim(0, 100)
            ax.set_xscale('linear')
            ax.set_xticks([1000, 20000, 40000, 60000, 80000, 100000])
            ax.set_xticklabels([1000, 20000, 40000, 60000, 80000, 100000], rotation=45, fontsize=12)
            ax.set_xlim(1000, 100000)
            ax.grid(True, axis='x', linestyle='--', alpha=0.3)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        if add_legend:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, title="Model", loc='upper right', bbox_to_anchor=(0.85, 0.10), 
                       prop={'size': 21}, title_fontsize=21)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        os.makedirs(save_folder, exist_ok=True)
        save_path = Path(save_folder) / f'{language}_accuracy_per_paradigm_{suffix}.png'
        plt.savefig(save_path)
        plt.close()
        print(f"Saved accuracy per paradigm plot: {save_path}")

    # Split into two parts
    midpoint = len(paradigms) // 2
    create_plot(paradigms[:midpoint], 'part1')
    create_plot(paradigms[midpoint:], 'part2', add_legend=True)


def main():
    """
    Main function to load data, process it, and generate plots.
    """
    language = input("Enter the language ('en', 'fr', 'de'): ").strip().lower()
    model_type = input("Enter the model type ('mlm' or 'clm'): ").strip()

    if language not in {'en', 'fr', 'de'} or model_type not in ['mlm', 'clm']:
        print("Invalid language input.")
        return
    
    if model_type == 'mlm':
        save_folder = PLOT_RESULTS_FOLDER_MLM
    elif model_type == 'clm':
        save_folder = PLOT_RESULTS_FOLDER_CLM

    json_folders = {
        "en": output_folders[f'zorro_{model_type}'], #change this to get the correct folder for blimp and zorro
        "fr": output_folders[f'fr_evalset_{model_type}'],
        "de": output_folders[f'de_evalset_{model_type}'],
    }

    base_folder = json_folders[language]
    json_folders_per_paradigm = {
        "childes": os.path.join(base_folder, f"{language}_child_{model_type}_results"),
        "wikipedia": os.path.join(base_folder, f"{language}_wiki_{model_type}_results"),
    }
    

    results = load_all_data(json_folders_per_paradigm)
    checkpoints, paradigms = extract_checkpoints_and_paradigms(results)
    accuracies, variances, sorted_paradigms = prepare_accuracy_variance_data(results, checkpoints, paradigms)

    if language == 'en':
        if 'clams' in base_folder.split('/')[-1]:
            save_folder =  save_folder + f"/{language}/clams"
            col1, col2 = get_colors_curves('clams', language)['childes'],get_colors_curves('clams', language)['wiki']
            plot_accuracy_per_paradigm(checkpoints, accuracies, variances, sorted_paradigms, language, save_folder,col1, col2) 
        elif 'blimp' in base_folder.split('/')[-1]:
            save_folder = save_folder + f"/{language}/blimp"
            plot_accuracy_per_paradigm_blimp(checkpoints, accuracies, variances, sorted_paradigms, language, save_folder)
        
        elif 'zorro' in base_folder.split('/')[-1]:
            save_folder = save_folder + f"/{language}/zorro"
            col1, col2 = get_colors_curves('zorro', language)['childes'],get_colors_curves('zorro', language)['wiki']
            plot_accuracy_per_paradigm(checkpoints, accuracies, variances, sorted_paradigms, language, save_folder, col1, col2)   
    else:
        save_folder = save_folder + f"/{language}"
        col1, col2 = get_colors_curves('clams', language)['childes'],get_colors_curves('clams', language)['wiki']
        plot_accuracy_per_paradigm(checkpoints, accuracies, variances, sorted_paradigms, language, save_folder, col1, col2) 
    
    plot_avg_accuracy_with_variance(results, language, save_folder)
    


if __name__ == "__main__":
    main()