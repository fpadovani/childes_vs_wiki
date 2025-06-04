import os
import sys
import pandas as pd
from utils_1 import *
import numpy as np
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from utils.variables import *

AVAILABLE_LANGUAGES = ['en', 'fr', 'de']
OLD_STYLE_BENCHMARKS = ['CLAMS_SCORES', 'BLIMP_SCORES', 'ZORRO_SCORES']
NEW_STYLE_BENCHMARK = 'FIT_CLAMS'
SUBSETS = ['childes', 'wiki']
language_mapping = {'en': 'ENGLISH', 'fr': 'FRENCH', 'de': 'GERMAN'}

def enrich_entries(entries):
        return [{
            **entry,
            "avg_acc": np.mean(entry["seed_accs"]),
            "std_acc": np.std(entry["seed_accs"])
        } for entry in entries]

def sort_entries(entries, order):
    """Sort entries by a custom paradigm order."""
    return sorted(entries, key=lambda e: order.index(e["paradigm"]) if e["paradigm"] in order else len(order))


def plot_bar_with_error(ax, x, accs, stds, offset, width, label, color):
    """Draw a bar chart with error bars on given axis."""
    ax.bar(x + offset, accs, width, yerr=stds, label=label,
           color=color, capsize=4, alpha=0.8)
    


def plot_comparison(key1, key2, title, dataset_dir, results, colors, paradigm_mapping, language):
    entries1 = results[key1]
    entries2 = results[key2]
    if not entries1 or not entries2:
        return

    entries1 = enrich_entries(entries1)
    entries2 = enrich_entries(entries2)

    paradigms = [e["paradigm"] for e in entries1]
    x = np.arange(len(paradigms))
    width = 0.35

    # Adjust plot size
    if "blimp" in dataset_dir.lower() or "zorro" in dataset_dir.lower():
        figsize = (20, 7)
    else:
        figsize = (14, 6)

    fig, ax = plt.subplots(figsize=figsize)

    accs1 = [e["avg_acc"] for e in entries1]
    accs2 = [e["avg_acc"] for e in entries2]
    stds1 = [e["std_acc"] for e in entries1]
    stds2 = [e["std_acc"] for e in entries2]

    ax.bar(x - width/2, accs1, width, yerr=stds1, label='CDL', 
                   color=colors["childes_best_clm"], capsize=4, alpha=0.8)
    ax.bar(x + width/2, accs2, width, yerr=stds2, label='Wikipedia', 
                   color=colors["wikipedia_best_clm"], capsize=4, alpha=0.8)

    xtick_labels = [paradigm_mapping.get(p, p) for p in paradigms]
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_title(f'Accuracy Scores on {dataset_dir.split("_")[0]} ({language.upper()})\n', 
                 fontdict={'fontsize': 15, 'fontweight': 'bold', 'family': 'serif'})

    ax.set_ylabel('Accuracy', fontdict={'fontsize': 16, 'fontweight': 'bold', 'family': 'serif'}, labelpad=25)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, linestyle='--', linewidth=1, color='red', alpha=0.5, zorder=10)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=15, ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout(rect=[0.02, 0.02, 1, 0.95])
    plt.savefig(os.path.join(
        BARPLOTS_DIR, 'old_benchmarks',
        f"{language}_{key1.split('_')[1]}_{key1.split('_')[2]}_{dataset_dir.split('_')[0]}.pdf"
    ), bbox_inches='tight')
    plt.show()


def plot_dual_comparison(
    key1_clm, key2_clm, key1_mlm, key2_mlm, 
    dataset_dir, results, colors, paradigm_mapping, language
):
    def sort_by_desired_order(entries, order):
        return sorted(entries, key=lambda e: order.index(e["paradigm"]))

    desired_order = [
        'Simple\nAgrmt', 'Agrmt in\nVP coords', 'Agrmt in\nlong VP coords',
        'Agrmt in\nprep phrases', 'Agrmt in\nsubj rel clauses',
        'Agrmt across\nobj rel clauses', 'Agrmt within\nobj rel clauses'
    ]


    entries_clm_1 = enrich_entries(sort_by_desired_order(results[key1_clm], desired_order))
    entries_clm_2 = enrich_entries(sort_by_desired_order(results[key2_clm], desired_order))
    entries_mlm_1 = enrich_entries(sort_by_desired_order(results[key1_mlm], desired_order))
    entries_mlm_2 = enrich_entries(sort_by_desired_order(results[key2_mlm], desired_order))

    if not entries_clm_1 or not entries_clm_2 or not entries_mlm_1 or not entries_mlm_2:
        return

    paradigms = desired_order
    x = np.arange(len(paradigms))
    width = 0.35
    figsize = (20, 5.8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    

    def plot_single(ax, entries1, entries2, title, color1, color2, show_ylabel=False):
        accs1 = [e["avg_acc"] for e in entries1]
        accs2 = [e["avg_acc"] for e in entries2]
        stds1 = [e["std_acc"] for e in entries1]
        stds2 = [e["std_acc"] for e in entries2]

        ax.bar(x - width/2, accs1, width, yerr=stds1, label='CHILDES', color=color1, capsize=4, alpha=1)
        ax.bar(x + width/2, accs2, width, yerr=stds2, label='Wikipedia', color=color2, capsize=4, alpha=1)


        ax.axhline(0.5, linestyle='--', linewidth=1, color='red', alpha=0.5, zorder=10)
        ax.set_xticks(x)

        xtick_labels = [paradigm_mapping.get(p, p) for p in paradigms]
        ax.set_xticklabels(xtick_labels, rotation=60, ha='right', fontsize=18)

        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylim(0, 1.0)
        if show_ylabel:
            ax.set_ylabel("Accuracy", fontsize=15, labelpad=15)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # ---- Generate both subplots ----
    plot_single(
        ax1, entries_clm_1, entries_clm_2,
        "Causal Language Model", 
        colors[key1_clm], colors[key2_clm],
        show_ylabel=True
    )

    plot_single(
        ax2, entries_mlm_1, entries_mlm_2,
        "Masked Language Model", 
        colors[key1_mlm], colors[key2_mlm],
        show_ylabel=False
    )

    # ---- Shared legend ----
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=13, ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(BARPLOTS_DIR, 'old_benchmarks',
                             f"{language}_dualplot_{dataset_dir.split('_')[0]}.pdf"),
                bbox_inches='tight')
    plt.show()


def evaluate_old_style(language, dataset_dir):
    print(f"\n=== Evaluating {language.upper()} - {dataset_dir} ===")
    full_dir = os.path.join(EVAL_PROB_DIRS, dataset_dir, language)
    if not os.path.exists(full_dir):
        print(f"Skipping missing directory: {full_dir}")
        return

    colors = get_colors_barplots(dataset_dir,language)

    paradigm_files = [f for f in os.listdir(full_dir) if f.endswith(".csv")]

    results = {
        "childes_best_clm": [],
        "wikipedia_best_clm": [],
        "childes_convergence_clm": [],
        "wikipedia_convergence_clm": [],
        "childes_best_mlm": [],
        "wikipedia_best_mlm": [],
    }

    for file in paradigm_files:
        df = pd.read_csv(os.path.join(full_dir, file))
        paradigm = file.replace(".csv", "")
        pair_count = df.shape[0] // 2

        for key in results:
            prefix = f"{language}_{key}"
            cols = extract_model_columns(df, prefix)
            if not cols:
                continue
            accs = [compute_accuracy_from_rows(df[col].tolist()) for col in cols]
            results[key].append({
                "paradigm": paradigm_mapping.get(paradigm, paradigm),
                "seed_accs": accs,
                "pair_count": pair_count
            })
    
    # ---- Print accuracy comparison ----
    print(f"\n>> Accuracy Comparison on {dataset_dir.split('_')[0].upper()} subset:")

    
    for key in results:
        print(f"\n>> {key.replace('_', ' ').title()} Results:")
        for entry in results[key]:
            paradigm = entry["paradigm"]
            seed_accs = entry["seed_accs"]
            pair_count = entry["pair_count"]
            avg_acc = np.mean(seed_accs)
            std_acc = np.std(seed_accs)
            print(f"{paradigm}: "
                f"{avg_acc:.3f} (±{std_acc:.3f}) across {len(seed_accs)} seeds with {pair_count} minimal pairs")
    
    print(f"\n>> Overall Accuracy for each model on {dataset_dir.split('_')[0].upper()} subset:")
    for model_key in results:
        # Gather seed-wise accuracies per paradigm
        seed_wise = {}  # {seed_index: [acc1, acc2, ...]}
        for entry in results[model_key]:
            for i, acc in enumerate(entry["seed_accs"]):
                seed_wise.setdefault(i, []).append(acc)

        # Compute average per seed across paradigms
        seed_avgs = []
        for seed_idx in sorted(seed_wise.keys()):
            avg = np.mean(seed_wise[seed_idx])
            seed_avgs.append(avg)
            print(f"Seed {seed_idx + 1}: Avg. Accuracy = {avg:.3f}")
            
        print('\n')
        # Final overall average across seeds
        if seed_avgs:
            print(f"{model_key.replace('_', ' ').title()}: "
                f"Overall Avg. Accuracy: {np.mean(seed_avgs):.3f} (±{np.std(seed_avgs):.3f})")
    # ---- Plotting ----
        
    if "CLAMS_SCORES" in dataset_dir:
        plot_dual_comparison(
            "childes_best_clm", "wikipedia_best_clm",
            "childes_best_mlm", "wikipedia_best_mlm",
            dataset_dir, results, colors, paradigm_mapping, language
)
    else:
        plot_comparison("childes_best_clm", "wikipedia_best_clm", "Best CLM", dataset_dir,results, colors, paradigm_mapping, language)
        

def evaluate_new_clams(language):
    print(f"\n=== Evaluating NEW CLAMS for {language.upper()} ===")

    MODEL_KEYS = ['childes_best_clm', 'wikipedia_best_clm']
    SUBSETS = ['childes', 'wiki']

    results = {f"{model}_on_{subset}": [] for model in MODEL_KEYS for subset in SUBSETS}

    paradigm_mapping = {
        "long_vp_coord": "Agrmt in long VP coords",
        "simple_agrmt": "Simple Agrmt",
        "subj_rel": "Agrmt in subj rel clauses",
        "obj_rel_across_anim": "Agrmt across obj rel clauses",
        "obj_rel_within_anim": "Agrmt within obj rel clauses",
        "prep_anim": "Agrmt in prep phrases",
        "vp_coord": "Agrmt in VP coordinates"
    }

    for subset in SUBSETS:
        subset_dir = os.path.join(EVAL_PROB_DIRS, 'evaluation_probabilities', NEW_STYLE_BENCHMARK, language, subset)
        if not os.path.exists(subset_dir):
            print(f"Skipping missing: {subset_dir}")
            continue

        files = [f for f in os.listdir(subset_dir) if f.endswith(".csv")]

        for file in files:
            df = pd.read_csv(os.path.join(subset_dir, file))
            paradigm = file.replace(".csv", "")
            pair_count = len(df) // 2

            for model in MODEL_KEYS:
                prefix = f"{language}_{model}"
                cols = extract_model_columns(df, prefix)
                if not cols:
                    continue

                accs = [compute_accuracy_from_rows(df[col].tolist()) for col in cols]
                avg = np.mean(accs)
                std = np.std(accs)

                key = f"{model}_on_{subset}"
                results[key].append({
                    "paradigm": paradigm_mapping.get(paradigm, paradigm),
                    "avg_acc": avg,
                    "std_acc": std,
                    "pair_count": pair_count
                })

    # ---- Print accuracy comparison ----
    for subset in SUBSETS:
        print(f"\n>> Accuracy Comparison on {subset.upper()} subset:")

        for model_key in MODEL_KEYS:
            key = f"{model_key}_on_{subset}"
            print(f"\n>> {model_key.replace('_', ' ').title()} Results:")
            for entry in results[key]:
                print(f"{entry['paradigm']}: {entry['avg_acc']:.3f} (±{entry['std_acc']:.3f}) with {entry['pair_count']} minimal pairs")

        # ---- Overall summary ----
        print(f"\n>> Overall Accuracy on {subset.upper()} subset:")
        for model_key in MODEL_KEYS:
            key = f"{model_key}_on_{subset}"
            model_accs = [entry["avg_acc"] for entry in results[key]]
            model_stds = [entry["std_acc"] for entry in results[key]]
            if model_accs:
                print(f"{model_key.replace('_', ' ').title()}: "
                      f"Avg. Accuracy: {np.mean(model_accs):.3f} (±{np.mean(model_stds):.3f})")



def evaluate_new_clams_all_languages(languages):
    all_results = {}
    all_std_devs = {}
    pair_counts = {}
    
    for language in languages:

        MODEL_KEYS = ['childes_best_clm', 'wikipedia_best_clm']
        SUBSETS = ['childes', 'wiki']

        results = {
            "childes_best_clm_on_childes": {},
            "childes_best_clm_on_wiki": {},
            "wikipedia_best_clm_on_wiki": {},
            "wikipedia_best_clm_on_childes": {},
        }

        std_devs = {key: {} for key in results}
        

        for subset in SUBSETS:
            subset_dir = os.path.join(SCORE_DIR_FITCLAMS, language, subset)
            if not os.path.exists(subset_dir):
                continue
            files = [f for f in os.listdir(subset_dir) if f.endswith(".csv")]

            for file in files:
                df = pd.read_csv(os.path.join(subset_dir, file))
                paradigm = file.replace(".csv", "")
                if paradigm not in pair_counts:
                    pair_counts[paradigm] = len(df) // 2
                for model in MODEL_KEYS:
                    prefix = f"{language}_{model}"
                    cols = extract_model_columns(df, prefix)
                    if not cols:
                        continue
                    accs = [compute_accuracy_from_rows(df[col].tolist()) for col in cols]
                    avg = np.mean(accs)
                    std = np.std(accs)
                    key = f"{model}_on_{subset}"
                    results[key][paradigm] = avg
                    std_devs[key][paradigm] = std

        all_results[language] = results
        all_std_devs[language] = std_devs

    fig, axes = plt.subplots(len(languages), 1, figsize=(14, 12), sharex=True)
    paradigms = ['simple_agrmt','vp_coord', 'long_vp_coord', 'prep_anim','subj_rel', 'obj_rel_across_anim', 'obj_rel_within_anim']
    
    x = np.arange(len(paradigms))
    width = 0.2

    bar_positions = {
        "childes_best_clm_on_childes": -1.5 * width,
        "childes_best_clm_on_wiki": -0.5 * width,
        "wikipedia_best_clm_on_wiki": 0.5 * width,
        "wikipedia_best_clm_on_childes": 1.5 * width,
    }

    colors = {
        "childes_best_clm_on_childes": "#8B0000",
        "childes_best_clm_on_wiki": "#FF6347",
        "wikipedia_best_clm_on_wiki": "#800080",
        "wikipedia_best_clm_on_childes": "#DDA0DD",
    }

    labels = {
        "childes_best_clm_on_childes": "CHILDES-model FIT-CLAMS-C",
        "childes_best_clm_on_wiki": "CHILDES-model FIT-CLAMS-W",
        "wikipedia_best_clm_on_wiki": "Wiki-model FIT-CLAMS-W",
        "wikipedia_best_clm_on_childes": "Wiki-model FIT-CLAMS-C",
    }

    for idx, language in enumerate(languages):
        ax = axes[idx]
        results = all_results[language]
        std_devs = all_std_devs[language]

        for key, offset in bar_positions.items():
            means = [results[key].get(p, 0) for p in paradigms]
            errors = [std_devs[key].get(p, 0) for p in paradigms]
            bars = ax.bar(x + offset, means, width, label=labels[key], yerr=errors,
                        capsize=5, color=colors[key], alpha=0.8)

        ax.set_ylim(0, 1.07)
        ax.axhline(0.5, linestyle='--', linewidth=1, color='black', alpha=0.5)
        ax.set_title(language_mapping[language] + " FIT-CLAMS", fontsize=25, pad=12)
        ax.tick_params(axis='y', labelsize=12)

    # Shared y-axis label
    fig.text(0.02, 0.57, 'Accuracy', va='center', rotation='vertical', fontsize=27)

    # X-axis: only on the bottom subplot
    xtick_labels = [paradigm_mapping.get(p, p) for p in paradigms]
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(xtick_labels, rotation=60, ha='right', fontsize=29)
    axes[-1].tick_params(axis='x', labelsize=29)

  
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_,
        loc='upper center',
        fontsize=25,
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.52, 1.09)  # slightly higher and well-centered
)

    plt.tight_layout(rect=[0.05, 0.03, 1, 0.96])
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(os.path.join(BARPLOTS_DIR, 'new_benchmark', 'fit_clams_languages_vertical_final.pdf'), bbox_inches='tight')
    plt.show()


# === MAIN SCRIPT === #
if __name__ == "__main__":
    print("Available languages:", ", ".join(AVAILABLE_LANGUAGES))
    language = input("Enter language code (or 'all'): ").strip()
    language_list = AVAILABLE_LANGUAGES if language == 'all' else [language]

    print("\nChoose benchmark type:\n1. Old Style (CLAMS, BLIMP, ZORRO)\n2. New CLAMS Subset")
    benchmark_type = input("Enter 1 or 2: ").strip()

    if benchmark_type == "1":
        print("Available benchmarks:", ", ".join(OLD_STYLE_BENCHMARKS))
        benchmark = input("Enter benchmark name (or 'all'): ").strip()
        for lang in language_list:
            evaluate_old_style(lang, benchmark)
                

    elif benchmark_type == "2":
        if language == "all":
            evaluate_new_clams_all_languages(['en', 'fr', 'de'])
        
        else:
            for lang in language_list:
                evaluate_new_clams(lang)
    else:
        print("Invalid option.")
