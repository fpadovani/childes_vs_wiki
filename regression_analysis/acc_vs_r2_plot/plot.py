import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np

import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from scipy.stats import pearsonr


def get_ols(df, y_var, x_vars=None):
    if x_vars is None:
        formula = f"{y_var} ~ nsubj1_log_freq + nsubj2_log_freq + verb_log_freq"
    else:
        formula = f"{y_var} ~ {'+'.join(x_vars)}"
    
    model = smf.ols(formula=formula, data=df).fit()
    
    y, X = dmatrices(formula, data=df, return_type='dataframe')
    
    return model


# Matplotlib defaults
palette = sns.color_palette('Paired', n_colors=20)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", palette)

plt.rcParams["lines.solid_capstyle"] = 'projecting'

sns.set_theme(style='whitegrid', font_scale=1.2, palette='tab10')
plt.rcParams["lines.solid_capstyle"] = 'projecting'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rc('hatch', color='1.0', linewidth=.5)
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

plt.style.use('default')
plt.rcParams["lines.solid_capstyle"] = 'projecting'
plt.rcParams['font.family'] = 'Helvetica'


langs = ['eng', 'de', 'fr']
types = ['childes', 'wiki']

print("lang\t", 'model_type', "corpus_type", sep='\t')

seed = "score_30"

seed2acc = ([], [])

for seed in ["score_30", "score_13", "score_42", "avg"]:
    for lang in langs:
        for model_type in types:
            for corpus_type in types:
                if corpus_type != model_type:
                    continue
                df = pd.read_csv(f'{corpus_type}_{lang}.csv')
                df['plural'] = [0,1] * (len(df)//2)
                df['pred'] = df[f'delta_{seed}'] > 0
    
                freqs = np.exp(df.groupby('plural').mean(['verb_log_freq']))
                accs = df.groupby('plural').mean(['pred'])['pred']
                
                ols = get_ols(df, f"delta_{seed}") #res_var)

                seed2acc[0].append(df.pred.mean().round(3))
                seed2acc[1].append(ols.rsquared)


plt.figure(figsize=(3.25,3.25))
ax = plt.gca()

sns.regplot(
    x=seed2acc[1],
    y=seed2acc[0],
    ax=ax,
    scatter_kws={'color': 'white'},
    line_kws={'color': '0.5', 'linewidth': 1, 'linestyle': '--', 'alpha': 0.8},
)

for i in range(len(seed2acc[0])):
    if i < 6:
        label = f"{langs[(i//2)%3].upper()} {types[i%2].title()}"
    else:
        label = None

    if i >= 18:
        plt.scatter(
            seed2acc[1][i],
            seed2acc[0][i],
            color='black',
            s=70,
            marker={0: 'v', 1: 'X', 2: 'o'}[(i//2)%3],
            label=label,
            zorder=99,
        )
    
    plt.scatter(
        seed2acc[1][i],
        seed2acc[0][i],
        color=('orange' if i%2==0 else 'royalblue'),
        marker={0: 'v', 1: 'x', 2: 'o'}[(i//2)%3],
        label=label,
        zorder=100,
        # color=(['r','b']*(len(seed2acc[0])//2)),
    )

plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Corpus')
ax.legend_.get_title().set_fontweight('bold')
ax.legend_.get_title().set_fontsize(11)

plt.xlabel("$R^2~OLS(\Delta \sim \phi)$", fontsize=10, labelpad=15)
plt.ylabel("FIT-CLAMS Accuracy")

plt.xticks([0., .1, .2, .3])
plt.yticks([.7, .8, .9, 1.], ["70%", "80%", "90%", "100%"])
plt.tick_params(labelcolor='0.0', labelsize=9)

plt.ylim(0.65,1.0)
plt.xlim(0., 0.35)

for x, text in zip([.05,.95], ["Relies less on frequency", "Relies more on frequency"]):
    ax.text(
        x, -.1, text, transform=ax.transAxes, color='0.5', ha='center', va='top',
        fontsize=8, fontstyle='italic',
    )

from matplotlib.patches import FancyArrowPatch

arrow = FancyArrowPatch(
    (0.35, -0.125), (0.65, -0.125),
    arrowstyle='<->',
    transform=ax.transAxes,
    color='0.5',
    mutation_scale=15,
    clip_on=False,
    linewidth=1,
)
ax.add_patch(arrow)

plt.grid(True, color='0.7', ls=':', lw=1)

rho, pval = pearsonr(seed2acc[0],seed2acc[1])
ax.text(0.95, 0.95, fr"$r: {rho:.2f}$", transform=ax.transAxes, color='0.5', ha='right', va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

plt.savefig("r2_acc.pdf", bbox_inches='tight')