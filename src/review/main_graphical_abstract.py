import pandas as pd, numpy as np, matplotlib.pyplot as plt
import seaborn as sns
from pypalettes import load_palette



# hard coded values
data = {
    'Predictor Model': [
        'Baseline clinical model',  # 1st
        'Lesion volume',             # 2nd
        'Lesion location',           # 3rd
        'Radiomics',                 # 4th
        'Brain health',              # 5th
        'Neural network'            # 6th
    ],
    'AUC': [
        0.82,  # Baseline
        0.82,  # Lesion Volume
        0.78,  # Lesion Location
        0.83,  # Radiomics
        0.86,  # Brain Health
        0.85   # Neural Networks
    ],
    'CI_Lower': [
        0.81,  # Baseline
        0.79,  # Lesion Volume
        0.70,  # Lesion Location
        0.77,  # Radiomics
        0.81,  # Brain Health
        0.80   # Neural Networks
    ],
    'CI_Upper': [
        0.83,  # Baseline
        0.85,  # Lesion Volume
        0.84,  # Lesion Location
        0.87,  # Radiomics
        0.90,  # Brain Health
        0.89   # Neural Networks
    ],
}

# keep one row per predictor model for point-and-CI plotting
df = pd.DataFrame(data)

pred_categories = [
    'Anamnesis',
    'Clinical status',
    'Lab tests',
    'Treatment',
    'Routine diagnostics',
    'Lesion volume',
    'Lesion location',
    'Radiomics',
    'Brain health',
    'Neural network'
]

advanced_imaging_categories = [
    'Neural_network',
    'Radiomics',
    'Lesion_volume',
    'Lesion_location',
    'Brain_health',
]

pred_categories_col = [cat.replace(' ', '_') for cat in pred_categories]
cat_to_color = dict(zip(pred_categories, sns.color_palette("tab20")))
cat_to_color['Baseline clinical model'] = (0.5, 0.5, 0.5)  # grey for baseline
cat_to_color = {k: cat_to_color[k] for k in ['Baseline clinical model', 'Lesion volume', 'Lesion location', 'Radiomics', 'Brain health', 'Neural network']}
# %%
import seaborn as sn
colors = ['cornflowerblue'] + load_palette('Althoff')
sns.set_theme(style='white', font='Andale Mono', font_scale=1.0)
sns.set_style('ticks')
plt.figure(figsize=(5, 5))

# Place baseline at the top and preserve explicit plotting order.
plot_df = df.copy()
plot_df['Predictor Model'] = pd.Categorical(
    plot_df['Predictor Model'],
    categories=data['Predictor Model'],
    ordered=True
)
plot_df = plot_df.sort_values('Predictor Model').reset_index(drop=True)

y_pos = np.arange(len(plot_df))
x = plot_df['AUC'].to_numpy()
xerr = np.vstack([
    x - plot_df['CI_Lower'].to_numpy(),
    plot_df['CI_Upper'].to_numpy() - x
])

for i, row in plot_df.iterrows():
    plt.errorbar(
        row['AUC'],
        y_pos[i],
        xerr=[[row['AUC'] - row['CI_Lower']], [row['CI_Upper'] - row['AUC']]],
        fmt='D',
        markersize=9,
        color=colors[i],
        ecolor=colors[i],
        elinewidth=3,
        capsize=0,
        zorder=3,
    )

baseline_x = plot_df.loc[plot_df['Predictor Model'] == 'Baseline clinical model', 'AUC'].iloc[0]
baseline_y = plot_df.index[plot_df['Predictor Model'] == 'Baseline clinical model'][0]
plt.vlines(
    baseline_x,
    ymin=-1,
    ymax=6,
    linestyles='dotted',
    colors='grey',
    linewidth=1.5,
    zorder=1,
)

plt.yticks(y_pos, plot_df['Predictor Model'])
plt.gca().invert_yaxis()
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.xlim(0.68, 0.93)
plt.ylim(5.5, -0.5)
plt.xticks(np.arange(0.7, 0.9, 0.05), fontsize=13)
plt.ylabel('')
plt.xlabel('AUC', fontsize=13)
plt.tight_layout()
plt.savefig('/Users/sebastiano/Desktop/auc.png', dpi=300)
plt.show()