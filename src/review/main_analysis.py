import os
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_ind
from ast import literal_eval
import plotly.express as px
import textwrap
import re
import logging


from src.review.analysis.meta import derive_se, \
    run_multimodal_meta_regression, \
    country_to_iso3, run_nested_meta_regression, learn_surrogate_and_rank, evaluate_probast_risk
from src.review.preprocessing.basic import count_predictors, from_long_to_wide
from src.review.visualize.plot import plot_recipe, forest_plot_from_df, prepare_forest_df
from src.review.visualize.stich_images import stitch_figures_vertically


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# change the following root path according to where the repo is located
root_dir = '/Users/sebastiano/Desktop/PostDoc/StrokeOutcomeReview/'
data_dir = os.path.join(root_dir, 'data')
fig_dir = os.path.join(root_dir, 'report', 'figures')
clear_fig_dir = True

if clear_fig_dir and os.path.exists(fig_dir):
    logger.warning(f"Clearing figures: {fig_dir}")
    for f in os.listdir(fig_dir):
        os.remove(os.path.join(fig_dir, f))

df = pd.read_excel(os.path.join(data_dir, 'final', 'database_18_06_26.xlsx'))
# these refs must have the letter suffix in the reference section of the paper
ref_check = df.loc[df['first_author_year'].str.endswith('a)'), ['first_author_year', 'Title']].drop_duplicates()
ref_check1 = df.loc[df['first_author_year'].str.endswith('b)'), ['first_author_year', 'Title']].drop_duplicates()
ref_check2 = df.loc[df['first_author_year'].str.endswith('c)'), ['first_author_year', 'Title']].drop_duplicates()

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
cat_to_color['Baseline'] = (0.5, 0.5, 0.5)  # grey for baseline

pred_count, all_predictors = count_predictors(df)
pred_count['category'] = pred_count['category'].str.capitalize().str.replace('_', ' ')

# convert from long to wide format
df = from_long_to_wide(df)
logger.warning(f"Dataframe shape after converting to wide format: {df.shape}")
logger.warning(f"Number of non-nested models: {df[-df['nested']].shape[0]}")

df['external_validation'] = (df['model.validation'] == 'external validation').astype(int)
logger.info(f'Sample stroke type distribution:\n{df.loc[-df['nested'], "sample.stroke_type"].value_counts()}')

# rename some columns for convenience
df = (df.rename(columns={
    "model.performance_value": "auc",
    "model.performance_variance_metric": "var_metric",
    "model.performance_variance_value": "var_value",
    "outcome.n_positive": "n_pos",
    "sample.size": "n_total",
    "first_author_year": "study_id",
    "outcome.cutoff": "outcome.worse_class_cutoff",
}))

# PROBAST risk of bias
df['overall_risk'] = df.apply(evaluate_probast_risk, axis=1)
logger.warning(df[-df['nested']]['overall_risk'].value_counts())
logger.warning(df[-df['nested']]['probast.analysis.risk'].value_counts())
logger.warning(df[-df['nested']]['probast.participants.risk'].value_counts())
logger.warning(df[-df['nested']]['probast.outcome.risk'].value_counts())
logger.warning(df[-df['nested']]['probast.predictors.risk'].value_counts())
probast_table = df.loc[-df['nested'], ['study_id', 'probast.participants.risk',
                                                    'probast.outcome.risk',
                                                    'probast.predictors.risk',
                                                    'probast.analysis.risk',
                                                    'overall_risk']]
# check = probast_table[probast_table['study_id'].duplicated(keep=False)].sort_values(by='study_id', ascending=False)
probast_table.to_excel(os.path.join(root_dir, 'report', 'tables', 'probast.xlsx'), index=False)

risk_order = ['low', 'unclear', 'high']
risk_colors = {
    'low': '#4daf4a',
    'unclear': '#ff7f00',
    'high': '#e41a1c'
}
domain_cols = [
    'probast.participants.risk',
    'probast.outcome.risk',
    'probast.predictors.risk',
    'probast.analysis.risk',
    'overall_risk',
]
domain_name_map = {
    'probast.participants.risk': 'Participants',
    'probast.outcome.risk': 'Outcome',
    'probast.predictors.risk': 'Predictors',
    'probast.analysis.risk': 'Analysis',
    'overall_risk': 'Overall',
}

dom = df.loc[-df['nested'], domain_cols].melt(var_name='domain', value_name='risk')
dom['domain'] = dom['domain'].map(domain_name_map)
dom = dom.groupby(['domain', 'risk']).size().reset_index(name='count')
dom['Percent'] = dom.groupby('domain')['count'].transform(lambda x: (x / x.sum() * 100)).round(1)
domain_order = ['Participants', 'Outcome', 'Predictors', 'Analysis', 'Overall']
pivot = dom.pivot(index='domain', columns='risk', values='Percent').reindex(index=domain_order,
                                                                            columns=risk_order).fillna(0)

ax = pivot.plot(kind='barh', stacked=True, color=[risk_colors[r] for r in risk_order],
                figsize=(9, 4))
plt.rcParams.update({
    'font.size': 14
})
ax.set_xlim(0, 100)
ax.set_xlabel('Percentage (%)')
ax.set_ylabel('PROBAST domain')
ax.legend(title='Risk', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'probast_domain_risk_stacked.png'), dpi=300)
plt.show()
del dom, pivot, ax

# check Treatments and severity
df['EVT'] = df['charms.hyper_acute_treatments'].str.contains('mechanical_thrombectomy_EVT', na=False)
df['IV_alteplase'] = df['charms.hyper_acute_treatments'].str.contains('IV_alteplase', na=False)
df['only_IV_alteplase'] = df['charms.hyper_acute_treatments'] == 'IV_alteplase'
df['post_treatment_models'] = (df['charms.predictors_acquisition_timing'] != 'pre_treatment').astype(int)

# count number of post-treatment models
logger.warning(
    f"Number of post-treatment models: {df.loc[-df['nested'] & (df['sample.stroke_type'] == 'ischemic'), 'post_treatment_models'].sum()}")

# add a column for each category in predictors.category
for category in pred_categories_col:
    df[category] = (df['predictors.category'].str.contains(category.replace('_', ' ').lower())).astype(int)



# ----------------------------------- Prepare data for meta-analysis ---------------------------------

# impute n_pos if var_value is missing
missing_npos_varvalue = df.loc[-df['nested'] & df['var_value'].isna() & df['n_pos'].isna()]
logger.warning(f"Number of models with missing n_pos and var_value: {missing_npos_varvalue.shape[0]}")

df.loc[df['var_value'].isna() & df['n_pos'].isna(), 'n_pos'] = df.loc[df['var_value'].isna() & df[
    'n_pos'].isna(), 'n_total'] * 0.5
df["n_neg"] = df["n_total"] - df["n_pos"]
df["se"] = df.apply(derive_se, axis=1)
df['vi_original'] = df['se'] ** 2
logger.info(f'AUC description:\n{df["auc"].describe()}')
# logit transform for analysis
df["yi"] = np.log((df["auc"]) / (1 - df["auc"]))
# get the variance (approximate SE with the Delta method)
df['se_logit'] = df['se'] / (df['auc'] * (1 - df['auc']))
# variance
df['vi'] = df['se_logit'] ** 2
# clip vi to 0.000001 (one instance of SE = 0, which causes error)
df['vi'] = df['vi'].clip(0.0000001, None)

df['n_total_log'] = np.log(df['n_total'])
df['three_months_outcome'] = (df['outcome.acquisition_time.value'] == 3).astype(int)
df['cutoff'] = np.select([df['outcome.worse_class_cutoff'] == 3,
                          df['outcome.worse_class_cutoff'] < 3,
                          df['outcome.worse_class_cutoff'] > 3],
                         [0, -1, 1])

# impute sample.nih_stroke_scale.value with median
median_nihss = df.loc[
    -df['nested'] & (df['sample.stroke_type'] == 'ischemic'), 'sample.nih_stroke_scale.value'].median()
df.loc[df['sample.nih_stroke_scale.value'].isna() & (
        df['sample.stroke_type'] == 'ischemic'), 'sample.nih_stroke_scale.value'] = median_nihss
logger.warning(f"Imputed median NIHSS {median_nihss}")
# separate into mild, moderate, severe
df['severity'] = pd.cut(df['sample.nih_stroke_scale.value'], bins=[-1, 4, 15, 42],
                        labels=['mild', 'moderate', 'severe'])
# count number of studies per severity
logger.warning(f"Severity distribution:\n{df.loc[-df['nested'], ['severity']].value_counts()}")

general = []
for stroke_type, group in df.loc[-df['nested']].groupby('sample.stroke_type'):
    mean_age = group['sample.age.value'].mean()
    females = group['sample.females.value'].mean()
    mean_n = group['n_total'].mean()
    mean_nihss = group['sample.nih_stroke_scale.value'].mean()
    # count IVT and EVT
    evt = group['EVT'].mean() * 100
    ivt = group['IV_alteplase'].mean() * 100
    ext_val = group['external_validation'].mean() * 100
    post_treat = group['post_treatment_models'].mean() * 100
    model_name = group['model.name'].value_counts(normalize=True).to_dict()
    # count per severity
    severity_counts = group['severity'].value_counts(normalize=True).to_dict()
    logger.warning(
        f"Stroke type: {stroke_type} | Mean age: {mean_age:.1f} | Females (%): {females:.1f} | Mean sample size: {mean_n:.1f} | Mean NIHSS: {mean_nihss:.1f} | Severity distribution: {severity_counts}")
    logger.warning(
        f"Stroke type: {stroke_type} | IV alteplase: {ivt} | EVT: {evt} | External validation: {ext_val} | Post-treatment models: {post_treat} | Model names: {model_name}")
    general.append({
        'Stroke type': stroke_type,
        'Age (M)': f"{mean_age:.1f}",
        'Females': f"{females:.1f}%",
        'Sample size (M)': f"{mean_n:.1f}",
        'NIHSS (M)': f"{mean_nihss:.1f}",
        'Severity': f'{severity_counts.get("mild", 0) * 100:.0f}% mild; {severity_counts.get("moderate", 0) * 100:.0f}% moderate; {severity_counts.get("severe", 0) * 100:.0f}% severe',
        'IVT': f'{ivt:.0f}%',
        'EVT': f'{evt:.0f}%',
        'External validation': f'{ext_val:.0f}%',
        'Post-treatment models': f'{post_treat:.0f}%',
        'Model type': f'{model_name[list(model_name.keys())[0]] * 100:.0f}% {list(model_name.keys())[0]}; {model_name[list(model_name.keys())[1]] * 100:.0f}% {list(model_name.keys())[1]}'
    })
general = pd.DataFrame(general).round(0)
# save as excel
general.to_excel(os.path.join(root_dir, 'report', 'tables', 'general_study_characteristics.xlsx'), index=False)

# count cutoff distribution
cutoff_counts = df.loc[-df['nested'], 'outcome.worse_class_cutoff'].value_counts().sort_index()
logger.warning(f"Cutoff distribution:\n{cutoff_counts}")
# again with percentages
cutoff_perc = df.loc[-df['nested'], 'outcome.worse_class_cutoff'].value_counts(normalize=True).sort_index() * 100
logger.warning(f"Cutoff distribution (%):\n{cutoff_perc.round(1)}")
# count number of studies with 3 months outcome
three_months_counts = df.loc[-df['nested'], 'three_months_outcome'].value_counts()
logger.warning(f"Three months outcome distribution:\n{three_months_counts}")
# again with percentages
three_months_perc = df.loc[-df['nested'], 'three_months_outcome'].value_counts(normalize=True) * 100
logger.warning(f"Three months outcome distribution (%):\n{three_months_perc.round(1)}")



# ----------------------------------- Biases analyses -----------------------------------
# plot auc vs sample size, log scale
df_best_models = df[-df['nested']].copy()
# do correlation among not externally validated models
corr_df = df_best_models.loc[
    df_best_models['external_validation'] == 0, ['auc', 'n_total', 'n_total_log']].dropna()
corr_value, spearman_p = spearmanr(corr_df['auc'], corr_df['n_total_log'])
# same correlation with external validation models
corr_ev_df = df_best_models.loc[
    df_best_models['external_validation'] == 1, ['auc', 'n_total', 'n_total_log']].dropna()
corr_value_ev, spearman_p_ev = spearmanr(corr_ev_df['auc'], corr_ev_df['n_total_log'])
logger.warning(f"Pearson correlation (no external validation): r={corr_value}, p={spearman_p}")
logger.warning(f"Pearson correlation (external validation): r={corr_value_ev}, p={spearman_p_ev}")
# compute the difference in auc between externally validated and not externally validated models
df_best_models_ev = df_best_models[['study_id', 'auc', 'n_total', 'external_validation']]
# do a two sample t-test
ttest_res = ttest_ind(df_best_models_ev.loc[df_best_models_ev['external_validation'] == 1, 'auc'],
                      df_best_models_ev.loc[df_best_models_ev['external_validation'] == 0, 'auc'],
                      equal_var=False)
logger.warning(
    f"T-test between externally validated and not externally validated models: t={ttest_res.statistic}, p={ttest_res.pvalue}")
# compute the mean auc for externally validated and not externally validated models and the difference
mean_auc_ev = df_best_models_ev.loc[df_best_models_ev['external_validation'] == 1, 'auc'].mean()
mean_auc_no_ev = df_best_models_ev.loc[df_best_models_ev['external_validation'] == 0, 'auc'].mean()
diff_auc = mean_auc_ev - mean_auc_no_ev
logger.warning(
    f"Mean AUC (external validation): {mean_auc_ev}, Mean AUC (no external validation): {mean_auc_no_ev}, Difference: {diff_auc}")

df_best_models['External validation'] = df_best_models['external_validation'].map({
    1: 'yes',
    0: 'no'
})
plt.figure(figsize=(12, 6))
plt.rcParams.update({
    'font.size': 16
})
sns.regplot(data=df_best_models[df_best_models['external_validation'] == 0], x='n_total', y='auc', logx=True,
            scatter=False,
            color='red')
sns.regplot(data=df_best_models[df_best_models['external_validation'] == 1], x='n_total', y='auc', logx=True,
            scatter=False)
sns.scatterplot(data=df_best_models, x='n_total', y='auc', hue='External validation', alpha=0.9, palette={
    'yes': 'blue',
    'no': 'red'
}, s=100)
plt.ylim(0.5, 1)
plt.xscale('log')
plt.xlabel('Sample size (log scale)')
plt.ylabel('AUC')
plt.text(plt.xlim()[1] * 0.15, plt.ylim()[1] - 0.055, f'rho={corr_value:.2f}, p={spearman_p:.3f}', fontsize=14, color='red')
plt.text(plt.xlim()[1] * 0.15, plt.ylim()[1] - 0.025, f'rho={corr_value_ev:.2f}, p={spearman_p_ev:.3f}', fontsize=14, color='blue')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='External validation')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'auc_vs_sample_size.png'), dpi=300)
plt.show()
del ttest_res, mean_auc_ev, mean_auc_no_ev, diff_auc, corr_df, corr_ev_df

# add all scales as boolean columns
df = pd.concat([df, pd.DataFrame(
    [df['predictors.scale'].str.contains(re.escape(s), na=False, regex=True).astype(int) for s in all_predictors],
    index=all_predictors).T, ], axis=1)


counts = []
for category in ['Baseline'] + advanced_imaging_categories:
    if category == 'Baseline':
        subset = df[-df['nested'] & ~(df[advanced_imaging_categories].sum(axis=1) >= 1)]
    else:
        subset = df[-df['nested'] & (df[category] == 1)]
    n_models = subset.shape[0]
    print(f'{category}: {n_models} models')
    for cat_x in pred_categories_col:
        n_cat = subset[cat_x].mean() * 100
        counts.append({
            'category': category.replace('_', ' '),
            'predictor_category': cat_x.replace('_', ' '),
            'n_models': n_models,
            'Percentage (%)': n_cat
        })
counts = pd.DataFrame(counts)
# increase font size
plt.rcParams.update({
    'font.size': 24
})
sns.catplot(data=counts, x='predictor_category', y='Percentage (%)', hue='category', palette=cat_to_color,
            row='category', kind='bar', height=3.5, aspect=4, sharex=True, legend=False,
            row_order=['Baseline', 'Lesion volume', 'Lesion location', 'Radiomics', 'Brain health', 'Neural network', ])
for ax in plt.gcf().axes:
    ax.set_title('')
    ax.set_ylabel('')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f%%', padding=3, fontsize=20)

plt.xticks(rotation=45, ha='right')
plt.xlabel('')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'predictor_categories_per_model_category.png'), dpi=300)
plt.show()
del counts

pred_count_subsets = []
for cat in ['Baseline'] + advanced_imaging_categories:  #  + advanced_imaging_categories
    if cat == 'Baseline':
        subset = df[-df['nested'] & ~(df[advanced_imaging_categories].sum(axis=1) >= 1)]
    else:
        subset = df[-df['nested'] & (df[cat] == 1)]

    all_predictors_subset = subset[all_predictors].mean().sort_values(ascending=False)
    all_predictors_subset = all_predictors_subset[all_predictors_subset >= 0.2].index.tolist()
    pred_count_subset = subset[all_predictors_subset].mean().reset_index()
    pred_count_subset.columns = ['scale', 'count']
    pred_count_subset['count'] = pred_count_subset['count'] * 100
    pred_count_subset['category'] = pred_count_subset['scale'].map(pred_count[['scale', 'category']].set_index('scale')['category'])
    pred_count_subset['Model'] = cat.replace('_', ' ')
    pred_count_subsets.append(pred_count_subset)

pred_count_subset = pd.concat(pred_count_subsets, ignore_index=True)
pred_count_subset['category'] = pd.Categorical(pred_count_subset['category'],
                                              categories=pred_categories,
                                              ordered=True)
# rename count tot Percentage (%)
pred_count_subset = pred_count_subset.rename(columns={'count': 'Percentage (%)'})

baseline = pred_count_subset[pred_count_subset['Model'] == 'Baseline'].reset_index(drop=True)
# order by Category and then by Percentage (%)
scale_order = baseline.sort_values(by=['category', 'Percentage (%)'], ascending=[True, False])[['category', 'scale']]
# to dict: category: [scales]
scale_order = scale_order.groupby('category')['scale'].apply(list).to_dict()


# add to scale_order the predictors that are not in baseline but are in other categories
for cat in ['Lesion volume', 'Lesion location', 'Radiomics', 'Brain health', 'Neural network']:
    temp = pred_count_subset.loc[(pred_count_subset['Model'] == cat), ['category', 'scale']]
    # add only those not in scale_order
    for _, row in temp.iterrows():
        if row['scale'] not in sum(scale_order.values(), []):
            scale_order[row['category']].append(row['scale'])

# flatten scale_order preserving order
scale_order = [scale for scales in scale_order.values() for scale in scales]

# bar plot with separate column for each category of models
plt.rcParams.update({
    'font.size': 24
})
# lets change the theme
sns.set_style('whitegrid')
sns.catplot(data=pred_count_subset, x='Percentage (%)', y='scale', hue='Model', palette=cat_to_color, col='Model',
            sharey=True, sharex=True,legend=False, order=scale_order,
            col_order=['Baseline', 'Lesion volume', 'Lesion location', 'Radiomics', 'Brain health', 'Neural network', ],
            kind='bar', height=23, aspect=0.2)

# iterate over each column to set title and labels
for i, ax in enumerate(plt.gcf().axes):
    if i == 0:
        ax.set_ylabel('Predictor', fontsize=28)
    model_cat = ax.get_title().split(' = ')[1]
    ax.set_title(f'{model_cat}', fontsize=34)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.axvline(0, color='black', linewidth=2)

plt.savefig(os.path.join(fig_dir, f'predictors_per_model_category.png'))
plt.show()


# print mean auc pre- and post-treatment models
mean_auc_pre = df.loc[
    -df['nested'] & (df['sample.stroke_type'] == 'ischemic') & (df['post_treatment_models'] == 0), 'auc'].mean()
mean_auc_post = df.loc[
    -df['nested'] & (df['sample.stroke_type'] == 'ischemic') & (df['post_treatment_models'] == 1), 'auc'].mean()
logger.warning(f"Mean AUC (pre-treatment models, ischemic stroke): {mean_auc_pre:.2f}")
logger.warning(f"Mean AUC (post-treatment models, ischemic stroke): {mean_auc_post:.2f}")

# mean n_total of neuroimaging models vs non-neuroimaging models
mean_n_neuro = df.loc[
    -df['nested'] & (df['sample.stroke_type'] == 'ischemic') & (df['Neural_network'] == 1) | (df['Radiomics'] == 1) | (
            df['Lesion_volume'] == 1) | (df['Lesion_location'] == 1) | (df['Brain_health'] == 1), 'n_total'].mean()
mean_n_non_neuro = df.loc[
    -df['nested'] & (df['sample.stroke_type'] == 'ischemic') & (df['Neural_network'] == 0) & (df['Radiomics'] == 0) & (
            df['Lesion_volume'] == 0) & (df['Lesion_location'] == 0) & (df['Brain_health'] == 0), 'n_total'].mean()
logger.warning(f"Mean sample size (neuroimaging models, ischemic stroke): {mean_n_neuro:.1f}")
logger.warning(f"Mean sample size (non-neuroimaging models, ischemic stroke): {mean_n_non_neuro:.1f}")

del mean_n_neuro, mean_n_non_neuro

# count females
logger.warning(df.loc[-df['nested'], 'sample.females.value'].mean())


# -----------------------------------Multimodal-between-study analysis: category -----------------------------------
results = run_multimodal_meta_regression(df,
                                         advanced_imaging_categories=advanced_imaging_categories,
                                         stroke_type='ischemic',
                                         covariates=['external_validation', 'post_treatment_models'],)
logger.warning(
    f"Covariate:\n{results[results['name'].isin(['external_validation', 'post_treatment_models'])][['stroke_type', 'name', 'category', 'delta', 'p-value']].round(3)}")
# drop post_treatment_models
results = results[~results['name'].isin(['post_treatment_models', 'external_validation'])].reset_index(drop=True)
results = prepare_forest_df(results, df, sensitivity=False, adv_pred_width=20)
# rename Validation into val
results = results.rename(columns={
    'Validated': 'Val.',
    'Post-treatment': 'Post',
    'Imaging feature': 'Imaging feat.',
})
plot_params = {
    'Lesion_volume': {
        'xlim': (0.5, 1),
        'figsize': (16, 24),
        'panel_width_ratios': [2.7, 0.8, 0.5],
        'text_col_widths': [0.25, 0.06, 0.06, 0.12, 0.16, 0.46],
    },
    'Lesion_location': {
        'xlim': (0.5, 1),
        'figsize': (16, 6.5),
        'panel_width_ratios': [2.8, 0.4, 0.5],
        'text_col_widths': [0.24, 0.06, 0.06, 0.11, 0.29, 0.4],
    },
    'Radiomics': {
        'xlim': (0.5, 1),
        'figsize': (16, 10.5),
        'panel_width_ratios': [2.1, 0.4, 0.4],
        'text_col_widths': [0.17, 0.06, 0.06, 0.1, 0.18, 0.35],
    },
    'Brain_health': {
        'xlim': (0.5, 1),
        'figsize': (16, 9),
        'panel_width_ratios': [2.3, 0.5, 0.4],
        'text_col_widths': [0.17, 0.06, 0.06, 0.1, 0.15, 0.3],
    },
    'Neural_network': {
        'xlim': (0.5, 1),
        'figsize': (16, 9),
        'panel_width_ratios': [2.5, 0.4, 0.4],
        'text_col_widths': [0.18, 0.06, 0.06, 0.11, 0.22, 0.36],
    },
}
for cat in ['Lesion_volume', 'Lesion_location', 'Radiomics', 'Brain_health', 'Neural_network']:

    results_subset = pd.concat([results.iloc[:1, :],
                                results[results['Study'] == (cat.replace('_', ' '))],
                                results[(results['name'] == cat)]], axis=0).reset_index(drop=True)
    fig, axes = forest_plot_from_df(
        results_subset,
        effect_col="estimate",
        ci_low_col="ci_0.025",
        ci_high_col="ci_0.975",
        label_col="Study",
        row_type_col="row_type",
        extra_cols=['Val.', 'Post', 'Severity', 'Imaging feat.', 'Other features'],
        show_vertical_lines=True,
        fontsize=12,
        header_fontsize=12,
        header_top_pad=0.9,
        show_study_separators=True,
        xticks=[0.6, 0.7, 0.8, 0.9, 1.0],
        xticks_labels=['0.6', '0.7', '0.8', '0.9', '1.0'],
        text_pad=0.02,
        row_height=0.85,  # controls vertical spacing
        group_gap=0.9,  # extra gap after each subgroup pooled row
        bold_group_summary=False,
        color_by_col='row_type_',
        color_map={
            'baseline': 'crimson',
            'group_summary': 'cornflowerblue',
            'study': 'mediumseagreen',
        },
        vertical_line_color={
            0: 'crimson'
        },
        **plot_params.get(cat, {})
    )
    plt.savefig(os.path.join(fig_dir, f'forest_{cat}.png'), dpi=300)
    plt.show()

stitch_figures_vertically([os.path.join(fig_dir, f'forest_{cat}.png') for cat in ['Lesion_volume', 'Lesion_location', 'Radiomics', 'Brain_health', 'Neural_network']],
                          output_path=os.path.join(fig_dir, f'forest_stiched.png'),
                          panels_output_dir=fig_dir,
                          label_size=100)

# represent only Baseline models
subset = df[-df['nested']
            & (df['Clinical_status'] == 1)
            & (df[advanced_imaging_categories].sum(axis=1) == 0)
            & (df['sample.stroke_type'] == 'ischemic')]

subset = subset.sort_values(by='auc', ascending=True).reset_index(drop=True)
add_single = []
add_single.append({
    'Study': 'Baseline',
    'row_type': 'group_header',
    'Validated': '',
    'Post-treatment': '',
    'Severity': '',
})
for _, row in subset.iterrows():
    auc = row['auc']
    se = row['se']
    ci_lower = auc - 1.96 * se
    ci_higher = np.clip(auc + 1.96 * se, 0, 1)
    row_cat = np.array(row['predictors.category'].split(', '))
    row_pred = np.array(row['predictors.scale'].split(', '))
    print(f'Study ID: {row["study_id"]} predictors scales: {row_pred}\n')
    other_pred = ', '.join(np.unique(row_pred).tolist())
    other_pred = '\n'.join(textwrap.wrap(other_pred, width=60))
    add_single.append({
        'Study': row['study_id'],
        'name': 'Baseline',
        'category': 'Baseline',
        'estimate': auc,
        'ci_0.025': ci_lower,
        'ci_0.975': ci_higher,
        'Severity': row['severity'],
        'Val.': 'yes' if row['external_validation'] == 1 else 'no',
        'Post': 'yes' if row['post_treatment_models'] == 1 else 'no',
        'row_type': 'study',
        'Features': other_pred,
    })

results = pd.DataFrame(add_single)
del add_single, subset
# split the plot into two parts
for subset in [0, 1]:
    fig, axes = forest_plot_from_df(
            results.iloc[:75] if subset == 0 else pd.concat([results.iloc[:1], results.iloc[75:]]),
            effect_col="estimate",
            ci_low_col="ci_0.025",
            ci_high_col="ci_0.975",
            label_col="Study",
            row_type_col="row_type",
            extra_cols=['Val.', 'Post', 'Severity', 'Features'],
            show_vertical_lines=True,
            fontsize=12,
            figsize=(16, 35),
            text_col_widths=[0.18, 0.05, 0.05, 0.08, 0.35],
            panel_width_ratios=[2.6, 0.5, 0.4],
            xlim=(0.45, 1),
            header_fontsize=12,
            header_top_pad=0.9,
            show_study_separators=True,
            text_pad=0.02,
            row_height=0.85,  # controls vertical spacing
            group_gap=0.9,  # extra gap after each subgroup pooled row
            bold_group_summary=False,
            color_by_col='row_type_',
            color_map={
                'baseline': 'crimson',
                'group_summary': 'cornflowerblue',
                'study': 'mediumseagreen',
            },
            vertical_line_color={
                0: 'crimson'
            },
    )
    plt.savefig(os.path.join(fig_dir, f'forest_baseline_subset{subset}.png'), dpi=300)
    plt.show()


# ------------------------------------- Sensitivity analysis ----------------------------------

# sensitivity analysis limited to studies on 0-2 vs 3-6 mRs at 3 months
results = run_multimodal_meta_regression(df[(df['three_months_outcome'] == 1) & (df['outcome.worse_class_cutoff'] == 3)],
                                         advanced_imaging_categories=advanced_imaging_categories,
                                         stroke_type='ischemic',
                                         covariates=['external_validation', 'post_treatment_models'],)
print(
    f"Covariate:\n{results[results['name'].isin(['external_validation', 'post_treatment_models'])][['stroke_type', 'name', 'category', 'delta', 'p-value']].round(3)}")
# drop covariates
results = results[~results['name'].isin(['post_treatment_models', 'external_validation'])].reset_index(drop=True)
results = prepare_forest_df(results, df, sensitivity=True)
results = results.rename(columns={
    'Validated': 'Val.',
    'Post-treatment': 'Post',
    'Imaging feature': 'Imaging feat.',
})
plot_params = {
    'Lesion_volume': {
        'xlim': (0.5, 1),
        'figsize': (16, 18),
        'panel_width_ratios': [2.7, 0.8, 0.5],
        'text_col_widths': [0.25, 0.06, 0.06, 0.12, 0.16, 0.46],
    },
    'Lesion_location': {
        'xlim': (0.5, 1),
        'figsize': (16, 5),
        'panel_width_ratios': [2.8, 0.4, 0.5],
        'text_col_widths': [0.24, 0.06, 0.06, 0.11, 0.29, 0.4],
    },
    'Radiomics': {
        'xlim': (0.5, 1),
        'figsize': (16, 8),
        'panel_width_ratios': [2.1, 0.4, 0.4],
        'text_col_widths': [0.17, 0.06, 0.06, 0.1, 0.18, 0.35],
    },
    'Brain_health': {
        'xlim': (0.5, 1),
        'figsize': (16, 6),
        'panel_width_ratios': [2.3, 0.5, 0.4],
        'text_col_widths': [0.17, 0.06, 0.06, 0.1, 0.15, 0.3],
    },
    'Neural_network': {
        'xlim': (0.5, 1),
        'figsize': (16, 7),
        'panel_width_ratios': [2.5, 0.4, 0.4],
        'text_col_widths': [0.18, 0.06, 0.06, 0.11, 0.22, 0.36],
    },
}
for cat in ['Lesion_volume', 'Lesion_location', 'Radiomics', 'Brain_health', 'Neural_network']:
    results_subset = pd.concat([results.iloc[:1, :],
                                results[results['Study'] == (cat.replace('_', ' '))],
                                results[(results['name'] == cat)]], axis=0).reset_index(drop=True)
    fig, axes = forest_plot_from_df(
        results_subset,
        effect_col="estimate",
        ci_low_col="ci_0.025",
        ci_high_col="ci_0.975",
        label_col="Study",
        row_type_col="row_type",
        extra_cols=['Val.', 'Post', 'Severity', 'Imaging feat.', 'Other features'],
        show_vertical_lines=True,
        fontsize=9,
        header_fontsize=10,
        header_top_pad=0.9,
        show_study_separators=True,
        text_pad=0.02,
        row_height=0.85,  # controls vertical spacing
        group_gap=0.9,  # extra gap after each subgroup pooled row
        bold_group_summary=False,
        color_by_col='row_type_',
        color_map={
            'baseline': 'crimson',
            'group_summary': 'cornflowerblue',
            'study': 'mediumseagreen',
        },
        vertical_line_color={
            0: 'crimson'
        },
        **plot_params.get(cat, {})
    )
    plt.savefig(os.path.join(fig_dir, f'forest_{cat}_subset.png'), dpi=300)
    plt.show()

# stich
stitch_figures_vertically([os.path.join(fig_dir, f'forest_{cat}_subset.png') for cat in ['Lesion_volume', 'Lesion_location', 'Radiomics', 'Brain_health', 'Neural_network']],
                          output_path=os.path.join(fig_dir, f'forest_stiched_subset.png'),
                          panels_output_dir=fig_dir,
                          panel_prefix='forest_subset_panel_',
                          label_size=100)
# sensitivity analysis separately on severe, moderate, mild ischemic stroke
results = []
for severity in ['mild', 'moderate', 'severe']:
    subset = df[(df['severity'] == severity) & (-df['nested'])]
    print(len(subset))
    result = run_multimodal_meta_regression(subset,
                                            advanced_imaging_categories=advanced_imaging_categories,
                                            stroke_type='ischemic',
                                            covariates=['external_validation', 'post_treatment_models'])
    result['Severity'] = severity
    results.append(result)
results = pd.concat(results).reset_index(drop=True)

print(f"Covariate p-values:\n{results[results['name'].isin(['external_validation', 'post_treatment_models'])][['Severity', 'name', 'category', 'delta', 'p-value']].round(3)}")
# drop covariates
results = results[~results['name'].isin(['post_treatment_models', 'external_validation'])].reset_index(drop=True)
# give category a specific order
results['category'] = pd.Categorical(results['category'], categories=[
    'Clinical_status',
    'Lesion_volume',
    'Lesion_location',
    'Radiomics',
    'Brain_health',
    'Neural_network',
], ordered=True)
results = results.sort_values(['category', 'Severity'], ascending=[True, True])
results['row_type'] = 'study'
results['Study'] = results['category'].str.replace('_', ' ').replace({'Clinical status': 'Baseline'})
results['Delta AUC'] = results.apply(lambda x: f'{x["delta"]:.3f} ({x["delta_ci_0.025"]:.3f}-{x["delta_ci_0.975"]:.3f}) p={x["p-value"]:.3f}', axis=1)
results['Heterogeneity'] = results.apply(lambda x: f'I² {x["I2"]:.1f}%, τ² {x["tau2"]:.2f} p={x["p_het"]:.3f}', axis=1)
results['N'] = results['count'].astype(int).astype(str)

add = []
for cat in ['mild', 'moderate', 'severe']:
    header = pd.DataFrame({'row_type': 'group_header', 'Study': [cat]})
    add.append(header)
    add.append(results[results['Severity'] == cat])
results = pd.concat(add).reset_index(drop=True)
# replace nan in N studies, Delta AUC, Heterogeneity with empty string
results['N'] = results['N'].fillna('')
results['Delta AUC'] = results['Delta AUC'].fillna('')
results['Heterogeneity'] = results['Heterogeneity'].fillna('')
results.loc[results['Study'] == 'Baseline', ['Delta AUC', 'Heterogeneity']] = ''
results = results.rename({'Study': 'Category'}, axis=1)
del add

fig, axes = forest_plot_from_df(
    results,
    effect_col="estimate",
    ci_low_col="ci_0.025",
    ci_high_col="ci_0.975",
    label_col="Category",
    row_type_col="row_type",
    extra_cols=['N', 'Delta AUC', 'Heterogeneity'],
    panel_width_ratios=[2.3, 1.0, 0.5],
    text_col_widths=[0.15, 0.05, 0.25, 0.2],
    show_vertical_lines=True,
    show_study_separators=True,
    fontsize=12,
    header_fontsize=13,
    header_top_pad=1,
    text_pad=0.02,
    figsize=(14, 7),
    row_height=0.85,  # controls vertical spacing
    group_gap=0.9,  # extra gap after each subgroup pooled row
    bold_group_summary=False,
    xlim=(0.55, 1),
    all_diamonds=True,
    color_by_col='Severity',
    color_map={
    'mild': '#4daf4a',
    'moderate': '#ff7f00',
    'severe': '#377eb8',
},
)
plt.savefig(os.path.join(fig_dir, f'forest_severity.png'), dpi=300)
plt.show()


# ------------------------------ Within study analyses (nested models) ------------------------------
results, add = run_nested_meta_regression(df, advanced_imaging_categories=advanced_imaging_categories)
add = add.explode(['Key', 'studies', 'auc_delta', 'var_delta']).reset_index(drop=True)

add_single = []
for cat in add['category'].unique():
    add_single.append({
        'Study': cat.replace('_', ' '),
        'row_type': 'group_header',
    })
    subset = add[add['category'] == cat].sort_values(by='auc_delta', ascending=True)
    for _, row in subset.iterrows():
        auc_delta = row['auc_delta']
        var_delta = row['var_delta']
        se_delta = np.sqrt(var_delta)
        ci_lower = auc_delta - 1.96 * se_delta
        ci_higher = auc_delta + 1.96 * se_delta
        row['predictors.category'] = df.loc[-df['nested'] & (df['Key'] == row['Key']), 'predictors.category'].values[0]
        row['predictors.scale'] = df.loc[-df['nested'] & (df['Key'] == row['Key']), 'predictors.scale'].values[0]
        row_cat = np.array(row['predictors.category'].split(', '))
        row_pred = np.array(row['predictors.scale'].split(', '))
        adv_pred = ', '.join(np.unique(row_pred[row_cat == cat.replace('_', ' ').lower()]).tolist())
        adv_pred = adv_pred.replace('white matter hyperintensities', 'WMH')

        if row['studies'] == 'Johnston et al (2009)':
            adv_pred = 'infarct volume'
        if row['studies'] == 'Oliveira et al (2023)':
            adv_pred = 'CTA raw image'
        other_pred = ', '.join(np.unique(row_pred[row_cat != cat.replace('_', ' ').lower()]).tolist())
        adv_pred = '\n'.join(textwrap.wrap(adv_pred, width=30))
        other_pred = '\n'.join(textwrap.wrap(other_pred, width=45))

        add_single.append({
            'Study': row['studies'],
            'name': cat,
            'category': cat,
            'estimate': auc_delta,
            'ci_0.025': ci_lower,
            'ci_0.975': ci_higher,
            'Added imaging': adv_pred,
            'Nested model': other_pred,
            'row_type': 'study',
        })
    # add group summary
    pooled = results.loc[results['category'] == cat, :].iloc[0].to_dict()
    mdl_summary = f'RE model - pooled ΔAUC p={pooled["p-value"]:.3f}'
    mdl_het = f'I² {pooled["I2"]:.1f}%, τ² {pooled["tau2"]:.3f} p={pooled["p_het"]:.3f}'
    add_single.append({
        'Study': mdl_summary,
        'row_type': 'group_summary',
        **pooled
    })
    pooled['estimate'] = np.nan
    pooled['ci_0.025'] = np.nan
    pooled['ci_975'] = np.nan
    add_single.append({
        'Study': mdl_het,
        'row_type': 'group_summary',
        **pooled
    })
results = pd.DataFrame(add_single)
del add_single, pooled, subset, row, se_delta, var_delta, auc_delta, ci_higher, ci_lower

# plot forest plot
fig, axes = forest_plot_from_df(
    results,
    effect_col="estimate",
    ci_low_col="ci_0.025",
    ci_high_col="ci_0.975",
    label_col="Study",
    row_type_col="row_type",
    extra_cols=['Nested model','Added imaging'],
    panel_width_ratios=[3, 1.0, 0.6],
    text_col_widths=[0.22, 0.35, 0.25],
    show_vertical_lines=True,
    show_study_separators=True,
    fontsize=12,
    header_fontsize=13,
    header_top_pad=1,
    text_pad=0.02,
    figsize=(16, 15),
    row_height=0.85,  # controls vertical spacing
    group_gap=0.9,  # extra gap after each subgroup pooled row
    bold_group_summary=False,
    xlim=None,
    xticks=[-0.05, 0, 0.05, 0.1],
    vertical_line_color={
        0: 'crimson'
    },
    color_by_col='row_type',
    color_map={
        'group_summary': 'cornflowerblue',
        'study': 'mediumseagreen',
    },
)
plt.savefig(os.path.join(fig_dir, f'forest_nested.png'), dpi=300)
plt.show()


# %%
# ------------------------------ Prediction analysis ------------------------------

predictor_counts = df.loc[-df['nested'] & (df['sample.stroke_type'] == 'ischemic'), all_predictors].sum().reset_index(
    name='count')
predictor_counts = predictor_counts[predictor_counts['count'] >= 5].reset_index(drop=True)
all_predictors_subset = predictor_counts['index'].tolist() + advanced_imaging_categories
print(f'N {len(all_predictors_subset)} predictors: {all_predictors_subset}')

subset = df.loc[(df['sample.stroke_type'] == 'ischemic'), all_predictors_subset +
                                                          ['auc', 'se', 'Key', 'external_validation',
                                                           'post_treatment_models', 'severity', 'sample.stroke_type',
                                                           'sample.nih_stroke_scale.value', 'n_total', 'nested']]
print(f'Subset size: {subset.shape[0]} models')
n_total_avg = subset.loc[-subset['nested'], 'n_total'].median()
logger.warning(f'Average sample size: {n_total_avg}')
result = learn_surrogate_and_rank(
    X_bin=subset[all_predictors_subset],
    auc=subset['auc'],
    se_auc=None,
    required=None,
    min_models=2,
    candidates=None,
    exact_match=False,
    penalty_lambda=0.05,
    max_size=None,
    study_ids=subset['Key'],
    n_outer_splits=10,
    n_inner_splits=10,
    random_state=12,
    model="svr",
    param_grid=None,
    nested_cv=True,
    covariates=subset[['external_validation', 'post_treatment_models', 'severity', 'n_total', 'nested']],
    covariate_values={
        'external_validation': [0, 1],
        'post_treatment_models': [0, 1],
        'severity': ['mild', 'moderate', 'severe'],
        'n_total': [n_total_avg],
        'nested': [0]
    },
)

print(result['final_model'])
print(result["outer_cv"]['pooled_metrics'])
print(round(result["outer_cv"]['pooled_metrics']['r2'], 2))
cv = pd.DataFrame(result["outer_cv"]["folds"])
print(cv[['best_params', 'mae', 'r2']])


for val in [0, 1]:
    ranked_df = []
    for scen in result["scenarios"]:
        print("Scenario:", scen["values"])
        scen_df = pd.DataFrame(scen["ranked"])
        for k, v in scen["values"].items():
            scen_df[k] = v
        ranked_df.append(scen_df)
    ranked_df = pd.concat(ranked_df).reset_index(drop=True)
    ranked_df = ranked_df[(ranked_df['support_n'] >= 4) & (ranked_df['n_total'] == n_total_avg) & (
            ranked_df['external_validation'] == val)].reset_index(drop=True)
    # remove combos not possible in post-treatment models
    ranked_df = ranked_df[~((ranked_df['post_treatment_models'] == 0) & (
        ranked_df['combo'].str.contains(r'recanalization status|thrombolysis|NIHSS post treatment|(?<!premorbid\s)mRS\b', na=False,
                                        regex=True)))].reset_index(drop=True)
    ranked_df['post_treatment_models'] = ranked_df['post_treatment_models'].replace({
        0: 'pre',
        1: 'post'
    })
    ranked_df['top'] = ranked_df.groupby(['severity', 'post_treatment_models'])['pred_auc'].transform(
        lambda x: x == x.max())  # x > (x.max() - 0.01)
    # check whether each predictor is present in combo
    for pred in all_predictors_subset:
        if any(pred in c for c in ranked_df['combo']):
            ranked_df[pred] = ranked_df['combo'].str.contains(re.escape(pred)).astype(int)

    best_combos = ranked_df[ranked_df['top']].reset_index(drop=True)
    # count the number of combos per severity and post_treatment_models
    print(best_combos.groupby(['severity', 'post_treatment_models']).size())
    predictors = best_combos.loc[:, 'top':].columns[1:].tolist()
    plot_recipe(best_combos, predictors, fig_dir,
                validation=val,
                figsize=(15, 12),
                constant_circle_size=700,
                scale_circle_size=False,
                feature_name_map={
                    'NIHSS': 'NIHSS admission',
                    'NIHSS post treatment': 'NIHSS post revascularization',
                    'mRS': 'discharge mRS',
                    'glucose': 'admission glucose',
                    'hemoglobin': 'admission hemoglobin',
                })


# %%
# ------------------------------ Geographical analysis ------------------------------

countries_df = df.loc[-df['nested'], ['study_id', 'Key', 'model_id', 'n_total', 'charms.sample_nation']].dropna(
    subset=['charms.sample_nation'])
countries_df['charms.sample_nation'] = countries_df['charms.sample_nation'].apply(lambda x: literal_eval(x))
countries_df = countries_df.explode('charms.sample_nation')
countries_df = countries_df.rename(columns={
    'charms.sample_nation': 'country'
})
# split evenly n_total if multiple countries
countries_df['n_total_scaled'] = countries_df.groupby('Key')['n_total'].transform(lambda x: x / len(x))
# replace some of the following values
countries_df = countries_df.replace({
    'USA': 'United States of America',
    'US': 'United States of America',
    'United States': 'United States of America',
    'UK': 'United Kingdom',
    'Korea': 'Republic of Korea',
})

# convert the country names to ISO 3166-1 alpha-3
countries_df['iso'] = countries_df['country'].map(country_to_iso3)
# drop None
check = countries_df[countries_df['iso'].isna()]['country']
countries_df = countries_df[countries_df['iso'].notna()]
# count occurrences of each country
country_counts = countries_df.groupby('iso').agg({
    'Key': 'nunique',
    'model_id': 'nunique',
    'n_total_scaled': 'sum'
}).reset_index()
# add country names
country_counts['country'] = country_counts['iso'].map({v: k for k, v in country_to_iso3.items()})
# print top countries by model_id
country_counts = country_counts.sort_values(by='n_total_scaled', ascending=False).reset_index(drop=True)
# %%
pop = pd.read_excel(os.path.join(data_dir, 'population', 'API_SP.POP.TOTL_DS2_en_excel_v2_23077.xls'), skiprows=3)
pop = pop[['Country Name', 'Country Code', '2024']].rename(columns={
    'Country Code': 'iso',
    '2024': 'population'
})
# add taiwan population which is missing
# merge with country_counts
country_counts = country_counts.merge(pop, on='iso', how='outer')
country_counts.loc[country_counts['iso'] == 'TWN', 'population'] = 23083961
# compute relative density per 10 million people
country_counts['rel_density'] = country_counts['n_total_scaled'] / country_counts['population'] * 10_000_000
country_counts = country_counts.sort_values(by='rel_density', ascending=False).reset_index(drop=True)
# drop rows with NaN rel_density
country_counts = country_counts[country_counts['rel_density'].notna()]
print(country_counts)
# %%

fig = px.choropleth(
    country_counts,
    locations="iso",  # column with country names
    color="rel_density",  # column with your values
    hover_name="country",
    color_continuous_scale="Viridis",
    projection="natural earth",
    scope='world',
    basemap_visible=True,
)
# change legend title
fig.update_layout(coloraxis_colorbar=dict(title="Stroke patients per 10M people"))
fig.write_image(os.path.join(fig_dir, f'country_reldensity.png'), scale=3, width=1000, height=600)
fig.show()
