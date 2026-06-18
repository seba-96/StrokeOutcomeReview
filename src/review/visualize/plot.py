import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



def plot_recipe(best_combos, predictors, fig_dir, validation=0, figsize=(14, 12), adjust=None, palette=None,
                scenario_legend=True, stroke_type='ischemic', feature_name_map=None,
                scale_circle_size=True, constant_circle_size=250):
    from matplotlib.patches import Patch

    def _to_phase(v):
        sval = str(v).strip().lower()
        if sval in {'1', 'true', 'post', 'yes'}:
            return 'post'
        return 'pre'

    severity_order = {'mild': 0, 'moderate': 1, 'severe': 2}
    scenario_order = ['mild-pre', 'mild-post', 'moderate-pre', 'moderate-post', 'severe-pre', 'severe-post']

    best_combos_sorted = best_combos.copy()
    best_combos_sorted['severity_'] = best_combos_sorted['severity'].astype(str).str.strip().str.lower()
    best_combos_sorted['phase_'] = best_combos_sorted['post_treatment_models'].apply(_to_phase)
    best_combos_sorted['scenario_'] = best_combos_sorted['severity_'] + '-' + best_combos_sorted['phase_']
    best_combos_sorted['severity_order_'] = best_combos_sorted['severity_'].map(severity_order).fillna(99)
    best_combos_sorted['phase_order_'] = best_combos_sorted['phase_'].map({'pre': 0, 'post': 1}).fillna(99)
    best_combos_sorted = best_combos_sorted.sort_values(
        by=['severity_order_', 'phase_order_', 'pred_auc'],
        ascending=[True, True, False]
    ).reset_index(drop=True)
    if best_combos_sorted.empty:
        raise Exception("No recipes found")


    # drop columns with all 0s
    best_combos_sorted = best_combos_sorted.loc[
        :, ['severity', 'post_treatment_models', 'pred_auc', 'support_n', 'combo'] + [pred for pred in predictors if
                                                                                      best_combos_sorted[
                                                                                          pred].max() > 0]]
    best_combos_sorted['severity_'] = best_combos_sorted['severity'].astype(str).str.strip().str.lower()
    best_combos_sorted['phase_'] = best_combos_sorted['post_treatment_models'].apply(_to_phase)
    best_combos_sorted['scenario_'] = best_combos_sorted['severity_'] + '-' + best_combos_sorted['phase_']
    scenario_counts = best_combos_sorted.groupby(['severity_', 'phase_'])['combo'].transform('size')
    combo_idx = best_combos_sorted.groupby(['severity_', 'phase_']).cumcount() + 1
    base_id = best_combos_sorted['severity_'].str.capitalize() + ' - ' + best_combos_sorted['phase_']
    best_combos_sorted['id'] = np.where(
        scenario_counts > 1,
        base_id + ' - comb. ' + combo_idx.astype(str),
        base_id
    )
    comb_ids = best_combos_sorted['id'].tolist()
    # sort predictors by overall frequency
    predictors_order = best_combos_sorted.loc[
        :, [pred for pred in predictors if pred in best_combos_sorted.columns]].sum().sort_values(
        ascending=False).index.tolist()

    # binary matrix: predictors (rows) x combinations (columns)
    pred_cols = [pred for pred in predictors_order if pred in best_combos_sorted.columns]
    mat = best_combos_sorted[pred_cols].T
    mat.columns = comb_ids

    # ID maps
    pred_auc_map = pd.Series(best_combos_sorted['pred_auc'].values, index=comb_ids)
    id_to_scenario = pd.Series(best_combos_sorted['scenario_'].values, index=comb_ids)

    # colors for scenarios
    present_scenarios = list(pd.unique(id_to_scenario.values))
    scenarios = [s for s in scenario_order if s in present_scenarios] + [
        s for s in present_scenarios if s not in scenario_order
    ]
    # scenarios = ['mild-pre', 'mild-post', 'moderate-pre', 'moderate-post', 'severe-pre', 'severe-post']
    # palette = sns.color_palette("Set2", n_colors=len(scenarios))
    if not palette:
        palette = ['limegreen', 'forestgreen', 'gold', 'darkorange', 'dodgerblue', 'mediumblue']
    scenario_to_color = dict(zip(scenarios, palette))
    id_to_color = id_to_scenario.map(scenario_to_color)

    # draw a neutral grid (no cell color), only circles will carry color
    grid = mat.copy()
    grid.loc[:, :] = 0  # uniform background
    plt.figure(figsize=figsize)
    plt.rcParams.update({
                            'font.size': 15
                        })
    ax = sns.heatmap(
        grid, annot=False, cbar=False, cmap='Greys', vmin=0, vmax=0,
        linewidths=0.5, linecolor='#dddddd'
    )

    # scale circle size by pred_auc (or keep constant size)
    col_ids = comb_ids
    auc_vals = pred_auc_map.reindex(col_ids).astype(float).values
    s_min, s_max = float(np.nanmin(auc_vals)), float(np.nanmax(auc_vals))

    def scale_size(x, xmin=s_min, xmax=s_max, s_lo=80, s_hi=800):
        if not np.isfinite(x):
            x = 0.0
        if xmax <= xmin:
            return (s_lo + s_hi) / 2.0
        return s_lo + (s_hi - s_lo) * (float(x) - xmin) / (xmax - xmin + 1e-9)

    # overlay circles at predictor==1
    for j, col in enumerate(col_ids):
        if scale_circle_size:
            s = scale_size(pred_auc_map.get(col, np.nan))
        else:
            s = constant_circle_size
        color = id_to_color.get(col, "#333333")
        ones = np.where(mat[col].values == 1)[0]
        if len(ones) > 0:
            ax.scatter(
                np.full_like(ones, j + 0.5, dtype=float),
                ones + 0.5,
                s=s,
                c=[color],
                marker='o',
                edgecolors='black',
                linewidths=0.4,
                zorder=3
            )

    # color x tick labels by scenario
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=20)
    if feature_name_map:
        ax.set_yticklabels([feature_name_map.get(lbl.get_text(), lbl.get_text()) for lbl in ax.get_yticklabels()], fontsize=20)
    for tl in ax.get_xticklabels():
        tl.set_color(id_to_color.get(tl.get_text(), "#333333"))

    # legends: scenario (color) and size (support_n)
    if scenario_legend:
        scenario_handles = [Patch(facecolor=clr, edgecolor='black', label=scn) for scn, clr in
                            scenario_to_color.items()]
        legend1 = ax.legend(scenario_handles, [h.get_label() for h in scenario_handles],
                            title='Scenario', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
                            fontsize=20, title_fontsize=20)
        ax.add_artist(legend1)

    # size legend (use up to 3 reference sizes)
    if scale_circle_size and np.isfinite(s_min) and np.isfinite(s_max):
        unique_vals = np.unique(auc_vals[np.isfinite(auc_vals)])
        if unique_vals.size > 0:
            if unique_vals.size >= 3:
                levels = [unique_vals[0], np.median(unique_vals), unique_vals[-1]]
            elif unique_vals.size == 2:
                levels = [unique_vals[0], unique_vals[-1]]
            else:
                levels = [unique_vals[0]]
            size_handles = [
                plt.scatter([], [], s=scale_size(v), facecolors='none', edgecolors='black', label=f"{float(v):.2f}")
                for v in levels
            ]
            ax.legend(size_handles, [h.get_label() for h in size_handles],
                      title='Pred. AUC', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0.,
                      fontsize=14, title_fontsize=15)
    # make some space on the right for legends
    if not adjust:
        adjust = dict(right=0.75, left=0.25, bottom=0.2)
    plt.subplots_adjust(**adjust)
    # plt.xlabel('Combination')
    # plt.ylabel('Predictor')
    plt.savefig(os.path.join(fig_dir, f'best_combinations_heatmap_val{validation}_{stroke_type}.png'), dpi=300)
    plt.show()



def prepare_forest_df(results, df, sensitivity=False, advanced_cat=None):
    import re
    if not advanced_cat:
        advanced_cat = ['Lesion_volume', 'Lesion_location', 'Radiomics', 'Brain_health', 'Neural_network']
    results['row_type'] = 'group_summary'
    add_single = []

    for cat in advanced_cat:
        if sensitivity:
            if cat == 'Baseline':
                subset = df[-df['nested']
                            & (df['Clinical_status'] == 1)
                            & (df[advanced_cat].sum(axis=1) == 0)
                            & (df['sample.stroke_type'] == 'ischemic')
                            & (df['three_months_outcome'] == 1)
                            & (df['outcome.worse_class_cutoff'] == 3)]
            else:
                subset = df[-df['nested']
                            & (df[cat] == 1)
                            & (df['sample.stroke_type'] == 'ischemic')
                            & (df['three_months_outcome'] == 1)
                            & (df['outcome.worse_class_cutoff'] == 3)]
        else:
            if cat == 'Baseline':
                subset = df[-df['nested']
                            & (df['Clinical_status'] == 1)
                            & (df[advanced_cat].sum(axis=1) == 0)
                            & (df['sample.stroke_type'] == 'ischemic')]
            else:
                subset = df[-df['nested'] & (df[cat] == 1) & (df['sample.stroke_type'] == 'ischemic')]

        # sort by auc
        subset = subset.sort_values(by='auc', ascending=True).reset_index(drop=True)

        if len(subset) < 2:
            print(f'Skipping category {cat} due to insufficient studies ({len(subset)})\n')
            continue
        add_single.append({
            'Study': cat.replace('_', ' '),
            'row_type': 'group_header',
            'Imaging feature': '',
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
            adv_pred = ', '.join(np.unique(row_pred[row_cat == cat.replace('_', ' ').lower()]).tolist())
            adv_pred = adv_pred.replace('infarct growth, ', '').replace('white matter hyperintensities', 'WMH')
            other_pred = ', '.join(np.unique(row_pred[row_cat != cat.replace('_', ' ').lower()]).tolist())
            # remove nan from other_pred
            # other_pred = other_pred.replace(', nan', '')
            # wrap text every 40 characters
            import textwrap
            adv_pred = '\n'.join(textwrap.wrap(adv_pred, width=10))
            other_pred = '\n'.join(textwrap.wrap(other_pred, width=45))
            if row['study_id'] == 'Neuberger et al (2023)' and cat == 'Lesion_location':
                adv_pred = 'location-specific ASPECTS'
            if row['study_id'] == 'Wong et al (2022)' and cat == 'Lesion_location':
                adv_pred = '30 selected brain regions'
            if row['study_id'] == 'Tolhuisen et al (2022)' and cat == 'Lesion_location':
                adv_pred = 'mRS-relevant regions'
            if row['study_id'] == 'Bonkhoff et al (2023)' and cat == 'Lesion_location':
                adv_pred = 'rich-club regions'
            if row['study_id'] == 'Bonkhoff et al (2022)' and cat == 'Lesion_location':
                adv_pred = '129 atlas based regions-tracts'
            if row['study_id'] == 'Regenhardt et al (2022)' and cat == 'Lesion_location':
                adv_pred = '128 atlas based regions-tracts'
            # get all capital letter abbreviations and join with -
            if cat in ['Radiomics', 'Neural_network'] and not adv_pred.startswith('ASPECTS'):
                abbreviations = '-'.join(re.findall(r'\b[A-Z]{2,}|lesion\b', adv_pred))
                if adv_pred != 'infarct density':
                    adv_pred = abbreviations + ' radiomics' if cat == 'Radiomics' else abbreviations + ' raw image'
            add_single.append({
                'Study': row['study_id'],
                'name': cat,
                'category': cat,
                'estimate': auc,
                'ci_0.025': ci_lower,
                'ci_0.975': ci_higher,
                'Severity': row['severity'],
                'Validated': 'yes' if row['external_validation'] == 1 else 'no',
                'Post-treatment': 'yes' if row['post_treatment_models'] == 1 else 'no',
                'row_type': 'study',
                'Imaging feature': adv_pred,
                'Other features': other_pred,
            })
        pooled = results.loc[results['name'] == cat, :].iloc[0].to_dict()
        mdl_summary = f'RE model (pooled AUC)'
        mdl_delta = f'Δ={pooled["delta"]:.3f} ({pooled["delta_ci_0.025"]:.3f}-{pooled["delta_ci_0.975"]:.3f}) p={pooled["p-value"]:.3f}'
        mdl_het = f'I² {pooled["I2"]:.1f}%, τ² {pooled["tau2"]:.2f} p={pooled["p_het"]:.3f}'
        pooled['Validated'] = ''
        pooled['Post-treatment'] = ''
        pooled['Severity'] = ''
        add_single.append({
            'Study': mdl_summary,
            'Imaging feature': '',
            **pooled
        })
        pooled['estimate'] = np.nan
        pooled['ci_0.025'] = np.nan
        pooled['ci_0.975'] = np.nan

        add_single.append({
            'Study': mdl_delta,
            'Imaging feature': '',
            **pooled
        })
        add_single.append({
            'Study': mdl_het,
            'Imaging feature': '',
            **pooled
        })

    add_single = pd.DataFrame(add_single)
    results['Study'] = 'Baseline'
    results['Imaging feature'] = ''
    results['Validated'] = ''
    results['Post-treatment'] = ''
    results['Severity'] = ''
    results = pd.concat(
        [results[results['category'] == 'Clinical_status'].reset_index(drop=True), add_single]).reset_index(
        drop=True)
    del add_single
    results['row_type_'] = results['row_type']
    results.loc[results['Study'] == 'Baseline', 'row_type_'] = 'baseline'
    # replace nan with empty string in Imaging feature, Validated, Post-treatment, Severity
    results['Imaging feature'] = results['Imaging feature'].fillna('')
    results['Other features'] = results['Other features'].fillna('')
    results['Validated'] = results['Validated'].fillna('')
    results['Post-treatment'] = results['Post-treatment'].fillna('')
    results['Severity'] = results['Severity'].fillna('')

    return results




def forest_plot_from_df(
    df: pd.DataFrame,
    effect_col: str,
    ci_low_col: str,
    ci_high_col: str,
    label_col: str = "Study",
    row_type_col: str = "row_type",
    extra_cols=None,
    weight_col: str | None = None,
    numeric_col_header: str = "AUC (95% CI)",
    x_axis_label: str = "AUC",
    figsize=None,
    xlim=None,
    fontsize: int = 8,
    header_fontsize: int | None = None,
    group_fontsize: int | None = None,
    row_height: float = 1.0,
    group_gap: float = 0.6,
    study_to_summary_gap: float = 0.3,
    text_col_widths=None,
    text_pad: float = 0.05,
    show_vertical_lines: bool = True,
    panel_width_ratios: list[int] | None = None,
    bold_labels=None,
    indent_labels=None,
    bold_groups: bool = True,
    bold_group_summary: bool = True,
    bold_overall: bool = True,
    indent_studies: bool = True,
    indent_group_summary: bool = True,
    pvalue_col: str | None = "p-value",
    pvalue_threshold: float = 0.05,
    color_by_col: str | None = None,
    color_map: dict | None = None,
    vertical_line_color: dict | None = None,
    all_diamonds: bool = False,
    xticks: list[float] | None = None,
    # --- NEW (optional) knobs for stable spacing ---
    linespacing: float = 1.2,
    row_pad_lines: float = 0.15,
    fig_height_pad_in: float = 0.9,
    subplots_adjust: dict | None = None,
    header_top_pad: float = 0.25,
    show_study_separators: bool = False,
    study_separator_kwargs: dict | None = None,
    xticks_labels=None) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes]]:

    def count_lines(text) -> int:
        if pd.isna(text) or text == "":
            return 1
        return str(text).count("\n") + 1

    if extra_cols is None:
        extra_cols = []
    if header_fontsize is None:
        header_fontsize = fontsize + 1
    if group_fontsize is None:
        group_fontsize = fontsize

    bold_labels = set(bold_labels or [])
    indent_labels = set(indent_labels or [])
    color_map = color_map or {}

    # ---------- Identify where to add gaps ----------
    should_add_gap = []
    should_add_study_gap = []
    for i, (_, rec) in enumerate(df.iterrows()):
        kind = str(rec.get(row_type_col, "study"))

        # gap after last group_summary in a consecutive block
        is_last_in_sequence = False
        if kind == "group_summary":
            if i + 1 >= len(df):
                is_last_in_sequence = True
            else:
                next_kind = str(df.iloc[i + 1].get(row_type_col, "study"))
                if next_kind != "group_summary":
                    is_last_in_sequence = True
        should_add_gap.append(is_last_in_sequence)

        # gap before the first group_summary after studies
        is_first_summary_after_study = False
        if kind == "group_summary" and i > 0:
            prev_kind = str(df.iloc[i - 1].get(row_type_col, "study"))
            if prev_kind == "study":
                is_first_summary_after_study = True
        should_add_study_gap.append(is_first_summary_after_study)

    # ---------- Build rows (NO y yet), compute row heights ----------
    rows = []
    for idx, (_, rec) in enumerate(df.iterrows()):
        kind = str(rec.get(row_type_col, "study"))
        label = "" if pd.isna(rec.get(label_col)) else str(rec.get(label_col))

        theta = float(rec[effect_col]) if pd.notnull(rec.get(effect_col)) else np.nan
        lo    = float(rec[ci_low_col]) if pd.notnull(rec.get(ci_low_col)) else np.nan
        hi    = float(rec[ci_high_col]) if pd.notnull(rec.get(ci_high_col)) else np.nan

        w = 1.0
        if weight_col is not None and pd.notnull(rec.get(weight_col, np.nan)):
            w = float(rec[weight_col])

        if pvalue_col and pvalue_col in df.columns:
            try:
                pval = float(rec[pvalue_col])
            except (TypeError, ValueError):
                pval = np.nan
        else:
            pval = np.nan

        lines_in_label = count_lines(label)
        lines_in_extra = [count_lines(rec.get(col, "")) for col in extra_cols]
        max_lines = max([lines_in_label, *lines_in_extra]) if extra_cols else lines_in_label

        # height in "row_height units"
        row_units = (max_lines + row_pad_lines) * row_height

        rows.append(dict(
            kind=kind,
            study_label=label,
            theta=theta,
            ci_low=lo,
            ci_high=hi,
            weight=w,
            extra={col: ("" if pd.isna(rec.get(col, "")) else str(rec.get(col, ""))) for col in extra_cols},
            p_value=pval,
            color_value=(rec.get(color_by_col) if color_by_col else None),
            row_units=row_units,
            add_group_gap=bool(should_add_gap[idx]),
            add_study_gap=bool(should_add_study_gap[idx]),
        ))

    # ---------- Compute y positions as CENTER of each allocated row band ----------
    header_y     = 0.0
    header_sep_y = header_y - row_height * 0.9

    y_cursor = header_sep_y
    for r in rows:
        if r["add_study_gap"]:
            y_cursor -= study_to_summary_gap * row_height

        h = r["row_units"]
        r["y"] = y_cursor - 0.5 * h   # center in its own band
        y_cursor -= h

        if r["add_group_gap"]:
            y_cursor -= group_gap * row_height

    ymin = y_cursor - row_height
    ymax = header_y + row_height * (0.8 + header_top_pad)

    # ---------- x-limits ----------
    effs  = np.array([r["theta"] for r in rows], dtype=float)
    ci_l  = np.array([r["ci_low"] for r in rows], dtype=float)
    ci_u  = np.array([r["ci_high"] for r in rows], dtype=float)
    mask_num = np.isfinite(effs)

    if xlim is None and np.any(mask_num):
        all_ci = np.concatenate([ci_l[mask_num], ci_u[mask_num]])
        width = float(all_ci.max() - all_ci.min())
        pad = 0.1 * width if width > 0 else 0.1
        x_min = float(all_ci.min() - pad)
        x_max = float(all_ci.max() + pad)
    elif xlim is not None:
        x_min, x_max = xlim
    else:
        x_min, x_max = -0.1, 0.1

    # ---------- weights for box size ----------
    study_weights = np.array([r["weight"] for r in rows if r["kind"] == "study" and np.isfinite(r["weight"])])
    max_w = float(study_weights.max()) if len(study_weights) else 1.0

    # ---------- AUTO FIG HEIGHT (key fix) ----------
    if figsize is None:
        total_units  = ymax - ymin
        total_points = total_units * (fontsize * linespacing)
        height_in    = max(4.0, total_points / 72.0 + fig_height_pad_in)
        figsize      = (10, height_in)

    fig = plt.figure(figsize=figsize)
    if panel_width_ratios is None:
        panel_width_ratios = [3, 4, 2]
    gs = GridSpec(1, 3, figure=fig, width_ratios=panel_width_ratios, wspace=0.02)

    ax_text = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1], sharey=ax_text)
    ax_num  = fig.add_subplot(gs[0, 2], sharey=ax_text)

    if subplots_adjust is None:
        subplots_adjust = dict(left=0.03, right=0.99, top=0.98, bottom=0.10, wspace=0.02)
    fig.subplots_adjust(**subplots_adjust)

    for ax in (ax_text, ax_plot, ax_num):
        ax.set_ylim(ymin, ymax)
        ax.spines["top"].set_visible(True)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(True)

    # ---------- Text panel ----------
    n_text_cols = 1 + len(extra_cols)
    if text_col_widths is None:
        text_col_widths = [2] + [1] * (n_text_cols - 1)
    if len(text_col_widths) != n_text_cols:
        raise ValueError("text_col_widths length does not match number of text columns.")

    col_pos = np.cumsum([0] + list(text_col_widths))
    col_pos = col_pos / col_pos[-1]

    ax_text.set_xlim(0, 1)
    ax_text.set_xticks([])
    ax_text.set_yticks([])

    # Per-column clip rectangles -> prevents horizontal overlap across columns
    col_clips = []
    for j in range(n_text_cols):
        rect = mpatches.Rectangle(
            (col_pos[j], ymin),
            col_pos[j + 1] - col_pos[j],
            ymax - ymin,
            transform=ax_text.transData,
            facecolor="none",
            edgecolor="none",
        )
        ax_text.add_patch(rect)
        col_clips.append(rect)

    if show_vertical_lines:
        for r in rows:
            if r["kind"] == "study" and r["y"] <= header_sep_y:
                yv = r["y"]
                row_half_height = 0.5 * r["row_units"]
                for x in col_pos[1:-1]:
                    ax_text.vlines(x, yv - row_half_height, yv + row_half_height,
                                   color="0.85", linewidth=0.8)

    col_names = [label_col] + list(extra_cols)
    for i, name in enumerate(col_names):
        x = 0.5 * (col_pos[i] + col_pos[i + 1])
        ax_text.text(
            x, header_y, name,
            ha="center", va="bottom",
            fontsize=header_fontsize, fontweight="bold",
            linespacing=linespacing,
            clip_on=True, clip_path=col_clips[i],
        )

    ax_text.axhline(header_sep_y, color="0.3", linewidth=0.8)

    for r in rows:
        yv = r["y"]
        if yv > header_sep_y:
            continue

        lbl  = r["study_label"]
        kind = r["kind"]

        fw = "normal"
        fs = fontsize
        if ((kind == "group_header" and bold_groups) or
            (kind == "group_summary" and bold_group_summary) or
            (kind == "overall" and bold_overall) or
            (lbl in bold_labels)):
            fw = "bold"
        if kind in ("group_header", "group_summary", "overall"):
            fs = group_fontsize

        indent = text_pad if ((kind == "study" and indent_studies) or
                              (kind == "group_summary" and indent_group_summary) or
                              (lbl in indent_labels)) else 0.0

        ax_text.text(
            col_pos[0] + indent, yv, lbl,
            ha="left", va="center",
            fontsize=fs, fontweight=fw,
            multialignment="left",
            linespacing=linespacing,
            clip_on=True, clip_path=col_clips[0],
        )

        for j, col in enumerate(extra_cols, start=1):
            ax_text.text(
                col_pos[j] + text_pad, yv, r["extra"].get(col, ""),
                ha="left", va="center",
                fontsize=fontsize,
                multialignment="left",
                linespacing=linespacing,
                clip_on=True, clip_path=col_clips[j],
            )

    if show_study_separators:
        sep_opts = dict(color="0.8", linewidth=0.6)
        if study_separator_kwargs:
            sep_opts.update(study_separator_kwargs)
        for r in rows:
            if r["kind"] != "study":
                continue
            bottom = r["y"] - 0.5 * r["row_units"]
            if bottom <= header_sep_y:
                ax_text.hlines(bottom, col_pos[0], col_pos[-1], **sep_opts)

    # ---------- Forest plot panel ----------
    ax_plot.set_xlim(x_min, x_max)
    ax_plot.set_xlabel(x_axis_label, fontsize=header_fontsize)

    if xticks is None:
        xticks = [0.6, 0.7, 0.8, 0.9, 1.0]
    if xticks_labels is None:
        xticks_labels = xticks
    ax_plot.set_xticks(xticks)
    ax_plot.set_xticklabels(xticks_labels, fontsize=12)

    ax_plot.yaxis.set_visible(False)

    if vertical_line_color:
        for i, c in vertical_line_color.items():
            if 0 <= i < len(rows):
                x_val = df[effect_col].iloc[i]
                y_top = rows[i]["y"]
                ax_plot.vlines(x_val, ymin=ymin, ymax=y_top, color=c, linestyle="--", linewidth=0.8, alpha=0.5)

    ax_plot.axhline(header_sep_y, color="0.3", linewidth=0.8)
    ax_plot.axvline(0, linestyle="--", color="0.5")

    for r in rows:
        eff = r["theta"]
        if not np.isfinite(eff):
            continue
        yv = r["y"]
        if yv > header_sep_y:
            continue

        lo, hi = r["ci_low"], r["ci_high"]
        kind = r["kind"]
        w = r["weight"]

        plot_color = color_map.get(r["color_value"], "C0")

        ax_plot.hlines(yv, lo, hi, color=plot_color)
        if kind == "study" and not all_diamonds:
            size = 50 * np.sqrt(w / max_w) if max_w > 0 else 50
            ax_plot.scatter([eff], [yv], s=size, marker="s", color=plot_color, zorder=3)
        else:
            ax_plot.scatter([eff], [yv], s=80, marker="D", color=plot_color, zorder=4)

    # ---------- Numeric CI panel ----------
    ax_num.set_xlim(0, 1)
    ax_num.set_xticks([])
    ax_num.set_yticks([])

    ax_num.text(
        1 - text_pad, header_y, numeric_col_header,
        ha="right", va="bottom",
        fontsize=header_fontsize, fontweight="bold",
        linespacing=linespacing,
    )
    ax_num.axhline(header_sep_y, color="0.3", linewidth=0.8)

    for r in rows:
        eff = r["theta"]
        if not np.isfinite(eff):
            continue
        yv = r["y"]
        if yv > header_sep_y:
            continue

        txt = f"{eff:.2f} ({r['ci_low']:.2f} to {r['ci_high']:.2f})"
        pval = r.get("p_value", np.nan)
        fw = "bold" if np.isfinite(pval) and pval < pvalue_threshold else "normal"
        ax_num.text(
            1 - text_pad, yv, txt,
            ha="right", va="center",
            fontsize=fontsize,
            fontweight=fw,
            linespacing=linespacing,
        )

    return fig, (ax_text, ax_plot, ax_num)

