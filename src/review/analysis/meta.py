import logging
import re, math, numpy as np, pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from itertools import product
from sklearn.model_selection import KFold, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


country_to_iso3 = {
    "China": "CHN",
    "United States of America": "USA",
    "Germany": "DEU",
    "Switzerland": "CHE",
    "Canada": "CAN",
    "France": "FRA",
    "Australia": "AUS",
    "Italy": "ITA",
    "Taiwan": "TWN",
    "South Korea": "KOR",
    "Netherlands": "NLD",
    "United Kingdom": "GBR",
    "Spain": "ESP",
    "Finland": "FIN",
    "Belgium": "BEL",
    "Poland": "POL",
    "Japan": "JPN",
    "Hong Kong SAR": "HKG",
    "Singapore": "SGP",
    "Sweden": "SWE",
    "Republic of Korea": "KOR",
    "Europe": None,
    "Brazil": "BRA",
    "Austria": "AUT",
    "Greece": "GRC",
    "Norway": "NOR",
    "Thailand": "THA",
    "Denmark": "DNK",
    "Portugal": "PRT",
    "Qatar": "QAT",
    "Oceania": None,
    "East Asia": None,
    "Saudi Arabia": "SAU",
    "Serbia": "SRB",
    "North America": None,
    "New Zealand": "NZL",
    "multiple": None,
    "European countries": None,
    "Philippines": "PHL",
    "Hong Kong": "HKG",
    "Scotland": None,  # part of GBR, not ISO
    "Ireland": "IRL",
    "The Netherlands": "NLD",
    "Asia": None
}



def parse_ci(s):
    if isinstance(s, str):
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
    elif isinstance(s, (list, tuple)) and len(s) == 2:
        return float(s[0]), float(s[1])
    elif isinstance(s, (int, float)):
        logger.error(f"Cannot parse confidence interval from a single number: {s}")

    return (np.nan, np.nan)


def hanley_mcneil_auc_se(auc, n1, n0):
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0)
    return math.sqrt(var) if var > 0 else np.nan


def derive_se(row):
    # 1) reported SE
    if isinstance(row.var_metric, str) and "standard error" in row.var_metric.lower():
        try:
            if isinstance(row.var_value, (float, int)):
                return float(row.var_value)
            else:
                se = pd.to_numeric(row.var_value.replace('[', '').replace(']', ''), errors="coerce")
            return se
        except AttributeError as e:
            logger.error(f"Error parsing {row.var_value} as standard error: {e}")
            return np.nan
    # 2) reported SD
    if isinstance(row.var_metric, str) and "standard deviation" in row.var_metric.lower():
        sd = pd.to_numeric(row.var_value.replace('[', '').replace(']', ''), errors="coerce")
        if pd.notna(sd) and pd.notna(row.n_total) and row.n_total > 0:
            return sd / math.sqrt(row.n_total)
    # 3) reported 95 % CI
    if isinstance(row.var_metric, str) and "confidence" in row.var_metric.lower():
        lo, hi = parse_ci(row.var_value)
        if not np.isnan(lo):  # got both limits
            return (hi - lo) / (2 * 1.96)
    # 4) counts → Hanley-McNeil
    if pd.notna(row.n_pos) and pd.notna(row.n_neg):
        return hanley_mcneil_auc_se(row.auc, row.n_pos, row.n_neg)
    return np.nan








# helper to back-transform logit-AUC to plain AUC
def inv_logit(x): return np.exp(x) / (1 + np.exp(x))



def meta_reg(df, predictors, yi_col='yi', vi_col='vi', covariates=None, alpha=0.05, fig_dir=None,
             drop_covariates=True, transform=True, re_formula='~ 1 | Key'):

    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    try:
        metafor = importr('metafor')
    except Exception:
        ro.r('install.packages("metafor", repos="https://cloud.r-project.org")')
        metafor = importr('metafor')

    if isinstance(covariates, str):
        covariates = [covariates]
    # check that there is at least one positive value for each row in the predictors
    if any(df[predictors].sum(axis=1) == 0):
        logger.warning("There is at least one row with all predictors equal to zero. Please remove these rows before running the meta-regression.")

    _rma_df = df[[yi_col, vi_col, 'Key', 'cutoff', 'three_months_outcome'] + predictors + (covariates if covariates else [])].copy()
    _rma_df['Key'] = _rma_df['Key'].astype(str)
    # transform also cutoff to string
    _rma_df['cutoff'] = _rma_df['cutoff'].astype(str)
    _rma_df['three_months_outcome'] = _rma_df['three_months_outcome'].astype(str)
    # identify predictors with spaces and replace them with _
    pred_with_spaces = [pred for pred in predictors if ' ' in pred]
    if pred_with_spaces:
        raise ValueError(f"Predictors with spaces are not allowed: {pred_with_spaces}. Please rename them before passing to the function.")
    with (ro.default_converter + pandas2ri.converter).context():
        df_r = ro.conversion.get_conversion().py2rpy(_rma_df)

    # Extract numeric vectors (avoid passing column names as strings)
    yi_r = df_r.rx2(yi_col)
    vi_r = df_r.rx2(vi_col)  # passed as V
    if predictors or covariates:
        formula = ro.Formula('~ ' + ' + '.join(predictors + (covariates if covariates else [])))
    else:
        # set as None
        formula = ro.Formula('~ 1')
    logger.warning(f"Formula {formula}")
    if re_formula:
        logger.warning(f"Re-formula {re_formula}")
        # if its a list of re_formula then transform into a list of random effects
        if isinstance(re_formula, list):
            re_formula = [ro.Formula(f) for f in re_formula]
            re_formula = ro.ListVector(re_formula)
        else:
            re_formula = ro.Formula(re_formula)
        fit_A = metafor.rma_mv(yi=yi_r,
                               V=vi_r,
                               mods=formula,
                               random=re_formula,
                               data=df_r,
                               method="REML")
    else:
        fit_A = metafor.rma(yi=yi_r,
                           vi=vi_r,
                           mods=formula,
                           data=df_r,
                           method="REML")
    ro.globalenv['fit_A'] = fit_A
    ro.r('''
        fit_A_summary <- summary(fit_A)
        coef_df <- as.data.frame(coef(fit_A_summary))
        tau2 <- fit_A$tau2
        I2   <- fit_A$I2
        p_het <- fit_A$QEp   # p-value for test of (residual) heterogeneity
    ''')
    with (ro.default_converter + pandas2ri.converter).context():
        res_df = ro.conversion.get_conversion().rpy2py(ro.globalenv['coef_df'])
        tau2 = ro.globalenv['tau2'][0]
        I2 = ro.globalenv['I2'][0]
        p_het = ro.globalenv['p_het'][0]
    # set index as column
    res_df = res_df.reset_index().rename(columns={
        'index': 'name',
        'ci.lb': 'ci_0.025',
        'ci.ub': 'ci_0.975',
        'pval': 'p-value'
    })
    res_df['name'] = res_df['name'].replace({
                                                'intrcpt': 'intercept'
                                            })
    logger.warning(res_df)
    if transform:
        # add the intercept to all other estimates
        res_df.loc[res_df['name'] != 'intercept', ['estimate','ci_0.025', 'ci_0.975']] += res_df.loc[res_df['name'] == 'intercept', 'estimate'].values[0]
        res_df['estimate'] = res_df['estimate'].apply(inv_logit)
        res_df['ci_0.025'] = res_df['ci_0.025'].apply(inv_logit)
        res_df['ci_0.975'] = res_df['ci_0.975'].apply(inv_logit)
        logger.warning("Transformed estimates back to AUC scale using inverse logit.")

    # print delta
    for _, row in res_df.iterrows():
        logger.info(f"{row['name']}: Pooled AUC: {row['estimate']:.3f} (p-value: {row['p-value']:.3f}")
    # mark significant predictors
    res_df['p-value_significant'] = res_df['p-value'] < alpha
    res_df['tau2'] = tau2
    res_df['I2'] = I2
    res_df['p_het'] = p_het

    return res_df



def run_multimodal_meta_regression(df, advanced_imaging_categories, stroke_type='ischemic', covariates='external_validation'):

    results = []
    for category in advanced_imaging_categories:

        subset_clinical = df[(df['sample.stroke_type'] == stroke_type) & (df['Clinical_status'] == 1) & (-df['nested'])].copy()
        subset_clinical = subset_clinical[subset_clinical[advanced_imaging_categories].sum(axis=1) == 0]

        subset_advanced = df[(df['sample.stroke_type'] == stroke_type) & (df[category] == 1) & (-df['nested'])].copy()


        # collapse together
        subset = pd.concat([subset_clinical, subset_advanced]).reset_index(drop=True)

        if subset_advanced.shape[0] < 2:
            logger.warning(f"Skipping category {category} due to insufficient models ({subset_advanced.shape[0]} models)")
            continue
        logger.warning(f"Number of models in the subset: {subset.shape[0]}")

        res_df = meta_reg(subset,
                          predictors=[category],
                          covariates=covariates,
                          re_formula=None,
                          fig_dir=None)
        res_df['category'] = category
        res_df['count'] = subset_advanced.shape[0]
        res_df['stroke_type'] = stroke_type
        res_df['external_validation'] = subset_advanced['external_validation'].mean() * 100
        res_df['studies'] = '; '.join(subset_advanced['study_id'])
        results.append(res_df)

    results = pd.concat(results).reset_index(drop=True)
    # take out (one) intercept row, ie clinical status only and put the correct information
    intercept = results[results['name'] == 'intercept'].iloc[:1]
    intercept['category'] = 'Clinical_status'
    intercept['count'] = subset_clinical.shape[0]
    intercept['external_validation'] = subset_clinical['external_validation'].mean() * 100
    intercept['studies'] = '; '.join(subset_clinical['study_id'])
    # drop all intercept rows from results and concat back one intercept
    results = results[results['name'] != 'intercept']
    results['delta'] = results['estimate'] - intercept['estimate'].values[0]
    results['delta_ci_0.025'] = results['ci_0.025'] - intercept['estimate'].values[0]
    results['delta_ci_0.975'] = results['ci_0.975'] - intercept['estimate'].values[0]
    results = pd.concat([intercept, results]).reset_index(drop=True)

    return results


def run_nested_meta_regression(df, advanced_imaging_categories, rho=0.7):

    subset = df[
        -df['nested'] & (df['model.nested_comparison'] == 1) & (df['model.comparison_has_clinical_status'] == 1)].copy()
    # drop Favilla et al (2025) as it does not report sufficient data
    subset = subset[subset['study_id'] != 'Favilla et al (2025)'].copy()
    logger.info(f"Number of studies with nested comparison: {subset.shape[0]}")
    # compute auc difference
    subset['auc_diff_adv'] = (subset['auc'] - subset['model.without_advanced_imaging_performance_value'])
    subset['auc_diff_clin'] = (subset['model.advanced_imaging_performance_value'] - subset['auc'])
    # merge together
    subset['auc_diff'] = subset['auc_diff_adv'].combine_first(subset['auc_diff_clin'])

    # manually categorise studies which have shown inferior predictive accuracy when adding advanced neuroimaging
    subset.loc[subset['study_id'] == 'Johnston et al (2009)', 'Lesion_volume'] = 1
    subset.loc[subset['study_id'] == 'Oliveira et al (2023)', 'Neural_network'] = 1
    # print mean delta auc for each advanced imaging category
    for category in advanced_imaging_categories:
        cat_subset = subset[(subset[category] == 1)]
        if not cat_subset.empty:
            print(
                f"Mean delta AUC for {category}: {cat_subset['auc_diff'].mean():.3f} ± {cat_subset['auc_diff'].std():.3f} (N={cat_subset.shape[0]})")

    subset = subset.drop(columns=['var_value'])
    # combine variance values
    subset['var_value'] = subset[
        'model.without_advanced_imaging_performance_variance_value'].combine_first(
        subset['model.advanced_imaging_performance_variance_metric'])
    subset['se2'] = subset.apply(derive_se, axis=1).clip(0.000000001, None)
    subset['se_diff'] = np.sqrt(
        subset['se'] ** 2 + subset['se2'] ** 2 - (
                    2 * rho * subset['se'] * subset['se2']))
    subset['vi_diff'] = subset['se_diff'] ** 2

    subset = subset.drop(columns=['yi', 'vi'])
    subset = subset.rename(columns={
        'auc_diff': 'yi',
        'vi_diff': 'vi'
    })
    # run the analysis for each category
    all_results = []
    studies = []
    for category in advanced_imaging_categories:
        for ischemic in [True]:
            if ischemic:
                cat_subset = subset[
                    (subset[category] == 1) & (subset['sample.stroke_type'].str.startswith('ischemic'))]
            else:
                cat_subset = subset[
                    (subset[category] == 1) & (subset['sample.stroke_type'] == 'hemorrhagic')]
            if cat_subset.shape[0] >= 2:
                res_df = meta_reg(cat_subset,
                                  predictors=[],
                                  covariates=[],
                                  re_formula=None,
                                  drop_covariates=False,
                                  transform=False,
                                  fig_dir=None)
                res_df['category'] = category
                res_df['count'] = cat_subset.shape[0]
                res_df['stroke_type'] = 'ischemic' if ischemic else 'hemorrhagic'
                res_df['meta_analysis'] = 'yes'
                res_df['studies'] = '; '.join(cat_subset['study_id'])
                all_results.append(res_df)
                studies.append({
                    'category': category,
                    'stroke_type': 'ischemic' if ischemic else 'hemorrhagic',
                    'studies': cat_subset['study_id'].tolist(),
                    'auc_delta': cat_subset['yi'].tolist(),
                    'var_delta': cat_subset['vi'].tolist(),
                    'Key': cat_subset['Key'].tolist()
                })
            else:
                logger.warning(
                    f"Not enough models for category {category} (N={subset.shape[0]})")

    all_results = pd.concat(all_results)
    studies = pd.DataFrame(studies)

    return all_results, studies




def find_valid_combinations(
    X: pd.DataFrame,
    required: Optional[List[str]],
    min_models: int,
    *,
    candidates: Optional[List[str]] = None,
    max_size: Optional[int] = None,
    use_index_labels: bool = True,
    exact_match: bool = False
) -> Dict[Tuple[str, ...], List]:
    """
    Enumerate predictor combinations that satisfy:
      (i) include `required` (if provided), and
      (ii) have support in at least `min_models` rows (models).

    A model *supports* a combo if:
      - exact_match=False: it contains ALL predictors in the combo (may contain others).
      - exact_match=True : it contains EXACTLY the predictors in the combo (no extras across all columns of X).

    Dedup rule (non-exact path): if multiple combos share the same supporting models,
    keep only the largest combo (most predictors). On ties by size, keep lexicographically smallest combo.

    Parameters
    ----------
    X : pd.DataFrame
        Rows=models, columns=predictors. Entries are 0/1 or booleans.
    required : list[str] or None
        Predictors that must be included in each returned combo. If None, no required predictors.
    min_models : int
        Minimum number of models that must support the combo.
    candidates : list[str] or None
        Optional pool of *additional* predictors (excluding `required`). Ignored when exact_match=True,
        because combos are taken from exact row patterns.
    max_size : int or None
        Maximum total size of the combo (including required). If None, no cap.
    use_index_labels : bool
        If True, return row index labels; else return 0-based row positions.
    exact_match : bool
        If True, require exact equality between a model's active predictors and the combo.

    Returns
    -------
    dict
        { combo_tuple (sorted): [model_index, ...] }
        Where combo_tuple contains predictor names in sorted order, and model_index
        contains either DataFrame index labels (if use_index_labels=True) or
        0-based row positions (if False).

    Raises
    ------
    ValueError
        If min_models < 1, required predictors not in X.columns, or candidates not in X.columns.

    Notes
    -----
    - In non-exact mode, the deduplication ensures that if combo1 ⊂ combo2 and they
      have identical support, only combo2 is kept.
    - In exact mode, each unique pattern of active predictors becomes one combo.
    - Combinations are returned sorted by: descending size, then descending pred_auc (if applicable),
      then ascending lexicographic order.
    """
    if min_models <= 0:
        raise ValueError("min_models must be >= 1")

    # --- normalize & validate required ---
    required = [] if required is None else list(required)
    missing = [c for c in required if c not in X.columns]
    if missing:
        raise ValueError(f"Required predictors not in columns: {missing}")

    # Precompute row ids for output
    row_ids = X.index.to_list() if use_index_labels else list(range(X.shape[0]))

    # ----------------------------------------------------------------------
    # EXACT MATCH MODE: group rows by their exact active-set of predictors
    # ----------------------------------------------------------------------
    if exact_match:
        # Build per-row tuple of active predictor names (sorted)
        B = X.astype(bool).to_numpy(dtype=bool)
        cols = np.array(X.columns)
        pattern_to_rows: Dict[Tuple[str, ...], List[int]] = {}

        for r, present in enumerate(B):
            act_cols = tuple(sorted(cols[present].tolist()))  # ensure sorted
            pattern_to_rows.setdefault(act_cols, []).append(r)

        out: Dict[Tuple[str, ...], List] = {}
        req_set = set(required)

        for combo, rows in pattern_to_rows.items():
            combo_set = set(combo)
            # must contain required, respect max_size if given,
            # and meet min_models
            if not req_set.issubset(combo_set):
                continue
            if max_size is not None and len(combo) > max_size:
                continue
            if len(rows) < min_models:
                continue
            # combo is already sorted
            out[combo] = [row_ids[i] for i in rows]

        # already unique by exact pattern; no dedup needed
        return dict(sorted(out.items(), key=lambda kv: (-len(kv[0]), kv[0])))

    # ----------------------------------------------------------------------
    # NON-EXACT (SUPERCOVER) MODE: branch-and-prune DFS + dedup by support
    # ----------------------------------------------------------------------
    B = X.astype(bool).to_numpy(dtype=np.bool_)
    cols = list(X.columns)
    col_to_pos = {c: i for i, c in enumerate(cols)}
    masks = {c: B[:, col_to_pos[c]] for c in cols}

    # base mask for required
    required_sorted = sorted(required)
    if len(required_sorted) == 0:
        base_mask = np.ones(B.shape[0], dtype=bool)
    else:
        base_mask = masks[required_sorted[0]].copy()
        for c in required_sorted[1:]:
            base_mask &= masks[c]

    base_support_idx = np.where(base_mask)[0]
    if base_support_idx.size < min_models:
        return {}

    # candidate pool (exclude required)
    if candidates is None:
        others = [c for c in cols if c not in set(required_sorted)]
    else:
        bad = [c for c in candidates if c not in X.columns]
        if bad:
            raise ValueError(f"Candidates not in columns: {bad}")
        others = [c for c in candidates if c not in set(required_sorted)]
    others_sorted = sorted(others)

    collected: List[Tuple[Tuple[str, ...], np.ndarray]] = []

    # include the required-only combo if it meets support & size cap
    if (max_size is None) or (len(required_sorted) <= max_size):
        collected.append((tuple(required_sorted), base_support_idx))

    if max_size is not None and max_size <= len(required_sorted):
        return _dedup_by_support(collected, row_ids)

    # DFS with pruning by min_models support
    def backtrack(prefix_names: List[str], prefix_mask: np.ndarray, start_pos: int):
        for j in range(start_pos, len(others_sorted)):
            name = others_sorted[j]
            new_mask = prefix_mask & masks[name]
            support_idx = np.where(new_mask)[0]
            if support_idx.size < min_models:
                continue

            combo = tuple(sorted(required_sorted + prefix_names + [name]))
            if max_size is None or len(combo) <= max_size:
                collected.append((combo, support_idx))

            if max_size is None or len(combo) < max_size:
                backtrack(prefix_names + [name], new_mask, j + 1)

    backtrack([], base_mask, 0)
    return _dedup_by_support(collected, row_ids)


def _dedup_by_support(
    collected: List[Tuple[Tuple[str, ...], np.ndarray]],
    row_ids: List
) -> Dict[Tuple[str, ...], List]:
    """
    Keep only the largest combo per unique support set (positions).
    Tie-break by size (larger first), then lexicographically smallest combo.

    Parameters
    ----------
    collected : list of tuple
        Each tuple is (combo_tuple, support_indices_array)
    row_ids : list
        Mapping from array positions to output identifiers

    Returns
    -------
    dict
        { combo_tuple: [row_id, ...] } sorted by descending size, then combo
    """
    best_for_support = {}
    for combo, supp_idx in collected:
        key = frozenset(supp_idx.tolist())  # use frozenset for clearer comparison
        prev = best_for_support.get(key)
        if prev is None:
            best_for_support[key] = (combo, supp_idx)
        else:
            prev_combo = prev[0]
            # Keep larger combo; on tie, keep lexicographically smaller
            if len(combo) > len(prev_combo) or (len(combo) == len(prev_combo) and combo < prev_combo):
                best_for_support[key] = (combo, supp_idx)

    items = sorted(best_for_support.values(), key=lambda t: (-len(t[0]), t[0]))
    out = {combo: [row_ids[i] for i in sorted(supp_idx.tolist())] for combo, supp_idx in items}
    return out


# ---------- metrics ----------
def weighted_mse(y_true, y_pred, w):
    return np.average((y_true - y_pred) ** 2, weights=w)

def weighted_mae(y_true, y_pred, w):
    return np.average(np.abs(y_true - y_pred), weights=w)

def weighted_r2(y_true, y_pred, w):
    y_bar = np.average(y_true, weights=w)
    ss_res = np.sum(w * (y_true - y_pred) ** 2)
    ss_tot = np.sum(w * (y_true - y_bar) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

@dataclass
class CVFoldResult:
    best_params: Dict[str, Any]
    wmse: float
    wmae: float
    mae: float
    r2: float
    r2_weighted: float


def learn_surrogate_and_rank(
    X_bin: pd.DataFrame,                  # binary inclusion (n x p) 0/1
    auc: pd.Series,                       # observed AUC in [0,1]
    se_auc: Optional[pd.Series],          # SE of AUC (optional)
    required: Optional[List[str]] = None,
    min_models: int = 5,
    *,
    candidates: Optional[List[str]] = None,
    exact_match: bool = False,
    penalty_lambda: float = 0.0,          # score = pred_auc - λ * size
    max_size: Optional[int] = None,       # cap combo size (incl. required)
    study_ids: Optional[pd.Series] = None,# groups for CV (paper IDs)
    covariates: Optional[pd.DataFrame] = None,        # extra columns
    covariate_values: Optional[Dict[str, Any]] = None,# scalars OR lists; grid is built
    model: str = "gbr",                   # "gbr", "xgb", "svr", or "ridge"
    param_grid: Optional[Dict[str, List[Any]]] = None,# override grids
    n_outer_splits: int = 5,
    n_inner_splits: int = 5,
    nested_cv: bool = True,               # NEW: skip nested CV when False
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Learn a surrogate model to predict AUC from predictor combinations and rank them.

    This function enumerates valid predictor combinations, trains a regression model
    (GBR, XGBoost, SVR, or Ridge) to predict AUC values, and ranks combinations based on
    predicted performance. It supports nested cross-validation for robust evaluation
    and can handle covariates and study-level grouping.

    Parameters
    ----------
    X_bin : pd.DataFrame
        Binary predictor inclusion matrix (n_models × n_predictors), where each entry
        is 0 or 1 indicating whether a predictor was used in that model.
    auc : pd.Series
        Observed AUC values for each model, must be in range [0, 1].
    se_auc : pd.Series, optional
        Standard errors of AUC values. If provided, used to compute sample weights
        as 1/SE² for weighted training.
    required : list of str, optional
        Predictors that must be included in all enumerated combinations.
    min_models : int, default=5
        Minimum number of models that must support a combination for it to be valid.
    candidates : list of str, optional
        Pool of candidate predictors to consider (excluding required). If None,
        uses all columns in X_bin not in required.
    exact_match : bool, default=False
        If True, combinations must exactly match model predictor sets (no extras).
        If False, models may contain additional predictors beyond the combination.
    penalty_lambda : float, default=0.0
        Penalty coefficient for combination size. Final score = pred_auc - λ × size.
        Use positive values to favor smaller combinations.
    max_size : int, optional
        Maximum total size of the combinations (including required predictors).
    study_ids : pd.Series, optional
        Study/paper identifiers for group-based cross-validation. If provided,
        ensures models from the same study stay together in train/test splits.
    covariates : pd.DataFrame, optional
        Additional covariate features (e.g., external_validation, treatment_type)
        to include as predictors. Categorical covariates are one-hot encoded.
    covariate_values : dict, optional
        Covariate values to use for prediction scenarios. Keys are covariate names,
        values can be scalars or lists. If lists provided, creates a grid of scenarios.
        Example: {"external_validation": [0, 1]} creates two scenarios.
    model : {"gbr", "xgb", "svr", "ridge"}, default="gbr"
        Regression model type:
        - "gbr": GradientBoostingRegressor
        - "xgb": XGBoost XGBRegressor
        - "svr": Support Vector Regression with scaling
        - "ridge": Ridge regression (L2-regularized linear regression) with scaling
    param_grid : dict, optional
        Custom hyperparameter grid for tuning. Keys must match sklearn pipeline
        parameter names (e.g., "reg__n_estimators"). If None, uses defaults.
    n_outer_splits : int, default=5
        Number of outer CV folds for nested CV evaluation (ignored if nested_cv=False).
    n_inner_splits : int, default=5
        Number of inner CV folds for hyperparameter tuning.
    nested_cv : bool, default=True
        If True, performs nested CV (outer folds for evaluation, inner for tuning).
        If False, tunes on all data and skips outer evaluation.
    random_state : int, default=42
        Random seed for reproducibility in CV splits and model training.

    Returns
    -------
    dict
        Result dictionary containing:
        - "outer_cv": dict with nested CV results (metrics, fold details) or empty if nested_cv=False
        - "final_model": dict with "best_params" and "estimator" (fitted sklearn Pipeline)
        - "scenarios": list of dicts, one per covariate scenario, each containing:
            - "values": dict of covariate values for this scenario
            - "ranked": list of combinations ranked by score, each with:
                - "combo": comma-separated predictor names
                - "size": number of predictors
                - "pred_auc": predicted AUC
                - "support_n": number of supporting models
                - "support_rows": row indices of supporting models
                - "score": pred_auc - penalty_lambda * size
        - "ranked": top-level ranked list (copy of first scenario if single scenario)
        - "note": string describing evaluation approach

    Notes
    -----
    - Sample weights are computed as 1/SE² if se_auc is provided
    - Predictions are clipped to [0, 1] to ensure valid AUC range
    - With nested_cv=True, combination predictions are averaged across outer folds
    - With nested_cv=False, predictions use the final model trained on all data
    - GroupKFold is used when study_ids provided to prevent data leakage

    """
    # sample weights
    if se_auc is not None:
        se = se_auc.to_numpy()
        sample_weight = 1.0 / np.clip(se, 1e-8, None) ** 2
    else:
        sample_weight = np.ones(len(auc), dtype=float)

    y = auc.to_numpy().astype(float)

    # predictors = [binary columns] + [covariates]
    if covariates is not None:
        X_all = pd.concat([X_bin, covariates], axis=1)
        covar_cols = list(covariates.columns)
    else:
        X_all = X_bin.copy()
        covar_cols = []

    bin_cols = list(X_bin.columns)
    for c in bin_cols:
        if X_all[c].dtype == bool:
            X_all[c] = X_all[c].astype(int)

    # preprocess: one-hot encode categorical covariates
    cat_cols = [c for c in covar_cols if X_all[c].dtype == 'object' or str(X_all[c].dtype).startswith('category')]
    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
        remainder="passthrough"
    )

    # model factory + grids
    model = model.lower()
    if model == "gbr":
        reg = GradientBoostingRegressor(random_state=random_state)
        default_grid = {
            "reg__n_estimators": [5, 10, 20, 50, 100, 500],
            "reg__max_depth": [2, 3, 4],
            "reg__min_samples_leaf": [3, 10, 15]
        }
        pipe = Pipeline([("pre", pre), ("reg", reg)])
    elif model == "xgb":
        from xgboost import XGBRegressor
        reg = XGBRegressor(
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=random_state,
            tree_method="auto"
        )
        default_grid = {
            "reg__n_estimators": [5, 10, 30, 50, 100],
            "reg__max_depth": [2, 3, 4, 5],
            'reg__lambda': [1.0, 2.0]
        }
        pipe = Pipeline([("pre", pre), ("reg", reg)])
    elif model == "svr":
        # Support Vector Regression with scaling
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        reg = SVR(gamma="auto")
        default_grid = {
            "reg__C": np.logspace(-3, 0.5, num=15),
            "reg__epsilon": np.logspace(-3, 0, num=15),
        }
        pipe = Pipeline([("pre", pre), ("scale", StandardScaler()), ("reg", reg)])
    elif model == "ridge":
        # Ridge regression with scaling
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        reg = Ridge(random_state=None)  # Ridge has no random_state affecting solution
        default_grid = {
            "reg__alpha": [0.1, 1.0, 10.0, 100.0, 250.0, 500.0, 1000.0],
        }
        pipe = Pipeline([("pre", pre), ("scale", StandardScaler()), ("reg", reg)])
    else:
        raise ValueError("model must be 'gbr', 'xgb', 'svr' or 'ridge'")

    grid = default_grid if param_grid is None else param_grid

    # ---------- enumerate combinations ONCE and build scenario grid ----------
    # Use provided mask if available, else all rows
    combos_X = None
    if covariates is not None and 'nested' in covariates.columns:
        try:
            combos_X = X_bin[~covariates['nested']]
        except Exception:
            print("covariates['nested'] not found, using all features")
            combos_X = X_bin.copy()
    else:
        combos_X = X_bin.copy()

    combos = find_valid_combinations(
        combos_X, required=required, min_models=min_models,
        candidates=candidates, max_size=max_size,
        use_index_labels=False, exact_match=exact_match
    )

    # Prepare covariate scenarios
    hold_base = {}
    if covariates is not None:
        for c in covariates.columns:
            if covariates[c].dtype == 'object' or str(covariates[c].dtype).startswith('category'):
                hold_base[c] = covariates[c].mode(dropna=True).iloc[0] if covariates[c].nunique() else ""
            else:
                hold_base[c] = float(np.nanmedian(covariates[c].to_numpy()))
    covariate_values = covariate_values or {}
    if covariates is not None:
        unknown = set(covariate_values.keys()) - set(covariates.columns)
        if unknown:
            raise ValueError(f"covariate_values includes unknown covariates: {sorted(unknown)}")

    def _as_list(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return list(v)
        return [v]

    grid_keys, grid_vals = [], []
    if covariates is not None:
        for c in covariates.columns:
            if c in covariate_values:
                grid_keys.append(c)
                grid_vals.append(_as_list(covariate_values[c]))
    scenario_assignments = [dict()] if not grid_keys else [dict(zip(grid_keys, vals)) for vals in product(*grid_vals)]

    all_cols = list(pd.concat([X_bin, covariates], axis=1).columns) if covariates is not None else list(X_bin.columns)
    def _scenario_key(d: Dict[str, Any]): return tuple(sorted(d.items()))

    def _predict_combo_row(model_pipe, combo: Tuple[str, ...], hold_vals: Dict[str, Any]) -> float:
        row = {c: 0 for c in all_cols}
        for c in combo:
            if c in row:
                row[c] = 1
        for c, v in hold_vals.items():
            if c in row:
                row[c] = v
        x_row = pd.DataFrame([row], columns=all_cols)
        return float(np.clip(model_pipe.predict(x_row)[0], 0.0, 1.0))

    # Helper to fit pipeline with proper sample_weight handling
    def _fit_pipe(pipe, X_fit, y_fit, w_fit):
        """Fit pipeline with sample weights, handling different model types."""
        try:
            # Try fitting with sample_weight parameter
            pipe.fit(X_fit, y_fit, reg__sample_weight=w_fit)
        except TypeError:
            # SVR doesn't support sample_weight directly, fit without it
            logger.warning("Model doesn't support sample_weight, fitting without weights")
            pipe.fit(X_fit, y_fit)
        return pipe

    # ---------- nested CV (outer eval, inner tuning) ----------
    param_names = list(grid.keys())
    param_values = list(grid.values())

    scenarios_out: List[Dict[str, Any]] = []
    top_level_ranked: List[Dict[str, Any]] = []

    if nested_cv:
        # accumulate per-fold combo predictions: {scenario_key: {combo: [pred, ...]}}
        agg_combo_preds: Dict[Tuple[Tuple[str, Any], ...], Dict[Tuple[str, ...], List[float]]] = {}

        def split_outer():
            if study_ids is not None:
                gkf = GroupKFold(n_splits=n_outer_splits, shuffle=True, random_state=random_state)
                yield from gkf.split(X_all, y, groups=study_ids)
            else:
                kf = KFold(n_splits=n_outer_splits, shuffle=True, random_state=random_state)
                yield from kf.split(X_all, y)

        def split_inner(train_idx):
            if study_ids is not None:
                groups_train = np.array(study_ids)[train_idx]
                uniq = np.unique(groups_train)
                if uniq.size >= 2:
                    gkf = GroupKFold(n_splits=min(n_inner_splits, uniq.size), shuffle=True, random_state=random_state)
                    for tr_local, va_local in gkf.split(train_idx, y[train_idx], groups_train):
                        yield train_idx[tr_local], train_idx[va_local]
                else:
                    # Fall back to KFold if not enough groups
                    n_splits_fallback = min(n_inner_splits, max(2, len(train_idx) // 3))
                    kf = KFold(n_splits=n_splits_fallback, shuffle=True, random_state=random_state)
                    for tr_local, va_local in kf.split(train_idx):
                        yield train_idx[tr_local], train_idx[va_local]
            else:
                kf = KFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)
                for tr_local, va_local in kf.split(train_idx):
                    yield train_idx[tr_local], train_idx[va_local]

        fold_results: List[CVFoldResult] = []
        n = len(y)
        oof_pred = np.full(n, np.nan, dtype=float)
        oof_true = np.full(n, np.nan, dtype=float)
        oof_w = np.zeros(n, dtype=float)

        for outer_tr, outer_te in split_outer():
            X_tr, X_te = X_all.iloc[outer_tr], X_all.iloc[outer_te]
            y_tr, y_te = y[outer_tr], y[outer_te]
            w_tr, w_te = sample_weight[outer_tr], sample_weight[outer_te]

            # inner tuning (min WA-MSE)
            best_params = None
            best_score = np.inf
            for values in product(*param_values):
                params = dict(zip(param_names, values))
                inner_scores = []
                for tr_idx, va_idx in split_inner(outer_tr):
                    X_tr2, X_va2 = X_all.iloc[tr_idx], X_all.iloc[va_idx]
                    y_tr2, y_va2 = y[tr_idx], y[va_idx]
                    w_tr2, w_va2 = sample_weight[tr_idx], sample_weight[va_idx]
                    pipe.set_params(**params)
                    _fit_pipe(pipe, X_tr2, y_tr2, w_tr2)
                    y_hat = np.clip(pipe.predict(X_va2), 0.0, 1.0)
                    inner_scores.append(weighted_mse(y_va2, y_hat, w_va2))
                mean_wa_mse = float(np.mean(inner_scores))
                if mean_wa_mse < best_score:
                    best_score, best_params = mean_wa_mse, params

            # fit best on outer train, evaluate on outer test
            pipe.set_params(**best_params)
            _fit_pipe(pipe, X_tr, y_tr, w_tr)
            y_pred = np.clip(pipe.predict(X_te), 0.0, 1.0)

            # store per-fold metrics
            fold_results.append(CVFoldResult(
                best_params=best_params,
                wmse=weighted_mse(y_te, y_pred, w_te),
                wmae=weighted_mae(y_te, y_pred, w_te),
                mae=float(np.mean(np.abs(y_te - y_pred))),
                r2=float(r2_score(y_te, y_pred)),
                r2_weighted=float(weighted_r2(y_te, y_pred, w_te))
            ))

            # fill pooled OOF arrays
            oof_pred[outer_te] = y_pred
            oof_true[outer_te] = y_te
            oof_w[outer_te] = w_te

            # predict all combinations for all scenarios with this fold's model
            for scen in scenario_assignments:
                hold_vals = hold_base.copy()
                hold_vals.update(scen)
                s_key = _scenario_key(scen)
                if s_key not in agg_combo_preds:
                    agg_combo_preds[s_key] = {}
                for combo in combos.keys():
                    auc_hat = _predict_combo_row(pipe, combo, hold_vals)
                    if combo not in agg_combo_preds[s_key]:
                        agg_combo_preds[s_key][combo] = []
                    agg_combo_preds[s_key][combo].append(auc_hat)

        # aggregate fold-wise metrics
        def _agg(attr):
            vals = np.array([getattr(fr, attr) for fr in fold_results], dtype=float)
            return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

        mean_metrics, std_metrics = {}, {}
        for name in ["wmse", "wmae", "mae", "r2", "r2_weighted"]:
            mean_metrics[name], std_metrics[name] = _agg(name)

        # pooled OOF metrics
        mask = np.isfinite(oof_pred)
        outer_dict = {
            "folds": [fr.__dict__ for fr in fold_results],
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
            "pooled_metrics": {
                "wmse": float(weighted_mse(oof_true[mask], oof_pred[mask], oof_w[mask])),
                "wmae": float(weighted_mae(oof_true[mask], oof_pred[mask], oof_w[mask])),
                "mae": float(np.mean(np.abs(oof_true[mask] - oof_pred[mask]))),
                "r2": float(r2_score(oof_true[mask], oof_pred[mask])),
                "r2_weighted": float(weighted_r2(oof_true[mask], oof_pred[mask], oof_w[mask]))
            }
        }

        # Build averaged scenario rankings from aggregated per-fold predictions
        scenarios_out = []
        for s_key, combo_map in agg_combo_preds.items():
            scen_vals = dict(s_key)
            ranked_list = []
            for combo, preds_list in combo_map.items():
                mean_pred = float(np.median(preds_list))
                ranked_list.append({
                    "combo": ', '.join(combo),
                    "size": len(combo),
                    "pred_auc": mean_pred,
                    "support_n": len(combos[combo]),
                    "support_rows": combos[combo],
                    "score": mean_pred - penalty_lambda * len(combo)
                })
            ranked = sorted(ranked_list, key=lambda d: (-d["score"], -d["pred_auc"], d["size"], d["combo"]))
            scenarios_out.append({"values": scen_vals, "ranked": ranked})

        top_level_ranked = scenarios_out[0]["ranked"] if len(scenarios_out) == 1 else []

        # also fit a final model on ALL data via inner CV for completeness
        if study_ids is not None and len(np.unique(study_ids)) >= 2:
            inner_splitter = GroupKFold(n_splits=min(n_inner_splits, len(np.unique(study_ids))), shuffle=True, random_state=random_state)
            inner_splits = list(inner_splitter.split(X_all, y, groups=study_ids))
        else:
            inner_splitter = KFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)
            inner_splits = list(inner_splitter.split(X_all, y))

        global_best_params, global_best_score = None, np.inf
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            scores = []
            for tr_idx, va_idx in inner_splits:
                tr_idx = np.asarray(tr_idx)
                va_idx = np.asarray(va_idx)
                X_tr2, X_va2 = X_all.iloc[tr_idx], X_all.iloc[va_idx]
                y_tr2, y_va2 = y[tr_idx], y[va_idx]
                w_tr2, w_va2 = sample_weight[tr_idx], sample_weight[va_idx]
                pipe.set_params(**params)
                _fit_pipe(pipe, X_tr2, y_tr2, w_tr2)
                y_hat = np.clip(pipe.predict(X_va2), 0.0, 1.0)
                scores.append(weighted_mse(y_va2, y_hat, w_va2))
            mean_score = float(np.mean(scores))
            if mean_score < global_best_score:
                global_best_score, global_best_params = mean_score, params

        pipe.set_params(**global_best_params)
        _fit_pipe(pipe, X_all, y, sample_weight)
        final_model = pipe

        return {
            "outer_cv": outer_dict,
            "final_model": {"best_params": global_best_params, "estimator": final_model},
            "scenarios": scenarios_out,
            "ranked": top_level_ranked,
            "note": "Evaluation uses nested CV; combination predictions averaged across outer folds."
        }

    else:
        # Skip nested CV entirely → tune on all data, predict combos once with final model
        outer_dict = {}

        # inner CV on all data to select params
        if study_ids is not None and len(np.unique(study_ids)) >= 2:
            inner_splitter = GroupKFold(n_splits=min(n_inner_splits, len(np.unique(study_ids))), shuffle=True, random_state=random_state)
            inner_splits = list(inner_splitter.split(X_all, y, groups=study_ids))
        else:
            inner_splitter = KFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)
            inner_splits = list(inner_splitter.split(X_all, y))

        global_best_params, global_best_score = None, np.inf
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            scores = []
            for tr_idx, va_idx in inner_splits:
                X_tr2, X_va2 = X_all.iloc[tr_idx], X_all.iloc[va_idx]
                y_tr2, y_va2 = y[tr_idx], y[va_idx]
                w_tr2, w_va2 = sample_weight[tr_idx], sample_weight[va_idx]
                pipe.set_params(**params)
                _fit_pipe(pipe, X_tr2, y_tr2, w_tr2)
                y_hat = np.clip(pipe.predict(X_va2), 0.0, 1.0)
                scores.append(weighted_mse(y_va2, y_hat, w_va2))
            mean_score = float(np.mean(scores))
            if mean_score < global_best_score:
                global_best_score, global_best_params = mean_score, params

        pipe.set_params(**global_best_params)
        _fit_pipe(pipe, X_all, y, sample_weight)
        final_model = pipe

        # predict all combos for each scenario with the final model
        scenarios_out = []
        for scen in scenario_assignments:
            hold_vals = hold_base.copy()
            hold_vals.update(scen)
            ranked_list = []
            for combo in combos.keys():
                auc_hat = _predict_combo_row(final_model, combo, hold_vals)
                ranked_list.append({
                    "combo": ', '.join(combo),
                    "size": len(combo),
                    "pred_auc": float(auc_hat),
                    "support_n": len(combos[combo]),
                    "support_rows": combos[combo],
                    "score": float(auc_hat) - penalty_lambda * len(combo)
                })
            ranked = sorted(ranked_list, key=lambda d: (-d["score"], -d["pred_auc"], d["size"], d["combo"]))
            scenarios_out.append({"values": scen, "ranked": ranked})

        top_level_ranked = scenarios_out[0]["ranked"] if len(scenarios_out) == 1 else []

        return {
            "outer_cv": outer_dict,
            "final_model": {"best_params": global_best_params, "estimator": final_model},
            "scenarios": scenarios_out,
            "ranked": top_level_ranked,
            "note": "Nested CV skipped; final model tuned via inner CV on all data."
        }


def evaluate_probast_risk(model):
    # get overall risk according to PROBAST scoring rules
    if all(model[col] == 'low' for col in ['probast.participants.risk', 'probast.outcome.risk',
                                            'probast.predictors.risk', 'probast.analysis.risk']):
        if model['model.validation'] == 'external validation':
            return 'low'
        else:
            # check if there is any validation and the sample size is larger than 500
            if (model['model.validation'] != 'none reported') and (model['n_total'] >= 500):
                return 'low'
            else:
                return 'high'
    if any(model[col] == 'high' for col in ['probast.participants.risk', 'probast.outcome.risk',
                                            'probast.predictors.risk', 'probast.analysis.risk']):
        return 'high'

    if any(model[col] == 'unclear' for col in ['probast.participants.risk', 'probast.outcome.risk',
                                            'probast.predictors.risk', 'probast.analysis.risk']):
        return 'unclear'
