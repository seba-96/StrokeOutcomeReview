"""Publication forest summary of within-study and across-study control analyses.

Requirements: Python >=3.10; numpy, pandas, matplotlib.

Run with the seven original CSV exports in ./data:
    python plot_control_analyses.py --input-dir data --output-dir output

Optional study names (rows expand to fit; no names are silently truncated):
    python plot_control_analyses.py --input-dir data --show-studies

Other options:
    --hide-external   --hide-pvalues   --layout compact   --dpi 600

Importable API:
    from plot_control_analyses import plot_control_analyses, save_figure
    fig = plot_control_analyses("data", bold_reference_label=False)
    save_figure(fig, "output/control_analyses_summary", dpi=600)

The function returns a Matplotlib Figure and does not save or display it.
The manuscript legend is available from get_figure_legend(); no legend is
embedded in the figure by default. Importing the module has no plotting or
file-writing side effects.

Only exported pooled point estimates and 95% confidence limits are plotted.
No meta-analysis is refitted, no AUCs are transformed, and no p values are
recalculated. Across-study delta and heterogeneity columns are not plotted.
The across-study external_validation column is a PERCENTAGE, converted to an
integer study count and checked. Within-study validation counts are derived
from individual-study Val. flags. Missing validation information stays missing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import textwrap
import warnings
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


@dataclass(frozen=True)
class FileSpec:
    filename: str
    framework: str
    analysis: str


FILE_SPECS = (
    FileSpec("forest_post_treatment_nested.csv", "within", "post"),
    FileSpec("forest_restricted_nested.csv", "within", "restricted"),
    FileSpec("forest_severity_nested.csv", "within", "severity"),
    FileSpec("forest_pre_treatment_across.csv", "across", "pre"),
    FileSpec("forest_post_treatment_across.csv", "across", "post"),
    FileSpec("forest_restricted_across.csv", "across", "restricted"),
    FileSpec("forest_severity_across.csv", "across", "severity"),
)

CATEGORY_LABELS = {
    "Clinical_status": "Routine-care reference",
    "Lesion_volume": "Lesion volume",
    "Lesion_location": "Lesion location",
    "Radiomics": "Lesion radiomics",
    "Brain_health": "Brain health",
    "Neural_network": "Whole-brain images",
}
CATEGORY_ORDER = tuple(CATEGORY_LABELS)
ANALYSIS_ORDER = ("pre", "post", "restricted", "mild", "moderate", "severe")
ANALYSIS_LABELS = {
    "pre": "Before reperfusion",
    "post": "After reperfusion",
    "restricted": "Restricted endpoint and uncertainty reporting",
    "mild": "Mild stroke",
    "moderate": "Moderate stroke",
    "severe": "Severe stroke",
}
DEFAULT_COLORS = {
    "pre": "#436B99",
    "post": "#247A82",
    "restricted": "#846499",
    "mild": "#47874E",
    "moderate": "#BE781E",
    "severe": "#3C74A6",
}
REQUIRED = ("category", "estimate", "ci_0.025", "ci_0.975", "count", "studies")


def _study_names(value) -> list[str]:
    if pd.isna(value):
        return []
    return [re.sub(r"\s+", " ", s).strip() for s in str(value).split(";") if s.strip()]


def _validation_flag(value) -> Optional[bool]:
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s in {"yes", "true", "1", "1.0", "external"}:
        return True
    if s in {"no", "false", "0", "0.0", "internal", "derivation"}:
        return False
    return None


def _matching_study_rows(raw, category, names):
    if "Study" not in raw or "row_type" not in raw:
        return raw.iloc[0:0].copy()
    rows = raw.loc[raw["row_type"].eq("study") & raw["category"].eq(category)].copy()
    rows["_study_id"] = rows["Study"].astype(str).map(lambda s: re.sub(r"\s+", " ", s).strip())
    return rows.loc[rows["_study_id"].isin(names)]


def _external_from_study_rows(rows, names) -> Optional[int]:
    if "Val." not in rows or set(rows.get("_study_id", [])) != set(names):
        return None
    flags = []
    for name in names:
        values = {_validation_flag(x) for x in rows.loc[rows["_study_id"].eq(name), "Val."]}
        if len(values) != 1:
            raise ValueError(f"Conflicting validation flags for {name!r}.")
        flag = values.pop()
        if flag is None:
            return None
        flags.append(flag)
    return int(sum(flags))


def load_pooled_estimates(
    input_dir: str | Path,
    file_specs: Sequence[FileSpec] = FILE_SPECS,
) -> pd.DataFrame:
    """Extract one row per exported pooled estimate, retaining full provenance.

    Severity-across exports label their pooled rows as 'study', so selection
    uses finite estimates, finite CI limits and a non-missing study count.
    Annotation rows have missing estimates; individual studies lack counts.
    Missing severity labels in nested summaries are inferred only when ALL
    contributing study rows have the same non-missing severity.
    """
    input_dir = Path(input_dir)
    records = []
    for spec in file_specs:
        path = input_dir / spec.filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing required export: {path}")
        raw = pd.read_csv(path)
        missing = set(REQUIRED) - set(raw.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        for col in ("estimate", "ci_0.025", "ci_0.975", "count"):
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        potential = np.isfinite(raw["estimate"]) & raw["count"].gt(0)
        if (potential & ~np.isfinite(raw[["ci_0.025", "ci_0.975"]]).all(axis=1)).any():
            raise ValueError(f"{path.name}: a pooled estimate is missing a confidence limit.")
        valid = np.isfinite(raw[["estimate", "ci_0.025", "ci_0.975", "count"]]).all(axis=1)
        summaries = raw.loc[valid & raw["count"].gt(0)]
        if summaries.empty:
            warnings.warn(f"No pooled estimates found in {path.name}.", stacklevel=2)
        for index, row in summaries.iterrows():
            category = str(row["category"])
            if not float(row["count"]).is_integer():
                raise ValueError(f"{path.name}, row {index + 2}: non-integer study count.")
            k = int(row["count"])
            names = _study_names(row["studies"])
            if len(names) != k or len(set(names)) != k:
                raise ValueError(f"{path.name}, {category}: study list does not match k={k}.")
            estimate, low, high = map(float, row[["estimate", "ci_0.025", "ci_0.975"]])
            if not low <= estimate <= high:
                raise ValueError(f"{path.name}, {category}: estimate lies outside its CI.")
            domain = (0, 1) if spec.framework == "across" else (-1, 1)
            if low < domain[0] or high > domain[1]:
                raise ValueError(f"{path.name}, {category}: CI is outside the {spec.framework} AUC domain.")
            detail = _matching_study_rows(raw, category, names)
            analysis = spec.analysis
            if analysis == "severity":
                severity = row.get("Severity")
                if pd.notna(severity):
                    analysis = str(severity).strip().lower()
                else:
                    severity_values = detail.get("Severity", pd.Series(dtype=object))
                    available = {str(v).strip().lower() for v in severity_values.dropna()}
                    if len(available) != 1 or severity_values.isna().any() or len(detail) != k:
                        raise ValueError(f"{path.name}, {category}: cannot infer a unique severity stratum.")
                    analysis = available.pop()
                if analysis not in {"mild", "moderate", "severe"}:
                    raise ValueError(f"Unknown severity stratum: {analysis!r}")
            from_details = _external_from_study_rows(detail, names)
            percentage = row.get("external_validation", np.nan)
            if spec.framework == "across" and pd.notna(percentage):
                percentage = float(percentage)
                if not 0 <= percentage <= 100:
                    raise ValueError(f"Invalid external-validation percentage: {percentage}")
                candidate = k * percentage / 100
                if not np.isclose(candidate, round(candidate), atol=1e-5, rtol=0):
                    raise ValueError(f"{path.name}, {category}: percentage does not imply an integer count.")
                external = int(round(candidate))
                ext_source = "exported percentage × study count / 100"
                if from_details is not None and from_details != external:
                    raise ValueError(f"{path.name}, {category}: validation metadata disagree.")
            else:
                external = from_details
                ext_source = "individual-study Val. flags" if external is not None else "not reported"
            records.append({
                "framework": spec.framework, "analysis": analysis, "category": category,
                "estimate": estimate, "ci_low": low, "ci_high": high,
                "k": k, "k_external": external,
                "p_value": pd.to_numeric(row.get("p-value", np.nan), errors="coerce"),
                "studies": "; ".join(names), "validation_count_source": ext_source,
                "source_file": path.name, "source_csv_row": int(index + 2),
            })
    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise ValueError("No pooled estimates were extracted.")
    key = ["framework", "analysis", "category"]
    if result.duplicated(key).any():
        raise ValueError("More than one pooled estimate for a framework/analysis/category; check the exports.")
    result["k_external"] = pd.array(result["k_external"], dtype="Int64")
    return result


def _fmt_number(value, decimals):
    # Use a typographic minus sign; avoid negative zero after rounding.
    value = 0.0 if abs(value) < 0.5 * 10 ** (-decimals) else value
    return f"{value:.{decimals}f}".replace("-", "−")


def _fmt_ci(row, decimals):
    f = lambda x: _fmt_number(x, decimals)
    return f"{f(row['estimate'])} ({f(row['ci_low'])}–{f(row['ci_high'])})"


def _fmt_p(value):
    return "—" if pd.isna(value) else ("<0.001" if value < 0.001 else f"{value:.3f}")


def _wrapped(text, width_in, fontsize, *, shorten_names=False):
    if shorten_names:
        text = re.sub(r"\s+et\s+al\.?", "", text)
    chars = max(8, int(width_in * 72 / (fontsize * 0.51)))
    return textwrap.fill(str(text), width=chars, break_long_words=False, break_on_hyphens=False)


def _panel_plan(data, width, *, framework, fontsize, row_height, show_external,
                show_studies, show_pvalues, study_width, labels, category_order,
                analysis_order, analysis_labels):
    within = framework == "within"
    p_visible = show_pvalues
    compact = width < 4.5
    label_width = 1.65 if not compact else 1.03
    columns = {"label": (0.025, label_width)}
    x = label_width + 0.08
    columns["k"] = (x, 0.29 if not compact else 0.21)
    x += columns["k"][1]
    if show_external:
        columns["external"] = (x, 0.35 if not compact else 0.25)
        x += columns["external"][1]
    ci_width = (1.48 if within else 1.14) * (fontsize / 8.2)
    p_width = 0.46 if p_visible else 0
    right_reserved = ci_width + p_width + (study_width + 0.12 if show_studies else 0) + 0.04
    plot_start = x + 0.10
    plot_width = width - plot_start - right_reserved - 0.10
    if plot_width < 0.62:
        raise ValueError("Figure is too narrow for the requested columns; increase width or hide a column.")
    columns["forest"] = (plot_start, plot_width)
    x = plot_start + plot_width + 0.10
    columns["ci"] = (x, ci_width)
    x += ci_width
    if p_visible:
        columns["p"] = (x, p_width)
        x += p_width
    if show_studies:
        columns["studies"] = (x + 0.12, study_width)

    groups = []
    y = 0.44  # panel title and column headings, in inches
    a_rank = {v:i for i,v in enumerate(analysis_order)}
    c_rank = {v:i for i,v in enumerate(category_order)}
    analysis_names = sorted(data["analysis"].unique(), key=lambda v:(a_rank.get(v,999),v))
    for analysis in analysis_names:
        rows = data.loc[data["analysis"].eq(analysis)].copy()
        rows["_rank"] = rows["category"].map(c_rank).fillna(999)
        rows = rows.sort_values(["_rank", "category"])
        group = {"analysis": analysis, "label": analysis_labels.get(analysis,analysis),
                 "top": y, "header_center": y + 0.10, "rows": []}
        y += 0.20
        for _, record in rows.iterrows():
            label = _wrapped(labels.get(record["category"],record["category"]), label_width, fontsize)
            studies = _wrapped(record["studies"],study_width,fontsize-0.7,shorten_names=True) if show_studies else ""
            lines = max(label.count("\n") + 1, studies.count("\n") + 1)
            h = max(row_height, lines * fontsize * 1.18 / 72 + 0.035)
            group["rows"].append({"record":record, "label":label, "studies":studies,
                                   "top":y, "center":y+h/2, "height":h})
            y += h
        group["bottom"] = y
        groups.append(group)
        y += 0.04
    axis_y = y + 0.025
    return {"groups":groups,"columns":columns,"width":width,
            "height":axis_y+0.36,"axis_y":axis_y,"framework":framework}


def _draw_panel(fig, plan, left, top, *, title, limits, fontsize, colors,
                decimals, reference_lines, font_family, bold_reference_label):
    figw, figh = fig.get_size_inches()
    width, height = plan["width"], plan["height"]
    ax = fig.add_axes([left/figw, (figh-top-height)/figh, width/figw, height/figh])
    ax.set_xlim(0,width)
    ax.set_ylim(height,0)
    ax.set_axis_off()
    ink, muted, rule = "#20262E", "#606974", "#D9DFE4"
    cols = plan["columns"]
    fx, fw = cols["forest"]
    lo, hi = limits
    to_x = lambda value: fx + (value-lo)/(hi-lo)*fw

    def txt(x,y,value,**kwargs):
        options=dict(fontsize=fontsize,color=ink,fontfamily=font_family,va="center",clip_on=False)
        if "size" in kwargs:
            kwargs["fontsize"]=kwargs.pop("size")
        options.update(kwargs)
        return ax.text(x,y,value,**options)

    txt(0.025,0.065,title,fontweight="bold",size=fontsize+1.5)
    heads = {"label":"Added imaging" if plan["framework"]=="within" else "Model",
             "k":"k", "external":"Val.", "forest":"Pooled estimate" if width>=4.5 else "Estimate",
             "ci":("ΔAUC" if plan["framework"]=="within" else "AUC")+" (95% CI)",
             "p":"p", "studies":"Contributing studies"}
    for key,(x,w) in cols.items():
        centered = key in {"k","external","forest","p"}
        txt(x+w/2 if centered else x,0.285,heads[key],fontweight="bold",
            ha="center" if centered else "left",size=fontsize-0.2)
    ax.plot([0,width],[0.39,0.39],color="#75818D",lw=0.7)

    if plan["framework"] == "within":
        ticks = MaxNLocator(nbins=6,steps=[1,2,5,10]).tick_values(lo,hi)
    else:
        ticks = np.arange(0,1.0001,0.2 if fw < 1.3 else 0.1)
    ticks = [t for t in ticks if lo-1e-10 <= t <= hi+1e-10]
    for tick in ticks:
        x = to_x(tick)
        ax.plot([x,x],[0.42,plan["axis_y"]],color="#E4E8EC",lw=0.55,zorder=0)
        ax.plot([x,x],[plan["axis_y"],plan["axis_y"]+0.035],color=muted,lw=0.7)
        ticktext = _fmt_number(tick,2 if plan["framework"]=="within" else 1)
        txt(x,plan["axis_y"]+0.105,ticktext,ha="center",size=fontsize-0.4,color=muted)
    if plan["framework"]=="within" and lo<=0<=hi:
        ax.plot([to_x(0)]*2,[0.42,plan["axis_y"]],color="#727D86",ls=(0,(3,2)),lw=0.8,zorder=1)

    for group in plan["groups"]:
        color = colors.get(group["analysis"],"#247A82")
        ax.add_patch(plt.Rectangle((0,group["top"]),width,0.195,color="#F0F3F5",lw=0,zorder=2))
        ax.add_patch(plt.Rectangle((0,group["top"]),0.025,0.195,color=color,lw=0,zorder=3))
        txt(0.085,group["header_center"],group["label"],fontweight="bold",size=fontsize+0.05)
        references = [item["record"] for item in group["rows"] if item["record"]["category"]=="Clinical_status"]
        if reference_lines and plan["framework"]=="across" and references:
            x = to_x(references[0]["estimate"])
            ax.plot([x,x],[group["top"]+0.20,group["bottom"]],color="#A5AEB6",lw=0.65,ls=(0,(2,2)),zorder=1)
        for item in group["rows"]:
            row,y = item["record"],item["center"]
            is_reference = row["category"]=="Clinical_status"
            weight = "bold" if is_reference else "normal"
            marker_color = "#37414B" if is_reference else color
            label_weight = "bold" if is_reference and bold_reference_label else "normal"
            txt(cols["label"][0]+0.07,y,item["label"],fontweight=label_weight,linespacing=1.12)
            txt(sum((cols["k"][0],cols["k"][1]/2)),y,str(row["k"]),ha="center")
            if "external" in cols:
                x,w = cols["external"]
                txt(x+w/2,y,"—" if pd.isna(row["k_external"]) else str(int(row["k_external"])),ha="center")
            ax.plot([to_x(row["ci_low"]),to_x(row["ci_high"])],[y,y],color=marker_color,lw=1.10,solid_capstyle="round",zorder=4)
            ax.plot(to_x(row["estimate"]),y,marker="D",ms=4.6,mec=marker_color,mew=0.55,mfc=marker_color,zorder=5)
            txt(cols["ci"][0],y,_fmt_ci(row,decimals),fontweight=weight,size=fontsize-0.2)
            if "p" in cols:
                x,w = cols["p"]
                # Reference-row p values test the model intercept, not a
                # category-versus-reference contrast. Keep them in the data
                # table for provenance, but do not present them as comparisons.
                p_text = "—" if is_reference and plan["framework"] == "across" else _fmt_p(row["p_value"])
                txt(x+w/2,y,p_text,ha="center",size=fontsize-0.2)
            if "studies" in cols:
                txt(cols["studies"][0],y,item["studies"],size=fontsize-0.7,linespacing=1.15,color=muted)
            if item is not group["rows"][-1]:
                ax.plot([0,width],[item["top"]+item["height"]]*2,color="#EBEEF1",lw=0.35,zorder=0)
        ax.plot([0,width],[group["bottom"]]*2,color=rule,lw=0.6,zorder=0)
    ax.plot([fx,fx+fw],[plan["axis_y"]]*2,color=muted,lw=0.75)
    xlabel = "ΔAUC after adding imaging" if plan["framework"]=="within" else "AUC"
    txt(fx+fw/2,plan["axis_y"]+0.265,xlabel,ha="center",size=fontsize+0.2)
    ax._control_summary_plan = plan
    return ax


def plot_control_summary(
    pooled: pd.DataFrame,
    *,
    layout: str = "stacked",
    show_external: bool = True,
    show_studies: bool = False,
    show_pvalues: bool = True,
    bold_reference_label: bool = True,
    width: Optional[float] = None,
    fontsize: float = 8.2,
    row_height: float = 0.17,
    study_width: float = 3.0,
    font_family: str = "DejaVu Sans",
    within_limits: Optional[tuple[float,float]] = None,
    across_limits: tuple[float,float] = (0.58,1.00),
    within_decimals: int = 3,
    across_decimals: int = 2,
    colors: Optional[Mapping[str,str]] = None,
    category_labels: Optional[Mapping[str,str]] = None,
    category_order: Sequence[str] = CATEGORY_ORDER,
    analysis_order: Sequence[str] = ANALYSIS_ORDER,
    analysis_labels: Optional[Mapping[str,str]] = None,
    reference_lines: bool = True,
    show_notes: bool = False,
) -> plt.Figure:
    """Return one figure with separate scales for the two analytical frameworks.

    Stacked (recommended): A within-study ΔAUC, B across-study AUC; same readable
    full-width table format as a conventional forest plot. Compact: within-
    study panel above two across-study panels (timing/restriction and severity).
    The optional study-name column is best treated as supplementary material:
    all names are retained and rows grow automatically. It uses stacked layout.
    k counts contributing studies; Val. counts externally validated studies.
    show_pvalues controls both panels. Within-study p values concern ΔAUC;
    across-study p values concern category-versus-reference comparisons.
    Reference rows display a dash rather than the exported intercept p value.
    bold_reference_label controls only the Routine-care reference text label.
    No explanatory legend is drawn unless show_notes=True is explicitly set.
    """
    if layout not in {"stacked","compact"}:
        raise ValueError("layout must be 'stacked' or 'compact'.")
    if fontsize<=0 or row_height<=0 or study_width<=0:
        raise ValueError("Font size and dimensions must be positive.")
    if show_studies and layout=="compact":
        warnings.warn("Using stacked layout to retain all study names legibly.",stacklevel=2)
        layout="stacked"
    if width is None:
        width = 10.9 if show_studies else (9.2 if layout=="compact" and show_pvalues else 8.0 if layout=="compact" else 7.48)
    labels = dict(CATEGORY_LABELS); labels.update(category_labels or {})
    alabels = dict(ANALYSIS_LABELS); alabels.update(analysis_labels or {})
    palette = dict(DEFAULT_COLORS); palette.update(colors or {})
    within = pooled.loc[pooled["framework"].eq("within")]
    across = pooled.loc[pooled["framework"].eq("across")]
    if within.empty or across.empty:
        raise ValueError("The combined figure requires both within-study and across-study estimates.")
    if within_limits is None:
        minimum = min(0.0,float(within["ci_low"].min()))
        maximum = max(0.0,float(within["ci_high"].max()))
        within_limits = (np.floor((minimum-0.005)*100)/100,np.ceil((maximum+0.005)*100)/100)
    for frame,limits,label in [(within,within_limits,"within"),(across,across_limits,"across")]:
        if limits[0]>=limits[1]:
            raise ValueError(f"Invalid {label}-study axis limits.")
        if frame["ci_low"].min()<limits[0] or frame["ci_high"].max()>limits[1]:
            raise ValueError(f"{label}-study limits would clip a confidence interval; widen them.")
    margin, gap = 0.18, 0.16
    full_width = width-2*margin
    common = dict(fontsize=fontsize,row_height=row_height,show_external=show_external,
                  show_studies=show_studies,show_pvalues=show_pvalues,study_width=study_width,
                  labels=labels,category_order=category_order,analysis_order=analysis_order,
                  analysis_labels=alabels)
    pa = _panel_plan(within,full_width,framework="within",**common)
    if layout=="stacked":
        pb = _panel_plan(across,full_width,framework="across",**common)
        specs = [(pa,margin,0.12,"A  Within-study incremental value",within_limits,within_decimals),
                 (pb,margin,0.12+pa["height"]+gap,"B  Across-study discrimination",across_limits,across_decimals)]
        bottom = specs[-1][2]+pb["height"]
    else:
        half=(full_width-gap)/2
        timing=across.loc[across["analysis"].isin(["pre","post","restricted"])]
        severity=across.loc[across["analysis"].isin(["mild","moderate","severe"])]
        small_common=dict(common)
        small_common["analysis_labels"]=dict(alabels,restricted="Restricted analysis")
        pb=_panel_plan(timing,half,framework="across",**small_common)
        pc=_panel_plan(severity,half,framework="across",**small_common)
        top=0.12+pa["height"]+gap
        specs=[(pa,margin,0.12,"A  Within-study incremental value",within_limits,within_decimals),
               (pb,margin,top,"B  Across studies: timing and restriction",across_limits,across_decimals),
               (pc,margin+half+gap,top,"C  Across studies: severity",across_limits,across_decimals)]
        bottom=top+max(pb["height"],pc["height"])
    notes = ["Diamonds: pooled estimates; lines: 95% CIs. k: studies" + ("; Val.: externally validated studies." if show_external else "."),
             "Restricted: three-month mRS 0–2 vs 3–6 plus reported paired-AUC correlations (A) or AUC variances (B"+("–C)." if layout=="compact" else ")."),
             "Only available pooled estimates are shown; studies can recur across analyses and imaging categories."]
    if reference_lines:
        notes.append("Dashed lines: no increment (A) or the subgroup reference-model AUC ("+("B–C" if layout=="compact" else "B")+").")
    if show_external and pooled["k_external"].isna().any():
        notes.append("—: external-validation information is unavailable.")
    note_font=fontsize-1.0
    note_text="\n".join(_wrapped(n,full_width,note_font) for n in notes)
    note_h=(note_text.count("\n")+1)*note_font*1.28/72+0.08 if show_notes else 0
    height=bottom+note_h+0.12
    with plt.rc_context({"font.family":font_family,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"}):
        fig=plt.figure(figsize=(width,height),facecolor="white")
        for plan,left,top,title,limits,decimals in specs:
            _draw_panel(fig,plan,left,top,title=title,limits=limits,fontsize=fontsize,colors=palette,
                        decimals=decimals,reference_lines=reference_lines,font_family=font_family,
                        bold_reference_label=bold_reference_label)
        if show_notes:
            fig.text(margin/width,(height-bottom-0.05)/height,note_text,ha="left",va="top",
                     fontsize=note_font,fontfamily=font_family,color="#4D5862",linespacing=1.28)
    return fig


def plot_control_analyses(
    input_dir: str | Path,
    *,
    file_specs: Sequence[FileSpec] = FILE_SPECS,
    **plot_kwargs,
) -> plt.Figure:
    """Load the control-analysis CSVs and return a publication-ready Figure.

    Parameters
    ----------
    input_dir : str or pathlib.Path
        Directory containing the seven original CSV exports.
    file_specs : sequence of FileSpec
        Optional file mapping; defaults to the supplied export filenames.
    **plot_kwargs
        Keyword options accepted by plot_control_summary, including
        show_external=True, show_studies=False, show_pvalues=True,
        bold_reference_label=True, show_notes=False, and layout='stacked'.

    Returns
    -------
    matplotlib.figure.Figure
        No saving or display occurs. Use save_figure(fig, output_stem) to
        export, or matplotlib.pyplot.show() to display in an interactive
        session. For direct access to the extracted table, call
        load_pooled_estimates followed by plot_control_summary instead.

    Example
    -------
    >>> fig = plot_control_analyses('data', bold_reference_label=False)
    >>> save_figure(fig, 'output/control_analyses_summary')
    """
    return plot_control_summary(load_pooled_estimates(input_dir, file_specs), **plot_kwargs)


def get_figure_legend(layout: str = "stacked") -> str:
    """Return manuscript legend text for the default columns and reference lines.

    Use 'stacked' for the study-name variant. Adapt the column definitions if
    optional columns or reference lines are hidden.
    """
    if layout not in {"stacked", "compact"}:
        raise ValueError("layout must be 'stacked' or 'compact'.")
    panel_b = "B" if layout == "stacked" else "B–C"
    across_description = (
        "(B) Across-study pooled AUCs for routine-care reference models and models incorporating quantitative imaging. "
        if layout == "stacked" else
        "(B–C) Across-study pooled AUCs, grouped by prediction timing and joint restriction (B), and study-level stroke severity (C). "
    )
    return (
        "Pooled estimates from control analyses. "
        "(A) Within-study changes in area under the receiver operating characteristic curve (ΔAUC) after adding quantitative imaging. "
        + across_description +
        "Diamonds indicate pooled estimates and horizontal lines indicate 95% confidence intervals; marker size does not encode study weight. "
        f"Dashed lines indicate no increment (A) or the corresponding pooled routine-care reference AUC ({panel_b}). "
        "Restricted analyses jointly required prediction of three-month modified Rankin Scale (mRS) 0–2 versus 3–6 and reported paired-AUC correlations for within-study analyses or AUC variances for across-study analyses. "
        "Severity strata use study-level mean or median admission National Institutes of Health Stroke Scale scores: mild, 0–4; moderate, 5–15; severe, 16–42. "
        "P values refer to within-study increments (A) or meta-regression comparisons with the routine-care reference category "
        f"({panel_b}); a dash in the p-value column denotes the reference category. "
        "Across-study comparisons do not represent within-study incremental effects. "
        "Only available pooled estimates are shown, and studies may contribute to multiple analyses and categories. "
        "k, contributing studies; Val., externally validated studies."
    )


def save_figure(fig: plt.Figure, output_stem: str | Path, dpi: int = 600,
                formats: Sequence[str] = ("pdf","svg","png")) -> list[Path]:
    """Save vector PDF/SVG and high-resolution PNG with a white background."""
    stem=Path(output_stem)
    stem.parent.mkdir(parents=True,exist_ok=True)
    outputs=[]
    with matplotlib.rc_context({"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"}):
        for extension in formats:
            path=stem.with_suffix("."+extension)
            fig.savefig(path,dpi=dpi,facecolor="white",edgecolor="none")
            outputs.append(path)
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir",type=Path,default=Path("data"))
    parser.add_argument("--output-dir",type=Path,default=Path("output"))
    parser.add_argument("--stem",default="control_analyses_summary")
    parser.add_argument("--layout",choices=["stacked","compact"],default="stacked")
    parser.add_argument("--show-studies",action="store_true")
    parser.add_argument("--hide-external",action="store_true")
    parser.add_argument("--hide-pvalues",action="store_true")
    parser.add_argument("--plain-reference-label",action="store_true")
    parser.add_argument("--width",type=float,default=None)
    parser.add_argument("--font-size",type=float,default=8.2)
    parser.add_argument("--row-height",type=float,default=0.17)
    parser.add_argument("--dpi",type=int,default=600)
    parser.add_argument("--formats",nargs="+",choices=["pdf","svg","png"],default=["pdf","svg","png"])
    args=parser.parse_args()
    pooled=load_pooled_estimates(args.input_dir)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pooled.to_csv(args.output_dir/"pooled_estimates_used.csv",index=False)
    fig=plot_control_summary(pooled,layout=args.layout,show_studies=args.show_studies,
                             show_external=not args.hide_external,show_pvalues=not args.hide_pvalues,
                             bold_reference_label=not args.plain_reference_label,
                             width=args.width,fontsize=args.font_size,row_height=args.row_height)
    actual_layout = "stacked" if args.show_studies else args.layout
    (args.output_dir/"control_analyses_legend.txt").write_text(get_figure_legend(actual_layout)+"\n",encoding="utf-8")
    paths=save_figure(fig,args.output_dir/args.stem,dpi=args.dpi,formats=args.formats)
    plt.close(fig)
    print(f"Plotted {len(pooled)} pooled estimates: " + str(pooled.groupby('framework').size().to_dict()))
    for path in paths:
        print(path.resolve())


if __name__=="__main__":
    main()
