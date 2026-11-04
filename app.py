"""
ACTG 175 - 24-month event prediction: interactive app for the semestral project.

Run:   streamlit run app.py

Pages
  1. Explore data   - pick a feature (or the target) -> summary, distribution, relation to the outcome, correlations
  2. Modelling      - pick Model A / B / C, choose baseline features, set hyperparameters, train, predict a patient
  3. Results        - the three tuned models (hyperparameters pre-computed in the notebook) compared

Everything (label definition, predictors, split, pipelines) mirrors aids.ipynb.
Data: downloaded from UCI (ucimlrepo id=890). If you have no internet, put the CSV next to
app.py as "actg175.csv" or upload it in the app.
"""
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from matplotlib.patches import Patch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             log_loss, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, SplineTransformer,
                                   StandardScaler)

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------------
# Branding - edit these two lines to change the title shown on every page
# ----------------------------------------------------------------------------------------
PROJECT_TITLE = "ACTG 175: Predicting 24-month clinical events in HIV treatment"
LOGO_TEXT = "175"                         # short text shown in the logo square
PROJECT_SUBTITLE = "Semestral project · binary classification · elastic net, gradient boosting and spline logistic models"

# ----------------------------------------------------------------------------------------
# Constants (from the notebook)
# ----------------------------------------------------------------------------------------
H = 730                                   # 24 months in days
SEED = 42
LOCAL_CSV = Path(__file__).with_name("actg175.csv")

LOG_COLS = ["cd80"]
NUM_COLS = ["age", "wtkg", "karnof", "cd40", "preanti"]
BIN_COLS = ["hemo", "homo", "drugs", "oprior", "z30", "race", "gender", "symptom"]
CAT_COLS = ["trt", "strat"]
FEATURES = LOG_COLS + NUM_COLS + BIN_COLS + CAT_COLS          # the 16 baseline predictors of the notebook
BASELINE_UNUSED = ["zprior", "str2", "treat"]                 # baseline columns the notebook left out
POST_BASELINE = ["time", "cid", "cd420", "cd820", "offtrt"]
THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40]

# Hyperparameters found by the grid searches in the notebook (cells 41, 69, 79; selection by log loss)
TUNED = {
    "A": {"C": 0.31622776601683794, "l1_ratio": 1.0},
    "B": {"max_depth": 2, "learning_rate": 0.2, "n_estimators": 25},
    "C": {"C": 1.0},
}
SEARCH_SPACE = {
    "A": "C = 0.01 ... 1 (9 log-spaced values) × l1_ratio ∈ {0, 0.25, 0.5, 0.75, 1}; repeated 5-fold CV (3 repeats)",
    "B": "max_depth ∈ {1, 2, 3, 4} × learning_rate ∈ {0.05, 0.1, 0.2, 0.3} × n_estimators ∈ {10, 25, 50, 100}; repeated 5-fold CV (3 repeats)",
    "C": "C = 0.001 ... 100 (11 log-spaced values); 5-fold CV",
}
TUNED_CV_LOGLOSS = {"A": 0.4296, "B": 0.4378, "C": 0.4326}
# Best grid candidates as printed in the notebook
TOP_GRID = {
    "A": pd.DataFrame({"C": [0.316, 0.316, 0.562, 0.316, 0.562, 0.178, 0.178, 0.316],
                       "l1_ratio": [1.0, 0.75, 1.0, 0.5, 0.75, 0.75, 0.5, 0.25],
                       "log loss": [0.4296, 0.4298, 0.4299, 0.4301, 0.4302, 0.4304, 0.4304, 0.4305],
                       "AUC": [0.7163, 0.7157, 0.7160, 0.7150, 0.7151, 0.7152, 0.7142, 0.7135],
                       "Brier": [0.1342, 0.1343, 0.1343, 0.1343, 0.1344, 0.1345, 0.1344, 0.1344]}),
    "B": pd.DataFrame({"max_depth": [2, 2, 2, 2, 1, 2, 1, 1],
                       "n_estimators": [25, 50, 100, 50, 50, 100, 100, 100],
                       "learning_rate": [0.2, 0.2, 0.1, 0.1, 0.3, 0.2, 0.2, 0.3],
                       "log loss": [0.4378, 0.4380, 0.4383, 0.4383, 0.4391, 0.4391, 0.4396, 0.4396],
                       "AUC": [0.6873, 0.6906, 0.6888, 0.6877, 0.6855, 0.6900, 0.6852, 0.6855],
                       "Brier": [0.1368, 0.1370, 0.1367, 0.1367, 0.1369, 0.1373, 0.1371, 0.1370]}),
    "C": pd.DataFrame({"C": [1.0, 0.3162, 3.1623, 10.0, 31.6228, 0.1, 100.0, 0.0316],
                       "log loss": [0.4326, 0.4333, 0.4342, 0.4365, 0.4394, 0.4400, 0.4421, 0.4512]}),
}

TRT = {0: "ZDV only", 1: "ZDV + ddI", 2: "ZDV + ddC", 3: "ddI only"}
STRAT = {1: "Antiretroviral naive", 2: ">1 but <=52 weeks prior therapy", 3: ">52 weeks prior therapy"}
VALUE_LABELS = {
    "trt": TRT, "strat": STRAT,
    "gender": {0: "Female", 1: "Male"},
    "race": {0: "White", 1: "Non-white"},
    "symptom": {0: "Asymptomatic", 1: "Symptomatic"},
    "y24": {0: "No event by 24 mo", 1: "Event by 24 mo"},
    "cid": {0: "Censored", 1: "Event"},
}
DESC = {
    "y24": "TARGET: event within 730 days (1) / event-free for 730 days (0)",
    "age": "Age at baseline (years)", "wtkg": "Weight at baseline (kg)",
    "karnof": "Karnofsky performance score (70-100)", "cd40": "CD4 count at baseline (cells/mm3)",
    "cd80": "CD8 count at baseline (cells/mm3)", "preanti": "Days of prior antiretroviral therapy",
    "hemo": "Hemophilia (0/1)", "homo": "Homosexual activity (0/1)", "drugs": "History of IV drug use (0/1)",
    "oprior": "Non-ZDV antiretroviral therapy before study (0/1)", "z30": "ZDV in the 30 days before study (0/1)",
    "zprior": "ZDV before study (0/1)", "race": "0 = white, 1 = non-white", "gender": "0 = female, 1 = male",
    "str2": "Antiretroviral history (0 = naive, 1 = experienced)", "strat": "Antiretroviral history stratum (1/2/3)",
    "symptom": "Symptomatic indicator (0/1)", "treat": "Treatment indicator (0 = ZDV only, 1 = other)",
    "trt": "Treatment arm (0 ZDV, 1 ZDV+ddI, 2 ZDV+ddC, 3 ddI)", "offtrt": "Off treatment before 96 weeks (0/1)",
    "time": "Days to event / censoring", "cid": "Event indicator (1 = event, 0 = censored)",
    "cd420": "CD4 at ~20 weeks (post-baseline)", "cd820": "CD8 at ~20 weeks (post-baseline)",
}

# Feature groups (used for colours, badges and filters)
# One accent (dark navy) + its tints + greys. Groups are always also named in text, never colour alone.
ACCENT = "#1b2a4e"
GROUP_COLOR = {"target": ACCENT, "baseline": "#5b7bb5", "baseline_other": "#aebbd6", "leakage": "#8b919c"}
GROUP_SHORT = {"target": "Target", "baseline": "Baseline", "baseline_other": "Baseline · not used",
               "leakage": "Leakage"}
GROUP_TEXT = {
    "target": "y24 - event within 730 days (what the models predict)",
    "baseline": "Known at the start of the trial · used as a predictor in Models A, B, C",
    "baseline_other": "Baseline columns left out of the models (they overlap with other columns)",
    "leakage": "Measured after the start or part of the outcome · must not be used as predictors",
}


def group_of(col):
    if col == "y24":
        return "target"
    if col in FEATURES:
        return "baseline"
    if col in BASELINE_UNUSED:
        return "baseline_other"
    return "leakage"


# colour per model: one hue (navy) in three lightness steps, also told apart by line style and marker
MODEL_COLOR = {"A": ACCENT, "B": "#5a6f9e", "C": "#8b95ad"}      # navy -> slate -> light slate (+ line style/marker)
MODEL_STYLE = {"A": "-", "B": "--", "C": ":"}
MODEL_MARKER = {"A": "o", "B": "s", "C": "^"}
MODEL_NAME = {"A": "A - Elastic-net logistic regression",
              "B": "B - Gradient boosting",
              "C": "C - Spline logistic regression (GAM)"}
INK, MUTED, GRID, SURFACE = "#1d2433", "#6b7280", "#e5e7ec", "#ffffff"
LIGHT = "#b8bfcf"                                   # light slate for secondary marks
# treatment arms: grey = control (ZDV only), navy shades = the other arms; lines get a darker tone + a dash style
ARM_COLOR = {0: "#8b919c", 1: ACCENT, 2: "#5b7bb5", 3: "#aebbd6"}
ARM_LINE = {0: "#6f7580", 1: ACCENT, 2: "#5b7bb5", 3: "#7f93bd"}
ARM_STYLE = {0: "--", 1: "-", 2: "-.", 3: ":"}


from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SEQUENTIAL = LinearSegmentedColormap.from_list("navy", ["#f1f2f6", ACCENT])
DIVERGING = LinearSegmentedColormap.from_list("grey_navy", ["#6b7280", "#f4f5f8", ACCENT])


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(w=6.4, h=3.6, **kw):
    fig, ax = plt.subplots(figsize=(w, h), **kw)
    fig.patch.set_facecolor(SURFACE)
    for a in np.atleast_1d(ax).ravel():
        style_ax(a)
    return fig, ax


def lab_val(col, v):
    return VALUE_LABELS.get(col, {}).get(v, str(v))


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Downloading the dataset from UCI ...")
def load_raw():
    if LOCAL_CSV.exists():
        return pd.read_csv(LOCAL_CSV)
    from ucimlrepo import fetch_ucirepo
    return fetch_ucirepo(id=890).data.original


def labelled_cohort(raw):
    """Notebook cell 15: keep event by day 730 (y=1) or followed >= 730 days (y=0)."""
    lab = raw[((raw.cid == 1) & (raw.time <= H)) | (raw.time >= H)].copy()
    lab["y24"] = ((lab.cid == 1) & (lab.time <= H)).astype(int)
    return lab.reset_index(drop=True)


def make_split(lab):
    """Notebook cell 29: 80/20 split stratified on outcome x treatment arm."""
    X, y = lab[FEATURES], lab["y24"]
    strata = y.astype(str) + "_" + lab["trt"].astype(str)
    return train_test_split(X, y, test_size=0.2, stratify=strata, random_state=SEED)


# ----------------------------------------------------------------------------------------
# Models (notebook: A = cells 36/39/41, B = cell 69, C = cell 79) - work with any subset of FEATURES
# ----------------------------------------------------------------------------------------
def split_cols(feats):
    f = set(feats)
    return ([c for c in LOG_COLS if c in f], [c for c in NUM_COLS if c in f],
            [c for c in BIN_COLS if c in f], [c for c in CAT_COLS if c in f])


def _enet(C, l1_ratio):
    # sklearn >= 1.8 deprecates penalty="elasticnet" (l1_ratio alone selects it)
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    kw = {} if (major, minor) >= (1, 8) else {"penalty": "elasticnet"}
    return LogisticRegression(solver="saga", C=C, l1_ratio=l1_ratio, max_iter=10000, **kw)


def make_A(C=0.1, l1_ratio=0.5, feats=FEATURES):
    log, num, binary, cat = split_cols(feats)
    parts = []
    if log:
        parts.append(("log", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                       ("scale", StandardScaler())]), log))
    if num:
        parts.append(("num", StandardScaler(), num))
    if cat:
        parts.append(("cat", OneHotEncoder(drop="first", sparse_output=False), cat))
    if binary:
        parts.append(("binary", "passthrough", binary))
    pre = ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)
    pre.set_output(transform="pandas")
    return Pipeline([("pre", pre), ("clf", _enet(C, l1_ratio))])


def make_B(max_depth=1, n_estimators=300, learning_rate=0.1, feats=FEATURES):
    log, num, binary, cat = split_cols(feats)
    parts = []
    if cat:
        parts.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat))
    parts.append(("keep", "passthrough", log + num + binary))
    pre = ColumnTransformer(parts, remainder="drop")
    return Pipeline([("pre", pre),
                     ("clf", GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators,
                                                        learning_rate=learning_rate,
                                                        n_iter_no_change=10, random_state=SEED))])


def make_C(C=1.0, feats=FEATURES):
    log, num, binary, cat = split_cols(feats)
    parts = []
    if log:
        parts.append(("log_num", FunctionTransformer(np.log1p, validate=False), log))
    if num:
        parts.append(("num_spline", Pipeline([("scale", StandardScaler()),
                                              ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False,
                                                                           extrapolation="linear"))]), num))
    if binary:
        parts.append(("bin", "passthrough", binary))
    if cat:
        parts.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat))
    pre = ColumnTransformer(parts, remainder="drop")
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(solver="lbfgs", C=C, max_iter=3000, random_state=SEED))])


def build(name, p, feats=FEATURES):
    if name == "A":
        return make_A(C=p["C"], l1_ratio=p["l1_ratio"], feats=feats)
    if name == "B":
        return make_B(max_depth=p["max_depth"], n_estimators=p["n_estimators"],
                      learning_rate=p["learning_rate"], feats=feats)
    return make_C(C=p["C"], feats=feats)


def cv_oof():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def metrics(y, p):
    y = np.asarray(y)
    return {"Log loss": log_loss(y, p), "AUC": roc_auc_score(y, p), "Brier": brier_score_loss(y, p),
            "PR-AUC": average_precision_score(y, p)}


def evaluate(est, X_train, y_train, X_test, y_test):
    """Out-of-fold (5-fold, as in notebook cell 83) + hold-out test predictions."""
    oof = cross_val_predict(est, X_train, y_train, cv=cv_oof(), method="predict_proba")[:, 1]
    p_test = est.predict_proba(X_test)[:, 1]
    return oof, p_test


@st.cache_resource(show_spinner=False)
def fit_tuned(name, _split):
    """Refit one model with the pre-computed tuned hyperparameters (all 16 baseline features)."""
    X_train, X_test, y_train, y_test = _split
    est = build(name, TUNED[name]).fit(X_train, y_train)
    oof, p_test = evaluate(est, X_train, y_train, X_test, y_test)
    return {"estimator": est, "oof": oof, "p_test": p_test}


# ----------------------------------------------------------------------------------------
# Plot helpers (pure functions -> matplotlib figure)
# ----------------------------------------------------------------------------------------
def is_numeric_feature(s):
    return s.nunique() > 10


def fig_hist(s, y, bins, log, split):
    x = np.log1p(s) if log else s
    fig, ax = new_fig(h=3.2)
    if split:
        for v, c in ((0, LIGHT), (1, ACCENT)):
            ax.hist(x[y == v], bins=bins, alpha=0.65, color=c, edgecolor="white", label=lab_val("y24", v))
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.hist(x, bins=bins, color=GROUP_COLOR[group_of(s.name)], edgecolor="white")
    ax.set_xlabel(f"log(1 + {s.name})" if log else s.name)
    ax.set_ylabel("patients")
    return fig


def fig_box(s):
    fig, ax = new_fig(w=2.6, h=3.2)
    c = GROUP_COLOR[group_of(s.name)]
    ax.boxplot(s.dropna(), showfliers=True, patch_artist=True,
               boxprops=dict(facecolor=c + "33", edgecolor=c),
               medianprops=dict(color=INK), flierprops=dict(marker="o", markersize=3, alpha=0.4))
    ax.set_xticks([])
    ax.set_title(s.name, fontsize=10)
    return fig


def fig_counts(s):
    vc = s.value_counts().sort_index()
    names = [lab_val(s.name, v) for v in vc.index]
    fig, ax = new_fig(h=3.2)
    bars = ax.bar(names, vc.values, color=GROUP_COLOR[group_of(s.name)], edgecolor="white")
    ax.set_ylim(0, vc.max() * 1.18)
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v / vc.sum():.0%})", ha="center", va="bottom",
                fontsize=8, color=INK)
    ax.set_ylabel("patients")
    plt.setp(ax.get_xticklabels(), rotation=15 if len(vc) > 3 else 0)
    return fig


def rate_table(lab, col, q):
    s = lab[col]
    if is_numeric_feature(s):
        zero = s == 0
        if zero.mean() > 1 / q:                              # spike at 0 gets its own bin (notebook cell 59)
            g = pd.Series(-1, index=s.index)
            g[~zero] = pd.qcut(s[~zero], max(q - 1, 1), labels=False, duplicates="drop")
        else:
            g = pd.qcut(s, q, labels=False, duplicates="drop")
        t = lab.assign(g=g).groupby("g").agg(x=(col, "median"), n=("y24", "size"), rate=("y24", "mean"))
    else:
        t = lab.groupby(col).agg(n=("y24", "size"), rate=("y24", "mean"))
        t["x"] = t.index
    t["se"] = np.sqrt(t.rate * (1 - t.rate) / t.n)
    return t.reset_index(drop=True)


def fig_rate(lab, col, q):
    t = rate_table(lab, col, q)
    base = lab["y24"].mean()
    fig, ax = new_fig(h=3.2)
    c = GROUP_COLOR[group_of(col)]
    if is_numeric_feature(lab[col]):
        ax.errorbar(t.x, t.rate, yerr=t.se, fmt="o-", color=c, capsize=3, label="observed +/- 1 SE")
        ax.set_xlabel(col)
    else:
        names = [lab_val(col, v) for v in t.x]
        ax.bar(names, t.rate, yerr=t.se, color=c, edgecolor="white", capsize=3)
        for i, (r, n) in enumerate(zip(t.rate, t.n)):
            ax.text(i, r + (t.se.iloc[i] if np.isfinite(t.se.iloc[i]) else 0) + 0.004, f"n={n}",
                    ha="center", fontsize=8, color=MUTED)
        plt.setp(ax.get_xticklabels(), rotation=15 if len(t) > 3 else 0)
    ax.axhline(base, color=MUTED, linestyle="--", linewidth=1, label=f"overall {base:.1%}")
    ax.set_ylabel("24-month event rate")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)
    return fig


def group_legend(ax, groups):
    handles = [Patch(color=GROUP_COLOR[g], label=GROUP_SHORT[g]) for g in groups]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")


def fig_corr_bar(lab, col, method):
    num = lab.drop(columns=["pidnum"], errors="ignore").select_dtypes("number")
    c = num.corr(method=method)[col].drop(col).sort_values()
    fig, ax = new_fig(h=max(3.2, 0.26 * len(c) + 0.8))
    groups = [group_of(k) for k in c.index]
    ax.barh(c.index, c.values, color=[GROUP_COLOR[g] for g in groups], edgecolor="white")
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID)
    ax.set_xlabel(f"{method} correlation with {col}")
    ax.set_xlim(-1, 1)
    group_legend(ax, [g for g in GROUP_COLOR if g in set(groups)])
    return fig


def fig_heatmap(lab, method):
    eda = lab.drop(columns=["time", "cid", "pidnum"], errors="ignore").select_dtypes("number")
    c = eda.corr(method=method)
    fig, ax = plt.subplots(figsize=(8, 6.6))
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(c.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(c)), c.columns, fontsize=8)
    for lbl, col in list(zip(ax.get_xticklabels(), c.columns)) + list(zip(ax.get_yticklabels(), c.columns)):
        lbl.set_color(GROUP_COLOR[group_of(col)])
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"{method.capitalize()} correlation (label colour = feature group)", color=INK)
    fig.tight_layout()
    return fig


def fig_roc(curves):
    fig, ax = new_fig(h=3.8)
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=MODEL_COLOR[name], linestyle=MODEL_STYLE[name], linewidth=2.2,
                label=f"{name}  (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle="-.")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="x", color=GRID)
    return fig


def calib_points(y, p, n=8):
    d = pd.DataFrame({"p": p, "y": np.asarray(y)})
    d["g"] = pd.qcut(d.p, n, duplicates="drop")
    return d.groupby("g", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"))


def fig_calib(curves):
    fig, ax = new_fig(h=3.8)
    hi = 0.05
    for name, (y, p) in curves.items():
        t = calib_points(y, p)
        hi = max(hi, t.pred.max(), t.obs.max())
        ax.plot(t.pred, t.obs, marker=MODEL_MARKER[name], color=MODEL_COLOR[name], linestyle=MODEL_STYLE[name],
                linewidth=2, markersize=5, label=name)
    ax.plot([0, hi * 1.05], [0, hi * 1.05], color=MUTED, linewidth=0.8, linestyle="-.")
    ax.set_xlabel("mean predicted risk (groups of equal size)")
    ax.set_ylabel("observed event rate")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="x", color=GRID)
    return fig


def threshold_table(y, p, thresholds=THRESHOLDS):
    y = np.asarray(y)
    rows = []
    for t in thresholds:
        flag = p >= t
        tp, fn = int((flag & (y == 1)).sum()), int((~flag & (y == 1)).sum())
        fp, tn = int((flag & (y == 0)).sum()), int((~flag & (y == 0)).sum())
        rows.append({"threshold": t, "flagged": flag.mean(), "sensitivity": tp / max(tp + fn, 1),
                     "specificity": tn / max(tn + fp, 1), "precision": tp / (tp + fp) if tp + fp else np.nan,
                     "missed events": fn, "false alarms": fp})
    return pd.DataFrame(rows)


def best_threshold(tt, fn_weight):
    """Threshold with the lowest cost = fn_weight * missed events + false alarms (ties -> lower threshold)."""
    cost = fn_weight * tt["missed events"] + tt["false alarms"]
    return float(tt.loc[cost.idxmin(), "threshold"])


def fig_confusion(y, p, t):
    flag = (np.asarray(p) >= t).astype(int)
    cm = confusion_matrix(y, flag, labels=[0, 1])
    fig, ax = new_fig(w=4.2, h=3.4)
    ax.grid(False)
    ax.imshow(cm, cmap=SEQUENTIAL)
    names = [["true negatives", "false alarms"], ["missed events", "caught events"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{names[i][j]}", ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_xticks([0, 1], ["no event", "event"])
    ax.set_yticks([0, 1], ["no event", "event"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    return fig, cm


def fig_importance(series, title, xlabel):
    s = series.sort_values()
    fig, ax = new_fig(h=max(3.0, 0.28 * len(s) + 0.8))
    ax.barh(s.index, s.values, color=[ACCENT if v >= 0 else LIGHT for v in s.values], edgecolor="white")
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    return fig


def show(fig):
    st.pyplot(fig, clear_figure=True)


def badge(group):
    c = GROUP_COLOR[group]
    return (f"<span class='badge' style='background:{c}2e;border-color:{c}'>"
            f"<span class='dot' style='background:{c}'></span>{GROUP_SHORT[group]}</span>")


def group_legend_html():
    rows = "".join(f"<div class='gl-row'><span class='dot' style='background:{GROUP_COLOR[g]}'></span>"
                   f"<b>{GROUP_SHORT[g]}</b><span class='gl-desc'>{GROUP_TEXT[g]}</span></div>"
                   for g in ("target", "baseline", "baseline_other", "leakage"))
    return f"<div class='group-legend'>{rows}</div>"


# ----------------------------------------------------------------------------------------
# Layout: banner + side navigation
# ----------------------------------------------------------------------------------------
CSS = """
<style>
/* page frame: no Streamlit top bar, banner flush with the top edge */
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {display: none !important;}
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"], [data-testid="stExpandSidebarButton"] {display: none !important;}
.block-container, [data-testid="stMainBlockContainer"] {
    padding: 0 2rem 4rem 2rem !important; max-width: 100% !important;}
[data-testid="stSidebarHeader"] {display: none;}
[data-testid="stSidebarUserContent"] {padding-top: 1.4rem;}

/* banner */
.app-banner {display: flex; align-items: center; gap: 1.1rem; color: #fff;
             background: linear-gradient(100deg, #131d3a 0%, #1b2a4e 55%, #26396b 100%);
             margin: 0 -2rem 1.6rem -2rem; padding: 1.5rem 2rem 1.6rem 2rem;}
.app-logo {flex: none; width: 54px; height: 54px; border-radius: 14px; background: #fff; color: #1b2a4e;
           display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.05rem;}
.app-banner .t {font-size: 1.6rem; font-weight: 700; line-height: 1.2;}
.app-banner .s {font-size: 0.95rem; color: #c3cbe0; margin-top: 0.25rem;}

/* side navigation: one accent (navy) + greys */
[data-testid="stSidebar"] .stButton > button {
    width: 100%; justify-content: flex-start; text-align: left; padding: 0.7rem 1rem;
    border-radius: 10px; font-weight: 600; font-size: 1rem; margin-bottom: 0.2rem;}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    border: 1px solid transparent; background: transparent; color: #1d2433;}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {background: #eef0f5; border-color: #d5d9e3;}
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: #1b2a4e; border-color: #1b2a4e; color: #fff;}
.nav-label {font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; color: #6b7280;
            margin: 0.2rem 0 0.5rem 0.2rem;}

/* main area buttons / metrics */
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    background: #1b2a4e; border-color: #1b2a4e; color: #fff;}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
    background: #26396b; border-color: #26396b; color: #fff;}
[data-testid="stMetric"] {background: #fff; border: 1px solid #e5e7ec; border-radius: 12px; padding: 0.8rem 1rem;}

/* feature groups */
.dot {display: inline-block; width: 0.7rem; height: 0.7rem; border-radius: 50%; flex: none;}
.badge {display: inline-flex; align-items: center; gap: 0.45rem; border: 1px solid; border-radius: 999px;
        padding: 2px 12px; font-size: 0.82rem; font-weight: 600;}
.group-legend {background: #fff; border: 1px solid #e5e7ec; border-radius: 12px; padding: 0.4rem 1.1rem;
               margin: 0.2rem 0 1rem 0;}
.gl-row {display: grid; grid-template-columns: 0.7rem 11rem 1fr; align-items: center; column-gap: 0.8rem;
         padding: 0.55rem 0; border-bottom: 1px solid #eef0f4;}
.gl-row:last-child {border-bottom: none;}
.gl-desc {color: #6b7280; font-size: 0.9rem;}
</style>
"""

PAGES = [("explore", "Explore data", ":material/monitoring:"),
         ("model", "Modelling", ":material/tune:"),
         ("results", "Results", ":material/leaderboard:"),
         ("causal", "Inference & causal", ":material/science:")]


def banner():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f"<div class='app-banner'><div class='app-logo'>{LOGO_TEXT}</div>"
                f"<div><div class='t'>{PROJECT_TITLE}</div><div class='s'>{PROJECT_SUBTITLE}</div></div></div>",
                unsafe_allow_html=True)


def set_page(key):
    st.session_state["page"] = key


def sidebar_nav():
    st.session_state.setdefault("page", "explore")
    st.sidebar.markdown("<div class='nav-label'>Navigation</div>", unsafe_allow_html=True)
    for key, label, icon in PAGES:
        st.sidebar.button(label, icon=icon, key=f"nav_{key}", on_click=set_page, args=(key,),
                          type="primary" if st.session_state["page"] == key else "secondary")
    return st.session_state["page"]


# ----------------------------------------------------------------------------------------
# Small widget helpers (st.pills = clickable buttons that stay highlighted when selected)
# ----------------------------------------------------------------------------------------
SHORT = {"age": "Age", "wtkg": "Weight", "karnof": "Karnofsky score", "cd40": "CD4", "cd80": "CD8",
         "preanti": "Prior therapy days", "hemo": "Hemophilia", "homo": "Homosexual activity",
         "drugs": "IV drug use", "oprior": "Non-ZDV prior therapy", "z30": "ZDV last 30 days", "race": "Race",
         "gender": "Sex", "symptom": "Symptoms", "trt": "Treatment arm", "strat": "Therapy history"}


def feat_label(c):
    short = SHORT.get(c, c)
    return short if short.lower() == c else f"{short} ({c})"


def pick_many(label, options, key, format_func=str):
    """Click-to-toggle buttons; the selection list lives in st.session_state[key]."""
    if hasattr(st, "pills"):
        return st.pills(label, options, selection_mode="multi", key=key, format_func=format_func,
                        label_visibility="collapsed")
    return st.multiselect(label, options, key=key, format_func=format_func, label_visibility="collapsed")


def pick_one(label, options, default, key, format_func=str):
    """Single choice as buttons; falls back to `default` when the user un-selects."""
    if hasattr(st, "pills"):
        v = st.pills(label, options, selection_mode="single", default=default, key=key, format_func=format_func)
        return default if v is None else v
    return st.radio(label, options, index=list(options).index(default), key=key, format_func=format_func,
                    horizontal=True)


# ----------------------------------------------------------------------------------------
# Inference (Kaplan-Meier at day 730, notebook cells 89-91) and causal estimate (cell 94)
# ----------------------------------------------------------------------------------------
Z95 = 1.959963984540054
BOOT_N = 1000
CAUSAL_BIN, CAUSAL_CAT = ["z30", "symptom"], ["strat"]
# Values printed by the notebook (1,000 bootstrap resamples)
CAUSAL_NOTEBOOK = {"risk0": 0.2554, "risk1": 0.1261, "ate": -0.1294, "ci": (-0.1755, -0.0849)}


def arm_label(a):
    return f"Arm {a} · {TRT[a]}"


def km_curve(time, event):
    """Kaplan-Meier estimate with Greenwood standard error, evaluated at the event times."""
    t, e = np.asarray(time, float), np.asarray(event, int)
    times = np.unique(t[e == 1])
    n_at = np.array([(t >= u).sum() for u in times], float)
    d = np.array([((t == u) & (e == 1)).sum() for u in times], float)
    surv = np.cumprod(1 - d / n_at)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(n_at > d, d / (n_at * (n_at - d)), np.nan)
    return times, surv, surv * np.sqrt(np.cumsum(term))


def km_at(time, event, t0=H):
    """Event-free probability at t0 with a 95% log-log confidence interval."""
    times, surv, se = km_curve(time, event)
    i = np.searchsorted(times, t0, side="right") - 1
    if i < 0:
        return {"prob": 1.0, "se": 0.0, "lo": 1.0, "hi": 1.0}
    p, s_ = surv[i], se[i]
    ll, sll = np.log(-np.log(p)), s_ / abs(p * np.log(p))
    return {"prob": p, "se": s_, "lo": np.exp(-np.exp(ll + Z95 * sll)), "hi": np.exp(-np.exp(ll - Z95 * sll))}


@st.cache_resource(show_spinner=False)
def inference_results(_raw):
    arms = sorted(_raw["trt"].unique())
    tab = pd.DataFrame({a: km_at(_raw.loc[_raw.trt == a, "time"], _raw.loc[_raw.trt == a, "cid"]) for a in arms}).T
    curves = {a: km_curve(_raw.loc[_raw.trt == a, "time"], _raw.loc[_raw.trt == a, "cid"])[:2] for a in arms}
    rows = []
    for b, a in combinations(arms, 2):               # "Arm a vs Arm b" = P(a) - P(b), as in the notebook
        diff = tab.loc[a, "prob"] - tab.loc[b, "prob"]
        se = np.sqrt(tab.loc[a, "se"] ** 2 + tab.loc[b, "se"] ** 2)
        rows.append({"comparison": f"Arm {int(a)} vs Arm {int(b)}", "difference": diff,
                     "lo": diff - Z95 * se, "hi": diff + Z95 * se})
    return tab, curves, pd.DataFrame(rows)


def _unpenalized():
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):                      # penalty=None is deprecated; C=inf means no penalty
        return LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000)
    return LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)


@st.cache_resource(show_spinner=False)
def causal_fit(_raw):
    """Arms 0 vs 1, unpenalised logistic regression + marginal standardisation (g-computation)."""
    d = _raw[_raw.trt.isin([0, 1])].copy()
    d["event_730"] = ((d.time <= H) & (d.cid == 1)).astype(int)
    pre = ColumnTransformer([("num", StandardScaler(), NUM_COLS + LOG_COLS),
                             ("bin", "passthrough", CAUSAL_BIN),
                             ("cat", OneHotEncoder(drop="first", sparse_output=False), CAUSAL_CAT)],
                            remainder="passthrough")
    pre.set_output(transform="pandas")
    Z = pre.fit_transform(d[LOG_COLS + NUM_COLS + CAUSAL_BIN + CAUSAL_CAT + ["trt"]])
    y = d["event_730"].to_numpy()
    model = _unpenalized().fit(Z, y)
    Z0, Z1 = Z.copy(), Z.copy()
    Z0["remainder__trt"], Z1["remainder__trt"] = 0, 1
    p0, p1 = model.predict_proba(Z0)[:, 1], model.predict_proba(Z1)[:, 1]
    coef = pd.DataFrame({"feature": [n.split("__")[-1] for n in Z.columns], "log-odds": model.coef_[0]})
    coef["odds ratio"] = np.exp(coef["log-odds"])
    return {"Z": Z, "y": y, "Z0": Z0, "Z1": Z1, "p0": p0, "p1": p1, "risk0": float(p0.mean()),
            "risk1": float(p1.mean()), "ate": float(p1.mean() - p0.mean()), "n": len(d),
            "coef": coef.sort_values("log-odds").reset_index(drop=True), "intercept": float(model.intercept_[0])}


def bootstrap_ate(fit, n=None, seed=SEED):
    """95% percentile interval for the ATE (the design matrix is built once; for an unpenalised
    model, re-standardising inside each resample does not change the fitted probabilities)."""
    n = BOOT_N if n is None else n
    Z, Z0, Z1, y = (fit[k].to_numpy() if hasattr(fit[k], "to_numpy") else fit[k] for k in ("Z", "Z0", "Z1", "y"))
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for b in range(n):
        i = rng.integers(0, len(y), len(y))
        m = _unpenalized().fit(Z[i], y[i])
        out[b] = m.predict_proba(Z1[i])[:, 1].mean() - m.predict_proba(Z0[i])[:, 1].mean()
    return tuple(float(v) for v in np.percentile(out, [2.5, 97.5]))


def run_bootstrap(fit):
    st.session_state["causal_ci"] = bootstrap_ate(fit)


def fig_km(curves):
    fig, ax = new_fig(h=3.8)
    lo = 1.0
    for a, (t, sv) in curves.items():
        x, yv = np.r_[0, t], np.r_[1, sv]
        ax.step(x, yv, where="post", color=ARM_LINE[a], linestyle=ARM_STYLE[a], linewidth=2, label=arm_label(a))
        lo = min(lo, np.nanmin(yv[x <= 1200]))
    ax.axvline(H, color=MUTED, linewidth=1, linestyle="-.")
    ax.set_xlim(0, 1200)
    ax.set_ylim(max(0.0, lo - 0.03), 1.0)
    ax.text(H + 12, ax.get_ylim()[0] + 0.01, "730 days", color=MUTED, fontsize=8)
    ax.set_xlabel("days")
    ax.set_ylabel("probability of staying event-free")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(axis="x", color=GRID)
    return fig


def fig_forest(pairs):
    fig, ax = new_fig(h=max(2.8, 0.55 * len(pairs) + 1.2))
    ypos = np.arange(len(pairs))[::-1]
    for yy, r in zip(ypos, pairs.itertuples()):
        clear = r.lo > 0 or r.hi < 0
        ax.errorbar(r.difference * 100, yy, xerr=[[(r.difference - r.lo) * 100], [(r.hi - r.difference) * 100]],
                    fmt="o", color=ACCENT if clear else "#8b919c", capsize=4, markersize=6, linewidth=2)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(ypos, list(pairs["comparison"]))
    ax.set_ylim(-0.7, len(pairs) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID)
    ax.set_xlabel("difference in event-free probability at 730 days (percentage points)")
    return fig


def fig_overlay_hist(groups, bins, xlabel):
    """groups = [(label, values, fill colour, line colour)].
    Two groups -> overlaid density histograms; more -> side-by-side bars per bin (stays readable)."""
    allv = np.concatenate([np.asarray(v, float) for _, v, _, _ in groups])
    edges = np.linspace(allv.min(), allv.max(), bins + 1)
    fig, ax = new_fig(h=3.5)
    if len(groups) <= 2:
        for label, v, fill, line in groups:
            ax.hist(v, bins=edges, density=True, histtype="stepfilled", color=fill, alpha=0.45)
            ax.hist(v, bins=edges, density=True, histtype="step", color=line, linewidth=2, label=label)
    else:
        _, _, containers = ax.hist([np.asarray(v, float) for _, v, _, _ in groups], bins=edges, density=True,
                                   color=[g[2] for g in groups], label=[g[0] for g in groups])
        for cont, g in zip(containers, groups):
            for patch in cont:
                patch.set_edgecolor(g[3])
                patch.set_linewidth(0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    return fig


# ----------------------------------------------------------------------------------------
# Page 1 - Explore
# ----------------------------------------------------------------------------------------
def page_explore(raw, lab):
    st.header("Explore the data")
    st.caption("ACTG 175 clinical trial (UCI id 890). The target `y24` is 1 if the patient had an event "
               "(AIDS-defining event, CD4 drop >= 50 %, or death) within 730 days, and 0 if followed >= 730 days "
               "event-free.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{len(raw):,}")
    c2.metric("Labelled at 24 months", f"{len(lab):,}")
    c3.metric("Dropped (censored early)", f"{len(raw) - len(lab):,}")
    c4.metric("24-month event rate", f"{lab.y24.mean():.1%}")

    st.markdown("**Feature groups**")
    st.markdown(group_legend_html(), unsafe_allow_html=True)

    order = {"target": 0, "baseline": 1, "baseline_other": 2, "leakage": 3}
    all_cols = ["y24"] + [c for c in lab.columns if c not in ("pidnum", "y24")]
    all_cols.sort(key=lambda c: order[group_of(c)])
    flt = st.radio("Show", ["All", "Target", "Baseline features", "Leakage"], horizontal=True)
    pool = {"All": all_cols,
            "Target": [c for c in all_cols if group_of(c) == "target"],
            "Baseline features": [c for c in all_cols if group_of(c) in ("baseline", "baseline_other")],
            "Leakage": [c for c in all_cols if group_of(c) == "leakage"]}[flt]
    default = pool.index("cd40") if "cd40" in pool else 0
    col = st.selectbox("Choose a feature to explore", pool, index=default,
                       format_func=lambda c: f"{c}  ·  {GROUP_SHORT[group_of(c)]}  ·  {DESC.get(c, '')}")
    grp = group_of(col)
    s = lab[col]
    numeric = is_numeric_feature(s)
    st.markdown(f"**{col}** - {DESC.get(col, '')}  \n{badge(grp)}", unsafe_allow_html=True)
    st.caption(f"{'numeric' if numeric else 'binary / categorical'} · {s.nunique()} distinct values · "
               f"{int(s.isna().sum())} missing")

    t_sum, t_dist, t_out, t_cor = st.tabs(["Summary", "Distribution", "Relation to outcome", "Correlations"])

    with t_sum:
        if col == "y24":
            vc = s.value_counts().sort_index()
            st.dataframe(pd.DataFrame({"value": [lab_val("y24", v) for v in vc.index], "patients": vc.values,
                                       "share": (vc.values / len(s)).round(3)}),
                         hide_index=True, use_container_width=True)
            st.caption("How the target was built from the full dataset:")
            built = pd.DataFrame({
                "group": ["Event on or before day 730 (y24 = 1)", "Event-free, followed >= 730 days (y24 = 0)",
                          "Censored before day 730 (dropped, label unknown)"],
                "patients": [int(((raw.cid == 1) & (raw.time <= H)).sum()),
                             int(((raw.cid == 0) & (raw.time >= H)).sum() + ((raw.cid == 1) & (raw.time > H)).sum()),
                             int(len(raw) - len(lab))]})
            st.dataframe(built, hide_index=True, use_container_width=True)
        elif numeric:
            d = s.describe().to_frame("value")
            d.loc["skew"] = s.skew()
            d.loc["share of zeros"] = (s == 0).mean()
            d.loc["missing"] = s.isna().sum()
            st.dataframe(d.round(3), use_container_width=True)
            st.caption("Same summary split by outcome:")
            st.dataframe(lab.groupby("y24")[col].describe().rename(index=lambda v: lab_val("y24", v)).round(2),
                         use_container_width=True)
        else:
            vc = s.value_counts(dropna=False).sort_index()
            tab = pd.DataFrame({"value": [lab_val(col, v) for v in vc.index], "patients": vc.values,
                                "share": (vc.values / len(s)).round(3),
                                "event rate": lab.groupby(col)["y24"].mean().reindex(vc.index).round(3).values})
            st.dataframe(tab, hide_index=True, use_container_width=True)

    with t_dist:
        if numeric:
            a, b, c = st.columns(3)
            bins = a.slider("Bins", 10, 80, 30)
            log = b.checkbox("log(1 + x) scale", value=False)
            split = c.checkbox("Split by outcome", value=False)
            left, right = st.columns([3, 1])
            with left:
                show(fig_hist(s, lab["y24"], bins, log, split))
            with right:
                show(fig_box(s))
            st.caption(f"Skewness raw {s.skew():.2f} · after log(1+x) {np.log1p(s).skew():.2f}")
        else:
            show(fig_counts(s))

    with t_out:
        if col == "y24":
            by_opts = [c for c in FEATURES]
            by = st.selectbox("Break the target down by", by_opts, index=by_opts.index("trt"),
                              format_func=lambda c: f"{c} - {DESC.get(c, '')}")
            q = st.slider("Number of quantile bins", 3, 10, 6) if is_numeric_feature(lab[by]) else 0
            show(fig_rate(lab, by, q))
            st.caption("Share of patients with an event within 24 months for each value of the chosen feature.")
        else:
            if col in ("time", "cid"):
                st.info(f"`{col}` is part of the outcome definition, so its relation to `y24` is built in.")
            q = st.slider("Number of quantile bins", 3, 10, 6) if numeric else 0
            show(fig_rate(lab, col, q))
            st.caption("Observed 24-month event rate (dots = bins of the feature, bars = +/- 1 standard error). "
                       "Dashed line = overall rate.")

    with t_cor:
        method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
        show(fig_corr_bar(lab, col, method))
        with st.expander("Full correlation matrix (features + y24)"):
            show(fig_heatmap(lab, method))


# ----------------------------------------------------------------------------------------
# Page 2 - Modelling
# ----------------------------------------------------------------------------------------
DEFAULTS = {"A_logC": round(float(np.log10(TUNED["A"]["C"])) * 4) / 4,      # -0.5 (slider step 0.25)
            "A_l1": TUNED["A"]["l1_ratio"], "B_depth": TUNED["B"]["max_depth"],
            "B_lr": TUNED["B"]["learning_rate"], "B_n": TUNED["B"]["n_estimators"],
            "C_logC": round(float(np.log10(TUNED["C"]["C"])) * 4) / 4}


def init_state():
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("feat_sel", list(FEATURES))


def apply_tuned(name):
    best = TUNED[name]
    if name == "A":
        st.session_state["A_logC"] = round(float(np.log10(best["C"])) * 4) / 4
        st.session_state["A_l1"] = float(best["l1_ratio"])
    elif name == "B":
        st.session_state["B_depth"] = int(best["max_depth"])
        st.session_state["B_lr"] = round(float(best["learning_rate"]), 2)
        st.session_state["B_n"] = int(best["n_estimators"])
    else:
        st.session_state["C_logC"] = round(float(np.log10(best["C"])) * 4) / 4


def set_features(feats):
    st.session_state["feat_sel"] = list(feats)


def hyperparameter_controls(name):
    st.subheader("Hyperparameters")
    if name == "A":
        st.caption("Logistic regression with an elastic-net penalty. Small C = stronger shrinkage; "
                   "l1_ratio 0 = ridge, 1 = lasso.")
        logc = st.slider("log10(C)", -3.0, 1.0, step=0.25, key="A_logC")
        l1 = st.slider("l1_ratio", 0.0, 1.0, step=0.05, key="A_l1")
        st.write(f"C = **{10 ** logc:.4g}**")
        return {"C": float(10 ** logc), "l1_ratio": float(l1)}
    if name == "B":
        st.caption("Gradient-boosted trees (stops early when the validation score stops improving).")
        d = st.slider("max_depth", 1, 5, key="B_depth")
        lr = st.slider("learning_rate", 0.01, 0.5, step=0.01, key="B_lr")
        n = st.slider("n_estimators", 5, 500, step=5, key="B_n")
        return {"max_depth": int(d), "learning_rate": float(lr), "n_estimators": int(n)}
    st.caption("Logistic regression on spline-expanded numeric features (a GAM-style model) with L2 penalty.")
    logc = st.slider("log10(C)", -3.0, 2.0, step=0.25, key="C_logC")
    st.write(f"C = **{10 ** logc:.4g}**")
    return {"C": float(10 ** logc)}


def feature_inputs(feats, X_train):
    """Render one input per selected feature; returns (one-row DataFrame, list of out-of-range numeric features)."""
    med = X_train.median()
    lo, hi = X_train.min(), X_train.max()
    row = {}
    cols = st.columns(3)
    for i, c in enumerate(feats):
        with cols[i % 3]:
            k = f"pf_{c}"
            if c == "age":
                row[c] = st.number_input("Age (years)", 12.0, 100.0, float(med[c]), 1.0, key=k)
            elif c == "wtkg":
                row[c] = st.number_input("Weight (kg)", 30.0, 200.0, float(med[c]), 0.5, key=k)
            elif c == "karnof":
                opts = [70, 80, 90, 100]
                row[c] = st.selectbox("Karnofsky score", opts, index=opts.index(min(opts, key=lambda v: abs(v - med[c]))),
                                      key=k)
            elif c == "cd40":
                row[c] = st.number_input("CD4 baseline (cells/mm3)", 0.0, 2000.0, float(med[c]), 5.0, key=k)
            elif c == "cd80":
                row[c] = st.number_input("CD8 baseline (cells/mm3)", 0.0, 6000.0, float(med[c]), 10.0, key=k)
            elif c == "preanti":
                row[c] = st.number_input("Days of prior antiretroviral therapy", 0.0, 5000.0, float(med[c]), 10.0,
                                         key=k)
            elif c == "trt":
                row[c] = st.selectbox("Treatment arm", list(TRT), format_func=TRT.get, key=k)
            elif c == "strat":
                opts = list(STRAT)
                row[c] = st.selectbox("Antiretroviral history", opts, format_func=STRAT.get,
                                      index=opts.index(int(med[c])) if int(med[c]) in opts else 0, key=k)
            elif c in ("symptom", "gender", "race"):
                title = {"symptom": "Symptoms", "gender": "Sex", "race": "Race"}[c]
                row[c] = st.selectbox(title, [0, 1], format_func=VALUE_LABELS[c].get,
                                      index=int(round(med[c])), key=k)
            else:
                label = {"hemo": "Hemophilia", "homo": "Homosexual activity", "drugs": "IV drug use history",
                         "oprior": "Non-ZDV antiretroviral before study", "z30": "ZDV in the 30 days before study"}[c]
                row[c] = int(st.checkbox(label, value=bool(round(med[c])), key=k))
    x = pd.DataFrame([row])[list(feats)]
    outside = [c for c in feats if c in LOG_COLS + NUM_COLS and not (lo[c] <= x.loc[0, c] <= hi[c])]
    return x, outside


def page_modelling(lab, split):
    X_train, X_test, y_train, y_test = split
    init_state()
    st.header("Modelling")
    name = st.radio("Model", list(MODEL_NAME), format_func=MODEL_NAME.get, horizontal=True)

    sel = st.session_state["feat_sel"]
    with st.container(border=True):
        h1, h2, h3 = st.columns([6, 1.2, 1])
        h1.markdown(f"**Baseline features** · {len(sel)} of {len(FEATURES)} selected  \n"
                    "Click a feature to add or remove it. Highlighted = used by the model. "
                    "Only variables known at the start of the trial are offered.")
        h2.button("Select all", on_click=set_features, args=(FEATURES,), key="sel_all")
        h3.button("Clear", on_click=set_features, args=([],), key="sel_none")
        pick_many("Features", FEATURES, key="feat_sel", format_func=feat_label)
        with st.expander("What do the features mean?"):
            st.dataframe(pd.DataFrame({"feature": FEATURES, "description": [DESC[c] for c in FEATURES]}),
                         hide_index=True, use_container_width=True)
    feats = [c for c in FEATURES if c in st.session_state["feat_sel"]]

    left, right = st.columns([1, 2])
    with left:
        params = hyperparameter_controls(name)
        st.button("Reset to tuned values", on_click=apply_tuned, args=(name,),
                  help="Hyperparameters found by the grid search in the notebook (all 16 features).")
        if not feats:
            st.warning("Select at least one feature.")
        go = st.button("Train model", type="primary", disabled=not feats)
    if go:
        with st.spinner("Training and cross-validating ..."):
            est = build(name, params, feats).fit(X_train[feats], y_train)
            oof, p_test = evaluate(est, X_train[feats], y_train, X_test[feats], y_test)
        st.session_state["fit"] = {"name": name, "params": params, "feats": feats, "est": est,
                                   "oof": oof, "p_test": p_test}

    fit = st.session_state.get("fit")
    with right:
        if not fit:
            st.info("Choose features and hyperparameters, then press **Train model**.")
            return
        if fit["name"] != name or fit["params"] != params or fit["feats"] != feats:
            st.warning("The settings changed since the last training. Showing the last trained model "
                       f"({MODEL_NAME[fit['name']]}, {len(fit['feats'])} features). Press **Train model** to apply.")
        m_cv, m_te = metrics(y_train, fit["oof"]), metrics(y_test, fit["p_test"])
        base_ll = log_loss(y_train, np.full(len(y_train), y_train.mean()))
        st.subheader(f"Results - {MODEL_NAME[fit['name']]}")
        st.caption(f"Hyperparameters: {fit['params']} · {len(fit['feats'])} features: {', '.join(fit['feats'])}")
        k = st.columns(4)
        for col, key in zip(k, ["Log loss", "AUC", "Brier", "PR-AUC"]):
            col.metric(key, f"{m_cv[key]:.3f}", help="5-fold out-of-fold on the training set")
        st.caption(f"Cross-validated (5-fold, training set). Base-rate log loss = {base_ll:.3f}. "
                   f"Hold-out test set: log loss {m_te['Log loss']:.3f} · AUC {m_te['AUC']:.3f} · "
                   f"Brier {m_te['Brier']:.3f}.")
        t1, t2, t3 = st.tabs(["ROC & calibration", "Threshold", "What drives it"])
        with t1:
            a, b = st.columns(2)
            with a:
                show(fig_roc({fit["name"]: (y_train, fit["oof"])}))
            with b:
                show(fig_calib({fit["name"]: (y_train, fit["oof"])}))
        with t2:
            thr = st.slider("Decision threshold (flag patient if predicted risk >=)", 0.02, 0.60, 0.10, 0.01)
            fig, cm = fig_confusion(y_train, fit["oof"], thr)
            tn, fp, fn_, tp = cm.ravel()
            a, b = st.columns([1, 1])
            with a:
                show(fig)
            with b:
                st.metric("Sensitivity", f"{tp / max(tp + fn_, 1):.1%}")
                st.metric("Specificity", f"{tn / max(tn + fp, 1):.1%}")
                st.metric("Precision", f"{tp / (tp + fp):.1%}" if tp + fp else "-")
            st.caption("Computed on out-of-fold predictions for the training set.")
        with t3:
            est = fit["est"]
            if fit["name"] == "C":                    # the log step of model C has no feature names
                names = []
            else:
                names = [n.split("__")[-1] for n in est["pre"].get_feature_names_out()]
            if fit["name"] == "A":
                coef = pd.Series(est["clf"].coef_[0], index=names)
                show(fig_importance(coef, "Coefficients (standardised inputs; log-odds)", "coefficient"))
                st.caption(f"Zeroed out by the penalty: {', '.join(coef[coef == 0].index) or 'none'}")
            elif fit["name"] == "B":
                imp = pd.Series(est["clf"].feature_importances_, index=names)
                show(fig_importance(imp, "Impurity-based feature importance", "importance"))
            else:
                st.write("Each numeric feature is expanded into several spline terms, so there is no single "
                         "coefficient per feature. Use the prediction form below to see how inputs change the risk.")

    st.divider()
    st.subheader("Predict for a new patient")
    st.caption(f"Inputs for the {len(fit['feats'])} features of the trained model "
               f"({MODEL_NAME[fit['name']].split(' - ')[0]}).")
    with st.form("patient"):
        x, outside = feature_inputs(fit["feats"], X_train)
        submitted = st.form_submit_button("Predict", type="primary")
    if not submitted:
        return
    est = fit["est"]
    p = float(est.predict_proba(x)[:, 1][0])
    base = float(y_train.mean())
    a, b = st.columns([1, 2])
    with a:
        st.metric(f"Predicted 24-month event risk (Model {fit['name']})", f"{p:.1%}",
                  delta=f"{(p - base) * 100:+.1f} pp vs. cohort average ({base:.1%})", delta_color="inverse")
        if outside:
            st.warning("Outside the range seen in training: " + ", ".join(outside) + ". Treat with caution.")
    with b:
        if "trt" in fit["feats"]:
            rows = []
            for arm in TRT:
                xa = x.copy()
                xa["trt"] = arm
                rows.append(float(est.predict_proba(xa)[:, 1][0]))
            fig, ax = new_fig(h=2.6)
            ax.bar([TRT[a_] for a_ in TRT], rows, color=ACCENT, edgecolor="white")
            ax.axhline(base, color=MUTED, linestyle="--", linewidth=1)
            for i, v in enumerate(rows):
                ax.text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=8, color=INK)
            ax.set_ylabel("predicted risk")
            ax.set_title("Same patient under each treatment arm", fontsize=10)
            show(fig)
        else:
            st.info("Add the treatment arm (`trt`) to the features to see predictions for each arm.")
    st.caption("Predictions are statistical associations from a randomised trial cohort, not clinical advice. "
               "Differences between arms here are model estimates and are not a causal analysis.")


# ----------------------------------------------------------------------------------------
# Page 3 - Results
# ----------------------------------------------------------------------------------------
def fmt_params(p):
    return ", ".join(f"{a} = {b:.4g}" if isinstance(b, float) else f"{a} = {b}" for a, b in p.items())


def page_results(split):
    X_train, X_test, y_train, y_test = split
    st.header("Results - the three tuned models")
    st.caption("Hyperparameters were tuned in the notebook with grid search (selection by log loss) and are shown "
               "below as pre-computed values. Here the models are refit on the training set with them "
               "(all 16 baseline features).")

    st.subheader("Tuned hyperparameters")
    st.dataframe(pd.DataFrame({
        "Model": [MODEL_NAME[k] for k in "ABC"],
        "Tuned hyperparameters": [fmt_params(TUNED[k]) for k in "ABC"],
        "CV log loss in the search": [TUNED_CV_LOGLOSS[k] for k in "ABC"],
        "Search space": [SEARCH_SPACE[k] for k in "ABC"]}), hide_index=True, use_container_width=True)
    with st.expander("Best candidates from the notebook grid searches"):
        k2 = st.radio("Model ", list("ABC"), format_func=lambda v: MODEL_NAME[v], horizontal=True, key="grid_model")
        st.dataframe(TOP_GRID[k2], hide_index=True, use_container_width=True)

    with st.spinner("Fitting the three tuned models ..."):
        tuned = {k: fit_tuned(k, split) for k in "ABC"}

    st.subheader("Performance")
    base_ll = log_loss(y_train, np.full(len(y_train), y_train.mean()))
    rows = []
    for k in "ABC":
        cv, te = metrics(y_train, tuned[k]["oof"]), metrics(y_test, tuned[k]["p_test"])
        rows.append({"Model": MODEL_NAME[k], "CV log loss": cv["Log loss"], "CV AUC": cv["AUC"],
                     "CV Brier": cv["Brier"], "CV PR-AUC": cv["PR-AUC"],
                     "Test log loss": te["Log loss"], "Test AUC": te["AUC"], "Test Brier": te["Brier"]})
    perf = pd.DataFrame(rows).set_index("Model")
    st.dataframe(perf.style.format("{:.4f}").highlight_min(subset=["CV log loss", "CV Brier", "Test log loss",
                                                                  "Test Brier"], color="#dfe3ec")
                 .highlight_max(subset=["CV AUC", "CV PR-AUC", "Test AUC"], color="#dfe3ec"),
                 use_container_width=True)
    st.caption(f"CV = 5-fold out-of-fold predictions on the training set (n={len(y_train)}). "
               f"Test = hold-out set (n={len(y_test)}), never used for tuning. "
               f"Base-rate log loss = {base_ll:.4f}; PR-AUC of a no-skill model = {y_train.mean():.3f}. "
               "Green = best in column. Differences are often within fold-to-fold noise.")

    where = st.radio("Show curves for", ["Cross-validated (training)", "Hold-out test"], horizontal=True)
    if where.startswith("Cross"):
        curves = {k: (y_train, tuned[k]["oof"]) for k in "ABC"}
    else:
        curves = {k: (y_test, tuned[k]["p_test"]) for k in "ABC"}
    a, b = st.columns(2)
    with a:
        st.markdown("**ROC curves**")
        show(fig_roc(curves))
    with b:
        st.markdown("**Calibration**")
        show(fig_calib(curves))

    st.subheader("Do the models agree?")
    oofs = pd.DataFrame({k: tuned[k]["oof"] for k in "ABC"})
    st.dataframe(oofs.corr(method="spearman").round(3), use_container_width=True)
    st.caption("Spearman rank correlation of out-of-fold predicted risks.")

    st.subheader("Thresholds and confusion matrix")
    st.caption("A threshold turns a predicted risk into a decision: patients at or above it are flagged as "
               "'will have an event'. Missing a real event is usually worse than a false alarm, so the "
               "recommended threshold (★) is the one with the lowest cost = missed events × weight + false alarms.")
    c1, c2 = st.columns([3, 2])
    with c1:
        k = st.radio("Model", list("ABC"), format_func=lambda v: MODEL_NAME[v], horizontal=True, key="thr_model")
    with c2:
        w = st.slider("One missed event counts as this many false alarms", 1, 10, 5, key="fn_weight")
    p_k = tuned[k]["oof"]
    tt = threshold_table(y_train, p_k)
    best = best_threshold(tt, w)
    pick = pick_one("Threshold (★ = recommended; click another one to compare)", THRESHOLDS, default=best,
                    key=f"thr_pick_{k}_{w}",
                    format_func=lambda t: f"{t:.2f}" + (" ★" if abs(t - best) < 1e-9 else ""))
    t_sel = best if pick is None else float(pick)

    left, right = st.columns([3, 2])
    with left:
        shown = tt.copy()
        shown.insert(1, "", np.where(np.isclose(shown["threshold"], best), "★", ""))

        def mark(row):
            hit = np.isclose(row["threshold"], t_sel)
            return ["background-color: #dfe3ec; font-weight: 700" if hit else ""] * len(row)
        st.dataframe(shown.style.apply(mark, axis=1)
                     .format({"threshold": "{:.2f}", "flagged": "{:.1%}", "sensitivity": "{:.3f}",
                              "specificity": "{:.3f}", "precision": "{:.3f}"}),
                     hide_index=True, use_container_width=True)
        st.caption("Out-of-fold predictions on the training set. 'Flagged' = share of patients at or above the "
                   "threshold. The highlighted row is the one shown on the right.")
    with right:
        fig, cm = fig_confusion(y_train, p_k, t_sel)
        show(fig)
        tn, fp, fn_, tp = cm.ravel()
        a, b = st.columns(2)
        a.metric("Missed events", f"{fn_}", help="Patients with an event that were not flagged (false negatives)")
        b.metric("False alarms", f"{fp}", help="Patients without an event that were flagged (false positives)")
        a.metric("Sensitivity", f"{tp / max(tp + fn_, 1):.1%}")
        b.metric("Precision", f"{tp / (tp + fp):.1%}" if tp + fp else "-")
    row_b = tt[np.isclose(tt["threshold"], best)].iloc[0]
    row_s = tt[np.isclose(tt["threshold"], t_sel)].iloc[0]
    if np.isclose(t_sel, best):
        st.success(f"Showing the recommended threshold {best:.2f} for Model {k}: {int(row_b['missed events'])} "
                   f"missed events and {int(row_b['false alarms'])} false alarms "
                   f"(weight {w}:1).")
    else:
        d_fn = int(row_s["missed events"] - row_b["missed events"])
        d_fp = int(row_s["false alarms"] - row_b["false alarms"])
        st.info(f"Threshold {t_sel:.2f} instead of the recommended {best:.2f}: {d_fn:+d} missed events and "
                f"{d_fp:+d} false alarms.")


def page_causal(raw):
    st.header("Inference and causal analysis")
    t_inf, t_cau = st.tabs(["Inference · survival at 24 months", "Causal · ZDV vs ZDV + ddI"])

    with t_inf:
        tab, curves, pairs = inference_results(raw)
        st.caption("Kaplan-Meier estimate of the probability of staying event-free up to day 730, per treatment "
                   "arm, using all patients (censored patients included). Intervals are 95 %: log-log "
                   "transformation for each arm, Wald interval for the differences.")
        st.subheader("Event-free probability at 730 days")
        st.dataframe(pd.DataFrame({"Arm": [arm_label(a) for a in tab.index], "Event-free probability": tab.prob.values,
                                   "SE (Greenwood)": tab.se.values, "95% CI lower": tab.lo.values,
                                   "95% CI upper": tab.hi.values})
                     .style.format({c: "{:.4f}" for c in ["Event-free probability", "SE (Greenwood)",
                                                          "95% CI lower", "95% CI upper"]}),
                     hide_index=True, use_container_width=True)
        a, b = st.columns(2)
        with a:
            st.markdown("**Kaplan-Meier curves**")
            show(fig_km(curves))
        with b:
            st.markdown("**Differences between arms (95% interval)**")
            show(fig_forest(pairs))
        st.subheader("All pairwise differences")
        shown = pairs.rename(columns={"comparison": "Comparison", "difference": "Difference",
                                      "lo": "95% CI lower", "hi": "95% CI upper"})
        shown["Interval excludes 0?"] = np.where((pairs.lo > 0) | (pairs.hi < 0), "yes", "no")
        st.dataframe(shown.style.format({c: "{:.4f}" for c in ["Difference", "95% CI lower", "95% CI upper"]}),
                     hide_index=True, use_container_width=True)
        clear = [r.comparison for r in pairs.itertuples() if r.lo > 0 or r.hi < 0]
        unclear = [r.comparison for r in pairs.itertuples() if not (r.lo > 0 or r.hi < 0)]
        st.caption("'Arm a vs Arm b' = event-free probability of a minus that of b (positive = a is better). "
                   f"Intervals that exclude 0: {', '.join(clear) or 'none'}. "
                   f"Intervals that include 0 (no clear difference): {', '.join(unclear) or 'none'}.")

    with t_cau:
        fit = causal_fit(raw)
        st.caption("Question: how much would the 24-month event risk change if everybody received ZDV + ddI instead "
                   "of ZDV alone? Only arms 0 and 1 are used. An unpenalised logistic regression adjusts for "
                   "baseline covariates, then every patient is predicted twice (all on ZDV, all on ZDV + ddI) and the "
                   "average risks are compared (marginal standardisation). Patients censored before day 730 count "
                   "as 'no event', as in the notebook.")
        boot = st.session_state.get("causal_ci")
        ci, source = (boot, f"live bootstrap, {BOOT_N:,} resamples") if boot else (
            CAUSAL_NOTEBOOK["ci"], "from the notebook, 1,000 bootstrap resamples")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Risk if everyone got ZDV only", f"{fit['risk0']:.1%}")
        m2.metric("Risk if everyone got ZDV + ddI", f"{fit['risk1']:.1%}")
        m3.metric("Average treatment effect", f"{fit['ate'] * 100:+.1f} pp")
        m4.metric("95% interval", f"[{ci[0] * 100:.1f}, {ci[1] * 100:.1f}] pp", help=source)
        st.caption(f"Interval source: {source}. Based on {fit['n']:,} patients (arms 0 and 1).")
        st.button(f"Recompute the interval here ({BOOT_N:,} bootstrap resamples)", on_click=run_bootstrap,
                  args=(fit,), help="Takes a few seconds; the result can differ slightly from the notebook.")
        st.markdown(f"Treating the whole population with ZDV + ddI instead of ZDV alone is estimated to change the "
                    f"24-month event risk by **{fit['ate'] * 100:+.1f} percentage points** "
                    f"(95% interval {ci[0] * 100:+.1f} to {ci[1] * 100:+.1f}). "
                    + ("The interval excludes 0." if ci[1] < 0 or ci[0] > 0 else "The interval includes 0."))
        with st.expander("The model behind the estimate"):
            cf = fit["coef"].copy()
            st.dataframe(cf.style.format({"log-odds": "{:.4f}", "odds ratio": "{:.3f}"}), hide_index=True,
                         use_container_width=True)
            or_trt = float(cf.loc[cf.feature == "trt", "odds ratio"].iloc[0])
            st.caption(f"Intercept {fit['intercept']:.4f}. Holding the covariates fixed, ZDV + ddI multiplies the "
                       f"odds of an event by {or_trt:.2f} (a {100 * (1 - or_trt):.0f} % reduction in odds).")

        st.subheader("Do the arms look alike at baseline?")
        st.caption("In a randomised trial the treatment groups should have similar baseline distributions, which is "
                   "what makes the comparison above causal. Colours show the treatment arm.")
        c1, c2, c3 = st.columns([2, 3, 1])
        cov = c1.selectbox("Baseline measurement", ["age", "wtkg", "cd40", "cd80", "preanti"],
                           format_func=lambda c: f"{c} - {DESC[c]}", key="cau_cov")
        which = c2.radio("Arms", ["All four arms", "Only the two compared arms (0 and 1)"], horizontal=True,
                         key="cau_arms")
        bins = c3.slider("Bins", 10, 60, 25, key="cau_bins")
        arms = [0, 1, 2, 3] if which.startswith("All") else [0, 1]
        show(fig_overlay_hist([(arm_label(a), raw.loc[raw.trt == a, cov], ARM_COLOR[a], ARM_LINE[a]) for a in arms],
                              bins, DESC[cov]))

        st.subheader("Predicted risk if everyone received each treatment")
        st.caption("Each patient's predicted 24-month event risk under ZDV only (grey) and under ZDV + ddI (navy). "
                   "The shift between the two histograms is the treatment effect.")
        show(fig_overlay_hist([("Everyone on ZDV only", fit["p0"], ARM_COLOR[0], ARM_LINE[0]),
                               ("Everyone on ZDV + ddI", fit["p1"], ARM_COLOR[1], ARM_LINE[1])],
                              30, "predicted 24-month event risk"))


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def get_raw():
    try:
        return load_raw()
    except Exception as e:                                       # no internet / UCI down
        st.error(f"Could not download the dataset ({e}). Upload `actg175.csv` (the UCI file, with columns "
                 "pidnum, time, cid, trt, age, ...) or place it next to app.py.")
        up = st.file_uploader("actg175.csv", type="csv")
        return pd.read_csv(up) if up else None


def main():
    st.set_page_config(page_title="ACTG 175 - 24-month event models", layout="wide")
    page = sidebar_nav()
    banner()
    raw = get_raw()
    if raw is None:
        st.stop()
    lab = labelled_cohort(raw)
    split = make_split(lab)
    st.sidebar.divider()
    st.sidebar.caption(f"{len(lab):,} labelled patients · train {len(split[0]):,} / test {len(split[1]):,}")
    if page == "explore":
        page_explore(raw, lab)
    elif page == "model":
        page_modelling(lab, split)
    elif page == "results":
        page_results(split)
    else:
        page_causal(raw)


if __name__ == "__main__":
    main()
