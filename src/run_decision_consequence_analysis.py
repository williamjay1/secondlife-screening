"""Leakage-aware, pack-grouped analysis of the public Salinas battery records.

This script deliberately separates three quantities that are often conflated in
second-life screening studies:

* pre-decision passport/history fields (BMS and date fields),
* paid electrical diagnostics, and
* delayed 100-cycle performance outcomes.

The 2021 thesis data are not an independent collection campaign.  They are an
outcome-linked subset of the 2019 Salinas collection.  All predictive results
therefore remain within-source evidence.  Pack-level leave-one-pack-out
predictions are used to prevent cells from the same original pack appearing in
both training and test sets.  Test time comes from the exactly linked 2019
cell-level workbook; no currency or route payoff is inferred.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures_decision_consequence"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

SEED = 20260910
# Cluster bootstrap over only 16 source packs is inherently coarse; 1,000
# resamples keeps the confidence intervals reproducible without making the
# analysis depend on a long-running hyperparameter search.
BOOT = 1000


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def parse_date(series: pd.Series) -> pd.Series:
    """Parse ISO and Spanish abbreviated dates used by the thesis CSV."""
    s = series.astype(str).str.strip().str.lower()
    month_map = {
        "ene": "jan",
        "feb": "feb",
        "mar": "mar",
        "abr": "apr",
        "may": "may",
        "jun": "jun",
        "jul": "jul",
        "ago": "aug",
        "sept": "sep",
        "sep": "sep",
        "oct": "oct",
        "nov": "nov",
        "dic": "dec",
    }
    for old, new in month_map.items():
        s = s.str.replace(old, new, regex=False)
    return pd.to_datetime(s, dayfirst=True, errors="coerce", format="mixed")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ch = pd.read_csv(DATA / "salinas_characterization.csv", sep=";", dtype=str)
    perf = pd.read_csv(DATA / "salinas_performance.csv", sep=";", dtype=str)
    linked = pd.read_csv(DATA / "salinas_1034_linked_audit.csv", dtype=str)

    # Keep the 2019 fields needed for exact linkage and resource accounting.
    linked = linked[
        [
            "Code",
            "N Zell",
            "Model",
            "dn",
            "Vo",
            "Ct",
            "Cn",
            "Co",
            "En",
            "t",
            "Corrosion",
            "Burns",
            "Holes",
            "No conductivity",
            "High self-discharge",
            "cb",
            "zb",
            "db",
            "sb",
        ]
    ]
    assert not ch.duplicated(["Code", "N Zell"]).any()
    assert not perf.duplicated(["Code", "N Zell"]).any()
    assert not linked.duplicated(["Code", "N Zell"]).any()
    assert (
        ch.merge(perf, on=["Code", "N Zell"], how="outer", indicator=True)["_merge"]
        .eq("both")
        .all()
    )
    assert (
        ch.merge(linked, on=["Code", "N Zell"], how="outer", indicator=True)["_merge"]
        .eq("both")
        .sum()
        == len(ch)
    )
    return ch, perf, linked


def build_analysis_frame(
    ch: pd.DataFrame, perf: pd.DataFrame, linked: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = ch.merge(linked, on=["Code", "N Zell"], how="left", validate="one_to_one")
    p = perf.copy()
    perf_cols = [str(i) for i in range(1, 101)]
    for col in perf_cols:
        p[col] = num(p[col])

    # Numeric characterization columns. x29 is explicitly delayed by three
    # months in the thesis and is only used in a leakage/delay sensitivity.
    # Keep the three date fields as strings until their date differences are
    # computed.  The remaining characterization fields are numeric.
    xcols = [f"x{i}" for i in range(2, 30) if i not in (7, 8)]
    for col in xcols:
        d[col] = num(d[col])
    for col in ["Vo", "Ct", "Cn", "Co", "En", "t", "cb", "zb", "sb"]:
        d[col] = num(d[col])

    d["date_pack"] = parse_date(ch["x1"])
    d["date_cell"] = parse_date(ch["x7"])
    d["date_test"] = parse_date(ch["x8"])
    # The date columns are in ch and are retained in d by the merge above.
    d["date_pack"] = parse_date(d["x1"].astype(str))
    d["date_cell"] = parse_date(d["x7"].astype(str))
    d["date_test"] = parse_date(d["x8"].astype(str))
    d["age_pack_years"] = (d["date_test"] - d["date_pack"]).dt.days / 365.25
    d["age_cell_years"] = (d["date_test"] - d["date_cell"]).dt.days / 365.25
    d["bms_soh"] = d["x3"] / d["x4"]
    d["initial_capacity_frac"] = d["x14"] / d["Cn"]
    d["bms_cb_fraction"] = d["cb"] / 100.0
    d["group"] = d["Code"]
    d["y_int"] = d["y"].astype(int)
    d["any_second_life"] = (d["y_int"] >= 1).astype(int)

    # Capacity trajectories are the delayed performance outcome. The seven
    # truncated trajectories end below the 70% threshold; they are retained as
    # failures for the explicitly defined 100-cycle survival indicator.
    rows: list[dict] = []
    cycles = np.arange(1, 101)
    for _, row in p.iterrows():
        vals = row[perf_cols].to_numpy(dtype=float)
        mask = np.isfinite(vals)
        n_obs = int(mask.sum())
        x = cycles[mask]
        y = vals[mask]
        slope = float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else np.nan
        cap100 = float(vals[-1]) if n_obs == 100 else np.nan
        success = int(n_obs == 100 and cap100 >= 0.70)
        rows.append(
            {
                "Code": row["Code"],
                "N Zell": row["N Zell"],
                "n_cycles_observed": n_obs,
                "cap_cycle_1": float(vals[0]),
                "cap_cycle_50": float(vals[49]) if np.isfinite(vals[49]) else np.nan,
                "cap_cycle_100": cap100,
                "capacity_auc": float(np.trapezoid(y, x)),
                "capacity_slope": slope,
                "min_capacity_observed": float(np.nanmin(vals)),
                "survives_100_at_70pct": success,
            }
        )
    traj = pd.DataFrame(rows)
    d = d.merge(traj, on=["Code", "N Zell"], how="left", validate="one_to_one")
    d["complete_trajectory"] = d["n_cycles_observed"].eq(100)
    return d, traj


def pipeline_classifier(cols: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=3000,
                ),
            ),
        ]
    )


def lopo_classification(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    target: str,
    groups: str = "group",
    n_classes: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-pack-out predictions for fixed logistic models and a dummy."""
    y = frame[target].astype(int).to_numpy()
    g = frame[groups].astype(str).to_numpy()
    records: list[dict] = []
    unique_groups = sorted(pd.unique(g))
    if n_classes is None:
        n_classes = int(np.max(y)) + 1

    for stage, cols in feature_sets.items():
        for held in unique_groups:
            test = g == held
            train = ~test
            y_train = y[train]
            classes_train = np.unique(y_train)
            X_train = frame.loc[train, cols]
            X_test = frame.loc[test, cols]
            if len(classes_train) >= 2:
                model = pipeline_classifier(cols)
                model.fit(X_train, y_train)
                p0 = model.predict_proba(X_test)
                class_order = model.named_steps["model"].classes_
                p = np.zeros((test.sum(), n_classes), dtype=float)
                for j, cls in enumerate(class_order):
                    p[:, int(cls)] = p0[:, j]
                pred = np.argmax(p, axis=1)
                model_name = "logistic_l2"
            else:
                model = DummyClassifier(strategy="prior")
                model.fit(np.zeros((train.sum(), 1)), y_train)
                p = np.zeros((test.sum(), n_classes), dtype=float)
                p[:, int(classes_train[0])] = 1.0
                pred = np.repeat(int(classes_train[0]), test.sum())
                model_name = "constant_fallback"
            for idx, (row_idx, yy, pp, pr) in enumerate(
                zip(np.where(test)[0], y[test], p, pred)
            ):
                rec = {
                    "stage": stage,
                    "model": model_name,
                    "group": held,
                    "row_index": int(row_idx),
                    "y_true": int(yy),
                    "y_pred": int(pr),
                }
                for j in range(n_classes):
                    rec[f"p_{j}"] = float(pp[j])
                records.append(rec)

    pred_df = pd.DataFrame(records)
    metric_rows: list[dict] = []
    rng = np.random.default_rng(SEED)
    for stage in feature_sets:
        sub = pred_df[pred_df.stage == stage].sort_values("row_index")
        yy = sub.y_true.to_numpy(int)
        pp = sub[[f"p_{j}" for j in range(n_classes)]].to_numpy(float)
        yp = sub.y_pred.to_numpy(int)
        safe_pp = np.clip(pp, 1e-6, 1 - 1e-6)
        safe_pp = safe_pp / safe_pp.sum(axis=1, keepdims=True)
        metric_names: dict[str, float] = {
            "accuracy": accuracy_score(yy, yp),
            "balanced_accuracy": balanced_accuracy_score(yy, yp),
            "macro_f1": f1_score(yy, yp, average="macro", zero_division=0),
            "log_loss": log_loss(yy, safe_pp, labels=list(range(n_classes))),
        }
        if n_classes == 2:
            metric_names.update(
                {
                    "roc_auc": roc_auc_score(yy, pp[:, 1]),
                    "average_precision": average_precision_score(yy, pp[:, 1]),
                    "brier": brier_score_loss(yy, pp[:, 1]),
                    "eligible_recall": (yp[yy == 1] == 1).mean() if np.any(yy == 1) else np.nan,
                    "recycle_specificity": (yp[yy == 0] == 0).mean() if np.any(yy == 0) else np.nan,
                }
            )

        # Cluster bootstrap: resample packs, keeping all cells in a pack.
        metric_boot: dict[str, list[float]] = {name: [] for name in metric_names}
        group_values = sorted(sub.group.unique())
        # Convert the grouping column once: repeated to_numpy() calls on the
        # Arrow-backed string column dominate the run time otherwise.
        group_array = sub.group.to_numpy()
        idx_by_group = {h: np.flatnonzero(group_array == h) for h in group_values}
        for _ in range(BOOT):
            sampled = rng.choice(group_values, size=len(group_values), replace=True)
            ix = np.concatenate([idx_by_group[h] for h in sampled])
            b_y, b_p, b_pred = yy[ix], pp[ix], yp[ix]
            safe_b_p = np.clip(b_p, 1e-6, 1 - 1e-6)
            safe_b_p = safe_b_p / safe_b_p.sum(axis=1, keepdims=True)
            vals = {
                "accuracy": accuracy_score(b_y, b_pred),
                "balanced_accuracy": balanced_accuracy_score(b_y, b_pred),
                "macro_f1": f1_score(b_y, b_pred, average="macro", zero_division=0),
                "log_loss": log_loss(b_y, safe_b_p, labels=list(range(n_classes))),
            }
            if n_classes == 2:
                vals.update(
                    {
                        "roc_auc": roc_auc_score(b_y, b_p[:, 1]) if len(np.unique(b_y)) == 2 else np.nan,
                        "average_precision": average_precision_score(b_y, b_p[:, 1]),
                        "brier": brier_score_loss(b_y, b_p[:, 1]),
                        "eligible_recall": (b_pred[b_y == 1] == 1).mean() if np.any(b_y == 1) else np.nan,
                        "recycle_specificity": (b_pred[b_y == 0] == 0).mean() if np.any(b_y == 0) else np.nan,
                    }
                )
            for name, value in vals.items():
                if np.isfinite(value):
                    metric_boot[name].append(float(value))
        for name, value in metric_names.items():
            arr = np.asarray(metric_boot[name], dtype=float)
            metric_rows.append(
                {
                    "target": target,
                    "stage": stage,
                    "metric": name,
                    "estimate": float(value),
                    "ci_low": float(np.nanquantile(arr, 0.025)),
                    "ci_high": float(np.nanquantile(arr, 0.975)),
                    "n_cells": int(len(sub)),
                    "n_packs": int(sub.group.nunique()),
                }
            )
        # Confusion cells are useful for explaining concrete decision errors.
        cm = confusion_matrix(yy, yp, labels=list(range(n_classes)))
        for i in range(n_classes):
            for j in range(n_classes):
                metric_rows.append(
                    {
                        "target": target,
                        "stage": stage,
                        "metric": f"confusion_{i}_{j}",
                        "estimate": int(cm[i, j]),
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n_cells": int(len(sub)),
                        "n_packs": int(sub.group.nunique()),
                    }
                )
    return pred_df, pd.DataFrame(metric_rows)


def lopo_regression(
    frame: pd.DataFrame, feature_sets: dict[str, list[str]], target: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = frame[target].to_numpy(float)
    g = frame.group.astype(str).to_numpy()
    recs: list[dict] = []
    for stage, cols in feature_sets.items():
        for held in sorted(pd.unique(g)):
            test = g == held
            train = ~test
            model = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=1.0)),
                ]
            )
            model.fit(frame.loc[train, cols], y[train])
            pred = model.predict(frame.loc[test, cols])
            for row_idx, yy, pp in zip(np.where(test)[0], y[test], pred):
                recs.append(
                    {
                        "stage": stage,
                        "group": held,
                        "row_index": int(row_idx),
                        "y_true": float(yy),
                        "y_pred": float(pp),
                    }
                )
    pred = pd.DataFrame(recs)
    rows: list[dict] = []
    rng = np.random.default_rng(SEED + 1)
    for stage in feature_sets:
        sub = pred[pred.stage == stage].sort_values("row_index")
        yy, pp = sub.y_true.to_numpy(), sub.y_pred.to_numpy()
        def values(ix: np.ndarray) -> dict[str, float]:
            return {
                "mae": mean_absolute_error(yy[ix], pp[ix]),
                "rmse": float(np.sqrt(mean_squared_error(yy[ix], pp[ix]))),
                "r2": r2_score(yy[ix], pp[ix]),
                "spearman": spearmanr(yy[ix], pp[ix]).statistic,
            }
        est = values(np.arange(len(sub)))
        boot = {k: [] for k in est}
        groups = sorted(sub.group.unique())
        group_array = sub.group.to_numpy()
        idx_by_group = {h: np.flatnonzero(group_array == h) for h in groups}
        for _ in range(BOOT):
            sampled = rng.choice(groups, size=len(groups), replace=True)
            ix = np.concatenate([idx_by_group[h] for h in sampled])
            for k, v in values(ix).items():
                if np.isfinite(v):
                    boot[k].append(float(v))
        for k, v in est.items():
            rows.append(
                {
                    "target": target,
                    "stage": stage,
                    "metric": k,
                    "estimate": float(v),
                    "ci_low": float(np.nanquantile(boot[k], 0.025)),
                    "ci_high": float(np.nanquantile(boot[k], 0.975)),
                    "n_cells": int(len(sub)),
                    "n_packs": int(sub.group.nunique()),
                }
            )
    return pred, pd.DataFrame(rows)


def deterministic_predictions(d: pd.DataFrame) -> pd.DataFrame:
    """Capacity threshold rules, evaluated descriptively on the linked records."""
    out: list[dict] = []
    for target in ["any_second_life", "survives_100_at_70pct"]:
        sub = d if target == "any_second_life" else d[d.complete_trajectory].copy()
        for rule, p in [
            ("diagnostic_capacity_70", sub.initial_capacity_frac >= 0.70),
            ("diagnostic_capacity_80", sub.initial_capacity_frac >= 0.80),
            ("diagnostic_capacity_90", sub.initial_capacity_frac >= 0.90),
            ("passport_bms_70", sub.bms_soh >= 0.70),
        ]:
            y = sub[target].astype(int).to_numpy()
            pp = p.astype(int).to_numpy()
            out.extend(
                [
                    {
                        "target": target,
                        "rule": rule,
                        "metric": "accuracy",
                        "estimate": accuracy_score(y, pp),
                    },
                    {
                        "target": target,
                        "rule": rule,
                        "metric": "balanced_accuracy",
                        "estimate": balanced_accuracy_score(y, pp),
                    },
                    {
                        "target": target,
                        "rule": rule,
                        "metric": "eligible_recall",
                        "estimate": (pp[y == 1] == 1).mean() if np.any(y == 1) else np.nan,
                    },
                    {
                        "target": target,
                        "rule": rule,
                        "metric": "recycle_specificity",
                        "estimate": (pp[y == 0] == 0).mean() if np.any(y == 0) else np.nan,
                    },
                ]
            )
    return pd.DataFrame(out)


def resource_frontier(
    d: pd.DataFrame, pred: pd.DataFrame, target: str, output_name: str
) -> pd.DataFrame:
    sub = d if target == "any_second_life" else d[d.complete_trajectory].copy()
    sub = sub.reset_index(drop=True)
    # lopo predictions were generated on the original frame; align by row index.
    pred = pred[pred.target == target].copy()
    rows: list[dict] = []
    # Only information available before completion of the reference test can
    # define a stopping policy.  Full diagnostics and the delayed field remain
    # prediction benchmarks, but using them to claim that the same completed
    # test could have been stopped would be a timing error.
    deployable_stages = {"passport", "passport_plus_physical", "early_diagnostics"}
    for stage in sorted(set(pred.stage.unique()) & deployable_stages):
        pp = pred[pred.stage == stage].sort_values("row_index")
        # the subset's original indices are preserved in row_index
        pp = pp.set_index("row_index").loc[sub.index]
        prob = pp.p_1.to_numpy(float)
        y = sub[target].astype(int).to_numpy()
        t = sub.t.to_numpy(float)
        for threshold in np.linspace(0, 1, 101):
            eligible = prob >= threshold
            positive = y == 1
            negative = y == 0
            rows.append(
                {
                    "target": target,
                    "stage": stage,
                    "threshold": threshold,
                    "eligible_recall": (eligible[positive]).mean() if positive.any() else np.nan,
                    "recycle_specificity": (~eligible[negative]).mean() if negative.any() else np.nan,
                    "false_reject_rate": (~eligible[positive]).mean() if positive.any() else np.nan,
                    "false_accept_rate": (eligible[negative]).mean() if negative.any() else np.nan,
                    "electrical_test_hours_saved": float(t[~eligible].sum()),
                    "electrical_test_hours_at_risk": float(t[(~eligible) & positive].sum()),
                    "fraction_test_hours_saved": float(t[~eligible].sum() / t.sum()),
                    "n_stopped": int((~eligible).sum()),
                    "n_false_reject": int(((~eligible) & positive).sum()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / output_name, index=False)
    return out


def pack_audit(linked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = linked.copy()
    for col in ["t", "Co", "cb", "zb", "sb"]:
        p[col] = num(p[col])
    p["eligible_fraction"] = p["Co"].ge(70).astype(float)
    pack = (
        p.groupby("Code", as_index=False)
        .agg(
            n_cells=("Code", "size"),
            electrical_cells=("Ct", lambda s: s.notna().sum()),
            measured_pass70=("Co", lambda s: (num(s).ge(70)).sum()),
            sb=("sb", "first"),
            cb=("cb", "first"),
            zb=("zb", "first"),
            db=("db", "first"),
            test_hours=("t", "sum"),
        )
    )
    pack["measured_fraction"] = pack.measured_pass70 / pack.n_cells
    pack["cb_fraction"] = pack.cb / 100.0
    rows: list[dict] = []
    for field in ["cb", "zb"]:
        ok = pack[[field, "measured_fraction"]].dropna()
        rho, pval = spearmanr(ok[field], ok.measured_fraction)
        rows.append(
            {
                "field": field,
                "n_packs": len(ok),
                "spearman_rho": rho,
                "p_value": pval,
            }
        )
    # Field quartiles provide a transparent, model-free information-value table.
    qrows: list[dict] = []
    for field in ["cb", "zb"]:
        ok = pack.dropna(subset=[field]).copy()
        ok["quartile"] = pd.qcut(ok[field], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        for q, g in ok.groupby("quartile", observed=True):
            qrows.append(
                {
                    "field": field,
                    "quartile": str(q),
                    "n_packs": len(g),
                    "mean_field": g[field].mean(),
                    "mean_measured_fraction": g.measured_fraction.mean(),
                    "total_test_hours": g.test_hours.sum(),
                    "mean_test_hours": g.test_hours.mean(),
                }
            )
    return pack, pd.DataFrame(rows + qrows)


def make_figures(
    d: pd.DataFrame,
    metrics: pd.DataFrame,
    front: pd.DataFrame,
    pack: pd.DataFrame,
) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

    # 1. delayed performance trajectories, shown by the source-defined class.
    perf = pd.read_csv(DATA / "salinas_performance.csv", sep=";", dtype=str)
    cols = [str(i) for i in range(1, 101)]
    for c in cols:
        perf[c] = num(perf[c])
    yy = d[["Code", "N Zell", "y_int"]].merge(perf, on=["Code", "N Zell"])
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    colors = {0: "#4c78a8", 1: "#f58518", 2: "#54a24b"}
    labels = {0: "recycle", 1: "repurpose", 2: "reuse"}
    x = np.arange(1, 101)
    for cls in [0, 1, 2]:
        arr = yy.loc[yy.y_int == cls, cols].to_numpy(float)
        mean = np.nanmean(arr, axis=0)
        se = np.nanstd(arr, axis=0, ddof=1) / math.sqrt(arr.shape[0])
        ax.plot(x, mean, label=labels[cls], color=colors[cls], lw=1.8)
        ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=colors[cls], alpha=0.14, linewidth=0)
    ax.axhline(0.7, color="black", lw=0.8, ls="--")
    ax.set(xlabel="Second-life cycle", ylabel="Relative capacity", ylim=(0, 1.02))
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "delayed_performance_by_route.pdf")
    fig.savefig(FIGURES / "delayed_performance_by_route.png", dpi=1000)
    plt.close(fig)

    # 2. grouped leave-one-pack-out balanced accuracy for the two decision targets.
    m = metrics[metrics.metric == "balanced_accuracy"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, target in zip(axes, ["any_second_life", "survives_100_at_70pct"]):
        q = m[m.target == target].copy()
        order = list(q.stage.drop_duplicates())
        q["x"] = [order.index(v) for v in q.stage]
        ax.errorbar(q.x, q.estimate, yerr=[q.estimate - q.ci_low, q.ci_high - q.estimate], fmt="o", capsize=3, color="#2f4b7c")
        ax.set_xticks(range(len(order)), order, rotation=35, ha="right")
        ax.set_title("Any second life" if target == "any_second_life" else "100-cycle retention")
        ax.set_ylim(0.35, 1.02)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Balanced accuracy (pack bootstrap 95% CI)")
    fig.tight_layout()
    fig.savefig(FIGURES / "grouped_information_blocks.pdf")
    fig.savefig(FIGURES / "grouped_information_blocks.png", dpi=1000)
    plt.close(fig)

    # 3. resource/risk frontier for the primary target.
    f = front[front.target == "any_second_life"].copy()
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    for stage, g in f.groupby("stage"):
        g = g.sort_values("fraction_test_hours_saved")
        ax.plot(g.fraction_test_hours_saved, g.eligible_recall, lw=1.6, label=stage)
    ax.set(xlabel="Fraction of recorded electrical-test hours stopped", ylabel="Recall of any second-life cell", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "resource_risk_frontier.pdf")
    fig.savefig(FIGURES / "resource_risk_frontier.png", dpi=1000)
    plt.close(fig)

    # 4. pack-level BMS field vs measured screening yield.
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    s = pack.dropna(subset=["cb"])
    size = 18 + 12 * (s.test_hours - s.test_hours.min()) / max(1e-9, s.test_hours.max() - s.test_hours.min())
    ax.scatter(s.cb, s.measured_fraction, s=size, alpha=0.75, color="#7a5195", edgecolor="white", linewidth=0.4)
    ax.plot([0, 100], [0, 1], color="black", lw=0.7, ls="--", alpha=0.5)
    ax.set(xlabel="BMS remaining capacity cb (%)", ylabel="Measured fraction at or above 70% capacity", xlim=(0, 105), ylim=(-0.03, 1.03))
    fig.tight_layout()
    fig.savefig(FIGURES / "bms_field_screening_yield.pdf")
    fig.savefig(FIGURES / "bms_field_screening_yield.png", dpi=1000)
    plt.close(fig)


def main() -> None:
    ch, perf, linked = load_data()
    d, traj = build_analysis_frame(ch, perf, linked)
    d.to_csv(DATA / "salinas_2021_decision_frame.csv", index=False)
    traj.to_csv(RESULTS / "salinas_2021_trajectory_summary.csv", index=False)

    passport = ["age_pack_years", "age_cell_years", "x2", "bms_soh"]
    physical = passport + ["x5", "x6"]
    early = physical + ["x9", "x10", "x11"]
    full = early + [f"x{i}" for i in range(12, 29)]
    delayed = full + ["x29"]
    stages = {
        "passport": passport,
        "passport_plus_physical": physical,
        "early_diagnostics": early,
        "full_diagnostics": full,
        "full_plus_delayed_x29": delayed,
    }

    # Primary route consequence: whether the source's delayed class assigns any
    # second-life route. Secondary target: physical 100-cycle retention.
    p_any, m_any = lopo_classification(d, stages, "any_second_life", n_classes=2)
    d_complete = d[d.complete_trajectory].copy()
    p_surv, m_surv = lopo_classification(d_complete, stages, "survives_100_at_70pct", n_classes=2)
    # Secondary three-class outcome, retaining the original source labels.
    p_three, m_three = lopo_classification(d, stages, "y_int", n_classes=3)
    metrics = pd.concat([m_any, m_surv, m_three], ignore_index=True)
    predictions = pd.concat(
        [p_any.assign(target="any_second_life"), p_surv.assign(target="survives_100_at_70pct"), p_three.assign(target="y_int")],
        ignore_index=True,
    )
    predictions.to_csv(RESULTS / "salinas_grouped_predictions.csv", index=False)
    metrics.to_csv(RESULTS / "salinas_grouped_metrics.csv", index=False)

    # Large linked capacity audit.  This endpoint is measured capacity in the
    # 2019 workbook, not the delayed route label.  It provides a separate
    # pack-held-out check of whether the predecision BMS fields carry signal
    # beyond the 100-cell performance subset.
    capacity = linked.copy()
    for col in ["Co", "Cn", "cb", "zb"]:
        capacity[col] = num(capacity[col])
    capacity = capacity[capacity.Co.notna() & capacity.Cn.notna()].copy().reset_index(drop=True)
    capacity["capacity_70"] = capacity.Co.ge(70).astype(int)
    capacity["group"] = capacity.Code.astype(str)
    capacity_stages = {
        "bms_capacity": ["cb"],
        "bms_capacity_plus_cycles": ["cb", "zb"],
    }
    p_capacity, m_capacity = lopo_classification(
        capacity, capacity_stages, "capacity_70", groups="group", n_classes=2
    )
    p_capacity.to_csv(RESULTS / "salinas_2019_capacity_predictions.csv", index=False)
    m_capacity.to_csv(RESULTS / "salinas_2019_capacity_metrics.csv", index=False)

    # Continuous delayed endpoint, only where the 100-cycle trajectory exists.
    reg_stages = {k: v for k, v in stages.items()}
    rpred, rmetrics = lopo_regression(d_complete, reg_stages, "cap_cycle_100")
    rpred.to_csv(RESULTS / "salinas_grouped_regression_predictions.csv", index=False)
    rmetrics.to_csv(RESULTS / "salinas_grouped_regression_metrics.csv", index=False)

    deterministic_predictions(d).to_csv(RESULTS / "salinas_deterministic_rules.csv", index=False)
    front_any = resource_frontier(d, p_any.assign(target="any_second_life"), "any_second_life", "salinas_resource_frontier_any_second_life.csv")
    front_surv = resource_frontier(d_complete, p_surv.assign(target="survives_100_at_70pct"), "survives_100_at_70pct", "salinas_resource_frontier_survival.csv")
    front = pd.concat([front_any, front_surv], ignore_index=True)

    pack, pack_stats = pack_audit(linked)
    pack.to_csv(RESULTS / "salinas_pack_yield_audit.csv", index=False)
    pack_stats.to_csv(RESULTS / "salinas_pack_field_yield_stats.csv", index=False)

    linkage = {
        "n_characterization_cells": int(len(ch)),
        "n_performance_cells": int(len(perf)),
        "n_exact_links_to_2019_cells": int(len(d)),
        "all_2021_cells_linked": bool(len(d) == len(ch)),
        "all_2021_cells_have_recorded_test_time": bool(d.t.notna().all()),
        "all_2021_cells_have_2019_visual_reject_false": bool((d[["Corrosion", "Burns", "Holes"]] == "False").all().all()),
        "exact_numeric_matches_2021_to_2019": {
            "x6_equals_Vo": bool(np.allclose(d.x6, d.Vo, equal_nan=False)),
            "x14_equals_Ct": bool(np.allclose(d.x14, d.Ct, equal_nan=False)),
            "x15_equals_En": bool(np.allclose(d.x15, d.En, equal_nan=False)),
        },
        "pack_level_history_matches": {
            "x2_equals_zb": bool(np.allclose(d.groupby("Code").x2.first().sort_index(), d.groupby("Code").zb.first().sort_index(), equal_nan=False)),
            # cb is rounded in the 2019 workbook; the maximum discrepancy is
            # reported rather than silently calling the fields identical.
            "x3_over_x4_matches_cb_fraction_within_0.1pct": bool(
                np.allclose(
                    d.groupby("Code").bms_soh.first().sort_index(),
                    (d.groupby("Code").cb.first() / 100.0).sort_index(),
                    atol=1e-3,
                    rtol=0,
                    equal_nan=False,
                )
            ),
            "max_abs_cb_fraction_difference": float(
                np.max(
                    np.abs(
                        d.groupby("Code").bms_soh.first().sort_index()
                        - (d.groupby("Code").cb.first() / 100.0).sort_index()
                    )
                )
            ),
        },
        "n_packs_2021": int(d.group.nunique()),
        "n_packs_2019": int(linked.Code.nunique()),
        "capacity_audit": {
            "n_cells_with_measured_capacity": int(len(capacity)),
            "n_packs": int(capacity.group.nunique()),
            "n_capacity_70": int(capacity.capacity_70.sum()),
        },
        "class_counts": d.y_int.value_counts().sort_index().to_dict(),
        "complete_trajectory_cells": int(d.complete_trajectory.sum()),
        "survives_100_at_70pct": int(d.survives_100_at_70pct.sum()),
        "source_limit": "The 2021 outcome subset is linked to the same Salinas collection; it is not independent external-batch validation.",
    }
    (RESULTS / "salinas_decision_linkage_audit.json").write_text(json.dumps(linkage, indent=2, default=str), encoding="utf-8")
    make_figures(d, metrics, front, pack)

    summary = {
        "n_cells": len(d),
        "n_packs_2021": int(d.group.nunique()),
        "any_second_life_counts": d.any_second_life.value_counts().sort_index().to_dict(),
        "survival_counts_complete": d_complete.survives_100_at_70pct.value_counts().sort_index().to_dict(),
        "passport_any_balanced_accuracy": float(m_any.query("stage == 'passport' and metric == 'balanced_accuracy'").estimate.iloc[0]),
        "full_any_balanced_accuracy": float(m_any.query("stage == 'full_diagnostics' and metric == 'balanced_accuracy'").estimate.iloc[0]),
        "passport_survival_balanced_accuracy": float(m_surv.query("stage == 'passport' and metric == 'balanced_accuracy'").estimate.iloc[0]),
        "full_survival_balanced_accuracy": float(m_surv.query("stage == 'full_diagnostics' and metric == 'balanced_accuracy'").estimate.iloc[0]),
        "note": "Metrics are pack-grouped within-source estimates, not external validation or realized route profit.",
    }
    (RESULTS / "salinas_decision_analysis_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(metrics.query("metric in ['balanced_accuracy','macro_f1','roc_auc','eligible_recall','recycle_specificity']").to_string(index=False))
    print(rmetrics.to_string(index=False))


if __name__ == "__main__":
    main()
