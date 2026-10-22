"""External, chronology-aware validation on the Wang et al. second-life collection.

The released processed files contain one first-cycle diagnostic record and a
capacity trajectory for each of 86 NMC/graphite cells.  The target is a
physical endpoint at the first reference sample at or beyond 150 second-life
cycles (cycle 151 under the source's five-cycle sampling): capacity at least
70% of the 2.4 Ah nominal capacity.  Whole primary-use protocols are held out
to avoid treating cells exposed to one common history as independent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "wang"
OUT = ROOT / "results"


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, solver="liblinear", max_iter=5000)),
        ]
    )


def group_lopo(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, names: list[str]):
    pred = np.full(len(y), np.nan, dtype=float)
    for group in sorted(np.unique(groups)):
        train = groups != group
        test = groups == group
        model = make_pipeline()
        model.fit(X.loc[train], y[train])
        pred[test] = model.predict_proba(X.loc[test])[:, 1]
    if not np.isfinite(pred).all():
        raise RuntimeError("group-held-out prediction contains missing values")
    rows = []
    for name, threshold in [("0.50", 0.5)]:
        yp = (pred >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
        rows.append(
            {
                "stage": name,
                "balanced_accuracy": float(balanced_accuracy_score(y, yp)),
                "roc_auc": float(roc_auc_score(y, pred)),
                "brier": float(brier_score_loss(y, pred)),
                "positive_recall": float(tp / (tp + fn)) if tp + fn else np.nan,
                "negative_specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            }
        )
    return pred, rows[0]


def bootstrap_group_ba(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, seed: int = 20260910):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(np.unique(groups)))
    values = []
    for _ in range(1000):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        ix = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        yp = (pred[ix] >= 0.5).astype(int)
        if len(np.unique(y[ix])) < 2 or len(np.unique(yp)) < 2:
            # balanced accuracy is still defined in one-class samples, but a
            # cluster bootstrap should retain its observed class support.
            continue
        values.append(balanced_accuracy_score(y[ix], yp))
    if not values:
        return [np.nan, np.nan]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main():
    f = pd.read_csv(RAW / "wang_features.csv")
    traj = pd.read_csv(RAW / "wang_capacity_trajectories.csv")
    # The processed files use source-file numbers (`bat_num`), whereas the
    # article supplement reports experiment Battery_IDs.  Resolve that
    # identity explicitly from the first-row audit of every released source
    # file; never infer a protocol from the source-file number itself.
    identity = pd.DataFrame(json.loads(
        (ROOT / "datasets" / "wang_source_identity_map.json").read_text(encoding="utf-8")
    ))
    required_identity = {"source_id", "battery_id", "protocol", "status"}
    if not required_identity.issubset(identity.columns):
        raise ValueError(f"identity map is missing columns: {required_identity - set(identity.columns)}")
    if not (identity["status"].eq("ok").all()):
        raise ValueError("identity map contains non-ok source files")
    identity = identity[["source_id", "battery_id", "protocol"]].copy()
    identity["source_id"] = identity["source_id"].astype(int)
    identity["battery_id"] = identity["battery_id"].astype(int)
    identity["protocol"] = identity["protocol"].astype(int)
    if identity["source_id"].duplicated().any() or identity["battery_id"].duplicated().any():
        raise ValueError("identity map is not one-to-one")

    f = f.rename(columns={"bat_num": "source_id"}).copy()
    traj = traj.rename(columns={"bat_num": "source_id"}).copy()
    d = f.merge(identity, on="source_id", how="left", validate="one_to_one")
    if d["protocol"].isna().any():
        raise ValueError("feature records contain source IDs absent from identity map")

    # Cycle 151 is an observed reference point in the original five-cycle
    # schedule (1, 6, ..., 151), even though the released trajectory file is
    # interpolated to integer cycles. We select this exact reference point.
    endpoint = traj.loc[traj["cycle"].eq(151), ["source_id", "capacity"]].copy()
    endpoint = endpoint.rename(columns={"capacity": "capacity_151"})
    d = d.merge(endpoint, on="source_id", how="inner", validate="one_to_one")
    d["survive_151_at_70"] = (d["capacity_151"] >= 0.70 * 2.4).astype(int)
    d = d.sort_values("source_id").reset_index(drop=True)

    explicit = ["capacity", "voltage", "internal_resistance", "coulombic_efficiency", "energy_efficiency"]
    implicit = [c for c in d.columns if c.startswith("feature_")]
    stages = {
        "capacity_only": ["capacity"],
        "explicit_snapshot": explicit,
        "implicit_single_cycle": implicit,
        "explicit_plus_implicit": explicit + implicit,
    }
    y = d["survive_151_at_70"].to_numpy(dtype=int)
    groups = d["protocol"].to_numpy(dtype=int)
    summary = []
    pred_rows = d[["source_id", "battery_id", "protocol", "capacity_151", "survive_151_at_70"]].copy()
    for stage, cols in stages.items():
        pred, metrics = group_lopo(d[cols], y, groups, cols)
        ci_low, ci_high = bootstrap_group_ba(y, pred, groups)
        metrics.update(
            {
                "stage": stage,
                "n": int(len(d)),
                "n_protocols": int(d.protocol.nunique()),
                "positive": int(y.sum()),
                "negative": int((1 - y).sum()),
                "bootstrap_ba_low": ci_low,
                "bootstrap_ba_high": ci_high,
                "n_features": int(len(cols)),
            }
        )
        summary.append(metrics)
        pred_rows[f"p_{stage}"] = pred

    OUT.mkdir(exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT / "wang_external_validation_metrics.csv", index=False)
    pred_rows.to_csv(OUT / "wang_external_validation_predictions.csv", index=False)
    report = {
        "source": "Wang, Gao and Tian, Zenodo 10.5281/zenodo.14859405; processed files wang_features.csv and wang_capacity_trajectories.csv",
        "n_cells": int(len(d)),
        "n_protocols": int(d.protocol.nunique()),
        "protocol_counts": {str(k): int(v) for k, v in d.protocol.value_counts().sort_index().items()},
        "nominal_capacity_Ah": 2.4,
        "endpoint": "capacity at cycle 151 >= 70% of 2.4 Ah nominal; cycle 151 is the source five-cycle reference at or beyond 150",
        "protocol_mapping": "source-file bat_num values were joined to experiment Battery_ID and primary protocol using the audited identity map derived from the released ZIP member first rows and Supplementary Table S2",
        "positive": int(y.sum()),
        "negative": int((1 - y).sum()),
        "stages": summary,
        "interpretation": "independent physical replication of chronology-aware diagnostic screening; no route allocation, processing cost or information maintenance cost is observed",
    }
    (OUT / "wang_external_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
