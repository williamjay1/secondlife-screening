"""Build the supplementary tables for the second-life screening manuscript.

Table S1  variable identifiers mapped to measurement family and availability
Table S2  full threshold sweep of the stopping replay
Table S3  complete discrimination, calibration and regression metrics
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
OUT = ROOT / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

# Measurement family per source column. Identifiers follow the column order of
# the released characterization workbook; the family assignment follows the
# block definition used in the analysis.
VARIABLE_MAP = [
    ("x1", "Pack level commissioning date", "Battery history"),
    ("x2", "Battery management system cycle count", "Battery history"),
    ("x3", "Battery management system remaining capacity", "Battery history"),
    ("x4", "Battery management system rated capacity", "Battery history"),
    ("x5", "Cell mass", "Physical observation"),
    ("x6", "Initial open circuit voltage", "Physical observation"),
    ("x7", "Cell manufacturing date", "Battery history"),
    ("x8", "Characterization test date", "Timing reference"),
    ("x9", "Short circuit flag", "Early diagnostics"),
    ("x10", "Transient voltage at the start of the first discharge step", "Early diagnostics"),
    ("x11", "Transient voltage at the end of the first discharge step", "Early diagnostics"),
    ("x12", "Voltage at the end of the reference discharge", "Full characterization"),
    ("x13", "Reference discharge voltage loss indicator", "Full characterization"),
    ("x14", "Measured discharge capacity", "Full characterization"),
    ("x15", "Measured discharge energy", "Full characterization"),
    ("x16", "Voltage at the start of the reference charge", "Full characterization"),
    ("x17", "Voltage at the end of the reference charge", "Full characterization"),
    ("x18", "Open circuit voltage after the reference charge", "Full characterization"),
    ("x19", "Reference charge acceptance indicator", "Full characterization"),
    ("x20", "Charge capacity of the reference charge", "Full characterization"),
    ("x21", "Charge energy of the reference charge", "Full characterization"),
    ("x22", "Partial discharge flag", "Full characterization"),
    ("x23", "Voltage after the partial discharge", "Full characterization"),
    ("x24", "Voltage after the recovery rest", "Full characterization"),
    ("x25", "Discharge capacity of the partial discharge", "Full characterization"),
    ("x26", "Discharge energy of the partial discharge", "Full characterization"),
    ("x27", "Voltage at the start of the recovery step", "Full characterization"),
    ("x28", "Voltage at the end of the recovery step", "Full characterization"),
    ("x29", "Float voltage after approximately three months of storage", "Delayed measurement"),
    ("cb", "Pack level battery management system remaining capacity", "Battery history"),
    ("zb", "Pack level battery management system cycle count", "Battery history"),
    ("db", "Pack level diagnostic status field", "Battery history"),
    ("sb", "Fraction of pack cells at or above 70 percent capacity", "Excluded as outcome-derived"),
]

VARIABLE_NOTE = (
    "Measurement families follow the information block definition used in the analysis and the "
    "released source documentation. x1 and x7 are the two pre-test date fields retained for age "
    "derivation, and x7 precedes x8 for all 100 characterized cells. Variables used in the "
    "deployable blocks are x2 to x6 and x9 to x11. The pack yield summary sb is excluded from all "
    "models because it reproduces the post-characterization fraction of cells at or above 70 "
    "percent capacity."
)

STAGE_LABEL = {
    "passport": "Battery history",
    "passport_plus_physical": "Plus physical observations",
    "early_diagnostics": "Plus early diagnostics",
    "full_diagnostics": "Plus full characterization",
    "full_plus_delayed_x29": "Plus delayed measurement",
    "capacity_only": "Capacity only",
    "explicit_snapshot": "Five explicit diagnostics",
    "implicit_single_cycle": "108 implicit features",
    "explicit_plus_implicit": "Explicit and implicit",
}
TARGET_LABEL = {
    "any_second_life": "Second-life eligibility",
    "survives_100_at_70pct": "100-cycle retention",
    "y_int": "Three-class source label",
    "cap_cycle_100": "Cycle 100 relative capacity",
}


def write_table_s1() -> None:
    rows = [
        {
            "Identifier": ident,
            "Measurement family": family,
            "Information block": block,
            "Available before the paid test": "yes" if block in {"Battery history", "Physical observation"} else ("yes, early" if block == "Early diagnostics" else "no"),
        }
        for ident, family, block in VARIABLE_MAP
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "table_s1_variable_map.csv", index=False)
    (OUT / "table_s1_note.txt").write_text(VARIABLE_NOTE + "\n", encoding="utf-8")
    print("S1", df.shape)
    print(df.to_string(index=False))


def write_table_s2() -> None:
    frames = []
    for name, target in [
        ("salinas_resource_frontier_any_second_life.csv", "any_second_life"),
        ("salinas_resource_frontier_survival.csv", "survives_100_at_70pct"),
    ]:
        path = RESULTS / name
        if not path.exists():
            continue
        d = pd.read_csv(path)
        d["target_label"] = d["target"].map(TARGET_LABEL).fillna(d["target"])
        d["block_label"] = d["stage"].map(STAGE_LABEL).fillna(d["stage"])
        frames.append(
            d[[
                "target_label",
                "block_label",
                "threshold",
                "eligible_recall",
                "recycle_specificity",
                "n_stopped",
                "n_false_reject",
                "electrical_test_hours_saved",
                "fraction_test_hours_saved",
            ]].rename(
                columns={
                    "target_label": "Target",
                    "block_label": "Information block",
                    "threshold": "Threshold",
                    "eligible_recall": "Recall of eligible cells",
                    "recycle_specificity": "Specificity for non-eligible cells",
                    "n_stopped": "Tests stopped",
                    "n_false_reject": "False rejections",
                    "electrical_test_hours_saved": "Stopped hours",
                    "fraction_test_hours_saved": "Stopped hours fraction",
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT / "table_s2_threshold_sweep.csv", index=False)
    print("S2", out.shape)


def write_table_s3() -> None:
    metrics = pd.read_csv(RESULTS / "salinas_grouped_metrics.csv")
    metrics = metrics[metrics.target.isin(["any_second_life", "survives_100_at_70pct"])].copy()
    metrics["Target"] = metrics["target"].map(TARGET_LABEL)
    metrics["Information block"] = metrics["stage"].map(STAGE_LABEL)
    wide = metrics.pivot_table(
        index=["Target", "Information block", "metric"],
        values=["estimate", "ci_low", "ci_high"],
        aggfunc="first",
    ).reset_index()
    wide = wide.rename(
        columns={
            "metric": "Metric",
            "estimate": "Estimate",
            "ci_low": "Interval low",
            "ci_high": "Interval high",
        }
    )
    order = ["balanced_accuracy", "accuracy", "macro_f1", "roc_auc", "brier", "eligible_recall", "recycle_specificity"]
    wide["_o"] = wide["Metric"].apply(lambda m: order.index(m) if m in order else len(order))
    wide = wide.sort_values(["Target", "Information block", "_o"]).drop(columns="_o")

    # The external-validation model is run separately; fall back to the
    # published outputs when the current run has not regenerated them.
    wang_path = RESULTS / "wang_external_validation_metrics.csv"
    if not wang_path.exists():
        wang_path = ROOT / "results_reference" / "wang_external_validation_metrics.csv"
    wang = pd.read_csv(wang_path)
    wang_rows = []
    for _, r in wang.iterrows():
        for metric, col in [
            ("balanced_accuracy", "balanced_accuracy"),
            ("roc_auc", "roc_auc"),
            ("brier", "brier"),
            ("eligible_recall", "positive_recall"),
            ("recycle_specificity", "negative_specificity"),
        ]:
            wang_rows.append(
                {
                    "Target": "Independent cohort, cycle 151 retention",
                    "Information block": STAGE_LABEL.get(r["stage"], r["stage"]),
                    "Metric": metric,
                    "Estimate": r[col],
                    "Interval low": r["bootstrap_ba_low"] if metric == "balanced_accuracy" else None,
                    "Interval high": r["bootstrap_ba_high"] if metric == "balanced_accuracy" else None,
                }
            )
    wang_df = pd.DataFrame(wang_rows)

    reg = pd.read_csv(RESULTS / "salinas_grouped_regression_metrics.csv")
    reg["Target"] = reg["target"].map(TARGET_LABEL)
    reg["Information block"] = reg["stage"].map(STAGE_LABEL)
    reg = reg.rename(
        columns={
            "metric": "Metric",
            "estimate": "Estimate",
            "ci_low": "Interval low",
            "ci_high": "Interval high",
        }
    )[["Target", "Information block", "Metric", "Estimate", "Interval low", "Interval high"]]

    out = pd.concat([wide, wang_df, reg], ignore_index=True)
    out = out[["Target", "Information block", "Metric", "Estimate", "Interval low", "Interval high"]]
    out.to_csv(OUT / "table_s3_full_metrics.csv", index=False)
    print("S3", out.shape)


def main() -> int:
    write_table_s1()
    write_table_s2()
    write_table_s3()
    print("\nsupplementary tables written to", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
