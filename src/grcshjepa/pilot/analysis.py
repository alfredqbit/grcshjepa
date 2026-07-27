
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from grcshjepa.stats.decision import bootstrap_ci
from grcshjepa.utils import save_json


ID_COLUMNS = {
    "phase", "study", "task", "predictor", "variant", "damage_type", "damage_level",
    "head_type", "seed", "status", "failure_type", "failure_message",
}


def load_pilot_results(results_dir: str | Path) -> pd.DataFrame:
    results_dir = Path(results_dir)
    preferred = results_dir / "phase1_pilot_combined_results.csv"
    if preferred.exists():
        return pd.read_csv(preferred)
    csvs = sorted(results_dir.glob("**/*.csv"))
    frames = []
    for path in csvs:
        if "summary" in path.name.lower() or "readiness" in path.name.lower():
            continue
        try:
            df = pd.read_csv(path)
            df["source_csv"] = str(path)
            frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _group_columns(df: pd.DataFrame) -> list[str]:
    candidates = ["study", "task", "predictor", "variant", "damage_type", "damage_level", "head_type"]
    return [c for c in candidates if c in df.columns and df[c].notna().any()]


def summarize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    group_cols = _group_columns(df)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in {"seed"}]
    if not numeric_cols:
        return pd.DataFrame()
    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for key, g in grouped:
        base = dict(zip(group_cols, key if isinstance(key, tuple) else (key,))) if group_cols else {}
        for col in numeric_cols:
            vals = g[col].dropna().astype(float).to_numpy()
            if vals.size == 0:
                continue
            lo, hi = bootstrap_ci(vals, n_boot=1000, seed=17) if vals.size >= 2 else (float("nan"), float("nan"))
            rows.append({
                **base,
                "metric": col,
                "n_rows": int(vals.size),
                "n_seeds": int(g["seed"].nunique()) if "seed" in g else int(vals.size),
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0,
                "median": float(np.median(vals)),
                "q25": float(np.quantile(vals, 0.25)),
                "q75": float(np.quantile(vals, 0.75)),
                "boot_ci_low": lo,
                "boot_ci_high": hi,
            })
    return pd.DataFrame(rows)


def _suggest_n(sd: float, target_half_width: float, max_n: int = 20) -> int | None:
    if not np.isfinite(sd) or sd <= 0 or target_half_width <= 0:
        return None
    n = math.ceil((1.96 * sd / target_half_width) ** 2)
    return int(min(max(n, 2), max_n))


def readiness_report(df: pd.DataFrame, summary: pd.DataFrame) -> dict:
    report: dict = {
        "phase": "phase1_pilot",
        "interpretation": "Pilot hardening only. These values estimate runtime, variance, failure modes, and metric sanity; they are not confirmatory dissertation results.",
        "rows": int(len(df)),
        "studies_present": sorted([str(x) for x in df.get("study", pd.Series(dtype=str)).dropna().unique()]),
        "seeds_present": sorted([int(x) for x in df.get("seed", pd.Series(dtype=float)).dropna().unique()]) if "seed" in df else [],
        "status_counts": df.get("status", pd.Series(dtype=str)).fillna("unknown").value_counts().to_dict() if not df.empty else {},
        "failure_rate": None,
        "suggested_confirmatory_n": [],
        "gate_checks": [],
        "next_actions": [
            "Inspect failure manifests and rerun only administrative failures under the same seed.",
            "Replace remaining toy/smoke stand-ins with production architecture modules where needed.",
            "Freeze model variants, primary endpoints, seed list, hyperparameter budget, and analysis scripts before confirmatory runs.",
            "Use pilot variance to select 12 to 20 independent confirmatory seeds per primary variant.",
        ],
    }
    if not df.empty and "status" in df:
        n = len(df)
        failures = int((df["status"].fillna("") == "failure").sum())
        report["failure_rate"] = failures / max(1, n)

    if not summary.empty:
        # Heuristic target half-widths for pilot planning only.
        target_by_metric = {
            "val_pred_loss": 0.02,
            "test_acc": 0.03,
            "test_mse": 0.02,
            "exactish_rate": 0.03,
            "normalized_surface": 0.05,
            "traffic_degradation": 1.0,
            "surface_degradation": 0.05,
        }
        for _, row in summary.iterrows():
            metric = row.get("metric")
            if metric in target_by_metric and row.get("n_seeds", 0) >= 2:
                n_suggest = _suggest_n(float(row.get("sd", float("nan"))), target_by_metric[metric])
                if n_suggest is not None:
                    item = {k: row[k] for k in ["study", "task", "predictor", "variant", "damage_type", "damage_level", "head_type"] if k in row and pd.notna(row[k])}
                    item.update({"metric": metric, "pilot_sd": float(row["sd"]), "target_half_width": target_by_metric[metric], "suggested_n_cap20": n_suggest})
                    report["suggested_confirmatory_n"].append(item)

    # Coarse gate checks. Human review is still required.
    if report["failure_rate"] is not None:
        report["gate_checks"].append({
            "check": "failure_rate_below_10_percent",
            "passed": bool(report["failure_rate"] <= 0.10),
            "value": report["failure_rate"],
        })
    if not summary.empty:
        ac = summary[(summary["metric"].astype(str).str.contains("effective_rank", case=False, na=False)) | (summary["metric"].astype(str).str.contains("eff_rank", case=False, na=False))]
        if not ac.empty:
            report["gate_checks"].append({"check": "effective_rank_logged", "passed": True, "rows": int(len(ac))})
        val_pred = summary[summary["metric"] == "val_pred_loss"]
        report["gate_checks"].append({"check": "val_prediction_loss_logged", "passed": bool(len(val_pred) > 0), "rows": int(len(val_pred))})
        surf = summary[summary["metric"] == "normalized_surface"]
        report["gate_checks"].append({"check": "routing_surface_logged", "passed": bool(len(surf) > 0), "rows": int(len(surf))})
    return report


def write_markdown_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    lines = [
        "# GR-CS-HJEPA Phase 1 Pilot Readiness Report",
        "",
        report["interpretation"],
        "",
        f"Rows analyzed: {report['rows']}",
        f"Studies present: {', '.join(report['studies_present'])}",
        f"Seeds present: {', '.join(map(str, report['seeds_present']))}",
        f"Status counts: {report['status_counts']}",
        f"Failure rate: {report['failure_rate']}",
        "",
        "## Gate checks",
    ]
    for g in report.get("gate_checks", []):
        lines.append(f"- {g.get('check')}: {'PASS' if g.get('passed') else 'REVIEW'} ({g})")
    lines.extend(["", "## Suggested confirmatory seed counts from pilot variance", ""])
    if report.get("suggested_confirmatory_n"):
        for item in report["suggested_confirmatory_n"][:60]:
            lines.append(f"- {item}")
    else:
        lines.append("- No seed-count suggestions were generated. This usually means too few seeds or missing target metrics.")
    lines.extend(["", "## Next actions", ""])
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    path.write_text("\n".join(lines))


def analyze_phase1_pilot(results_dir: str | Path, output_dir: str | Path) -> dict:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_pilot_results(results_dir)
    df.to_csv(output_dir / "phase1_pilot_loaded_results.csv", index=False)
    summary = summarize_numeric(df)
    summary.to_csv(output_dir / "phase1_pilot_numeric_summary.csv", index=False)
    report = readiness_report(df, summary)
    save_json(report, output_dir / "phase1_pilot_readiness_report.json")
    write_markdown_report(report, output_dir / "phase1_pilot_readiness_report.md")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase 1 pilot hardening outputs.")
    parser.add_argument("--results-dir", default="runs/phase1_pilot")
    parser.add_argument("--output-dir", default="analysis/phase1_pilot")
    args = parser.parse_args()
    report = analyze_phase1_pilot(args.results_dir, args.output_dir)
    print(pd.Series({
        "rows": report["rows"],
        "studies_present": ",".join(report["studies_present"]),
        "seeds_present": ",".join(map(str, report["seeds_present"])),
        "failure_rate": report["failure_rate"],
    }).to_string())
    print(f"Report written to {Path(args.output_dir) / 'phase1_pilot_readiness_report.md'}")


if __name__ == "__main__":
    main()
