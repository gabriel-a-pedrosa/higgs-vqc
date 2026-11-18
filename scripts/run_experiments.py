"""Rank the features, train the VQC on the top 3 and top 5, and draw the figures.

Needs the ATLAS CSV, see data/README.md.

    python scripts/run_experiments.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from higgs_vqc import plotting
from higgs_vqc.config import DEFAULT_FEATURES
from higgs_vqc.experiments import best_config, ensemble_ams
from higgs_vqc.ranking import feature_ranking, top_features

RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "experiments" / "figures"
SEEDS = [1, 2, 3, 4, 5]


def save(obj, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def main():
    print("Feature ranking ...")
    ranking = feature_ranking()
    save(ranking, "feature_ranking.json")
    plotting.plot_feature_ranking(ranking, FIGURES)
    top5 = top_features(ranking, 5)
    print("Top 5:", top5)
    if top5 != DEFAULT_FEATURES:
        print("Note: the ranking differs from DEFAULT_FEATURES in config.py")

    runs = {}
    for k in (3, 5):
        print(f"VQC on the top {k} features ...")
        res = ensemble_ams(seeds=SEEDS, base=best_config(features=top5[:k]), verbose=True)
        res["config"]["output_dir"] = "experiments"  # keep local paths out of the JSON
        runs[k] = res
        save(res, f"vqc_top{k}.json")

    steps = [
        ("Top 3 features (3 qubits)", runs[3]["single_mean"], runs[3]["single_std"]),
        ("Top 5 features (5 qubits)", runs[5]["single_mean"], runs[5]["single_std"]),
        (f"Top 5, average of {len(SEEDS)} VQCs", runs[5]["ensemble_ams"], 0.0),
    ]
    plotting.plot_ams_results(steps, FIGURES, trivial=runs[5]["trivial_ams"],
                              classical_ref=runs[5]["gboost_mean"])
    plotting.plot_circuit(best_config(features=top5), FIGURES)
    print("Done. See experiments/results/ and experiments/figures/.")


if __name__ == "__main__":
    main()
