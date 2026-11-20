"""Small reproducible logs, descriptive metrics, and plots for the synthetic run."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, fields
from importlib.metadata import version
from importlib.util import find_spec
import csv
import json
import os
from pathlib import Path
from platform import python_version
from statistics import mean, median
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from synthetic_environment import SPLITS, derive_seed

if TYPE_CHECKING:
    from experiment import Episode, ExperimentRun


STATUS = ("completed", "unreachable", "blocked", "timeout")
POLICY_NAMES = {
    "static": "Static A*", "traffic_aware": "Traffic-aware A*",
    "epsilon_greedy": "ε-greedy", "first_candidate": "First candidate",
}
POLICY_COLORS = {
    "static": "#305e9e", "traffic_aware": "#c65a22",
    "epsilon_greedy": "#078275", "first_candidate": "#7054a0",
}


def prepare_output_dir(path: Path, *, plots: bool = True) -> Path:
    directory = path.resolve()
    root = Path(__file__).resolve().parent
    if root not in directory.parents:
        raise ValueError("Output directory must be inside Routing Algorithms.")
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("Use a new or empty output directory.")
    if plots and find_spec("matplotlib") is None:
        raise ValueError("Plots require Matplotlib. Use --no-plots for CSV-only output.")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _mean(values):
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _percentile_90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = 0.9 * (len(ordered) - 1)
    lower = int(position)
    fraction = position - lower
    return (1 - fraction) * ordered[lower] + fraction * ordered[min(lower + 1, len(ordered) - 1)]


def summarize(episodes: tuple[Episode, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in episodes:
        for condition, pattern in (("all", "all"), (row.condition, "all"), (row.condition, row.pattern)):
            groups[row.split, row.method, row.epsilon, condition, pattern].append(row)
    summary = []
    for (split, method, epsilon, condition, pattern), rows in groups.items():
        statuses = Counter(row.status for row in rows)
        waits = [row.modeled_pickup_s for row in rows if row.status == "completed"]
        count = len(rows)
        summary.append({
            "experiment_id": rows[0].experiment_id, "label": rows[0].label,
            "split": split, "method": method, "epsilon": epsilon,
            "condition": condition, "pattern": pattern, "episodes": count,
            **{status: statuses[status] for status in STATUS},
            "completion_rate": statuses["completed"] / count,
            "failure_rate": (count - statuses["completed"]) / count,
            "timeout_rate": statuses["timeout"] / count,
            "mean_pickup_s": _mean(waits), "median_pickup_s": median(waits) if waits else None,
            "p90_pickup_s": _percentile_90(waits),
            "mean_estimated_travel_s": _mean(row.estimated_travel_s for row in rows),
            "mean_realized_route_s": _mean(row.realized_route_s for row in rows),
            "mean_routing_time_s": _mean(row.routing_time_s for row in rows),
            "mean_candidate_count": _mean(row.candidate_count for row in rows)
            if method in ("epsilon_greedy", "first_candidate") else None,
            "traffic_fallback_episode_rate": sum(row.fallback_road_count > 0 for row in rows) / count
            if method != "static" else None,
        })
    return summary


def route_counts(episodes: tuple[Episode, ...]) -> list[dict]:
    available, selected = Counter(), Counter()
    for row in episodes:
        group = (row.split, row.pair_seed, row.method, row.epsilon, row.context)
        for route in row.candidate_routes:
            available[(*group, route)] += 1
        if row.selected_route is not None:
            selected[(*group, row.selected_route)] += 1
    return [
        {
            "experiment_id": episodes[0].experiment_id, "label": episodes[0].label,
            "split": split, "pair_seed": seed, "method": method, "epsilon": epsilon,
            "context": context, "road_ids": route, "available_as_candidate": available[key],
            "selected": selected[key],
        }
        for key in available | selected
        for split, seed, method, epsilon, context, route in (key,)
    ]


def paired_test_rows(episodes: tuple[Episode, ...]) -> list[dict]:
    groups = defaultdict(dict)
    for row in episodes:
        if row.split == "test":
            if row.method in groups[row.scenario_id]:
                raise ValueError("Each policy must contribute one row per held-out scenario.")
            groups[row.scenario_id][row.method] = row
    comparisons = [
        ("traffic_aware", "static"), ("epsilon_greedy", "static"),
        ("epsilon_greedy", "traffic_aware"), ("epsilon_greedy", "first_candidate"),
    ]
    pairs = []
    for policies in groups.values():
        for method, reference in comparisons:
            if reference not in policies or method not in policies:
                continue
            left, right = policies[method], policies[reference]
            left_s, right_s = left.modeled_pickup_s, right.modeled_pickup_s
            pairs.append({
                "experiment_id": left.experiment_id, "label": left.label,
                "scenario_id": left.scenario_id, "pair_seed": left.pair_seed,
                "condition": left.condition, "pattern": left.pattern, "method": method,
                "reference": reference, "method_status": left.status, "reference_status": right.status,
                "method_pickup_s": left_s, "reference_pickup_s": right_s,
                "delta_pickup_s": left_s - right_s if left_s is not None and right_s is not None else None,
            })
    return pairs


def _write_csv(path: Path, columns, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in rows:
            writer.writerow({
                key: json.dumps(value, separators=(",", ":")) if isinstance(value, tuple) else value
                for key, value in record.items()
            })


def write_outputs(result: ExperimentRun, directory: Path, *, plots: bool = True) -> None:
    config = result.config
    rows = result.episodes
    summary = summarize(rows)
    pairs = paired_test_rows(rows)
    routes = route_counts(rows)
    manifest = {
        "label": rows[0].label, "experiment_id": config.experiment_id,
        "configuration": asdict(config), "chosen_epsilon": result.chosen_epsilon,
        "validation_criterion": ["highest completion rate", "lowest completed-only mean pickup seconds",
                                 "lowest epsilon"],
        "split_seeds": {seed: {split: derive_seed(seed, "scenario", split) for split in SPLITS}
                        for seed in config.seeds},
        "runtime": {"python": python_version(), "networkx": version("networkx"),
                    "pandas": version("pandas"), "matplotlib": version("matplotlib") if plots else None},
    }
    (directory / "config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_csv(directory / "episodes.csv", [item.name for item in fields(rows[0])], map(asdict, rows))
    _write_csv(directory / "summary.csv", list(summary[0]), summary)
    _write_csv(directory / "paired_test.csv", list(pairs[0]), pairs)
    route_columns = ["experiment_id", "label", "split", "pair_seed", "method", "epsilon", "context",
                     "road_ids", "available_as_candidate", "selected"]
    _write_csv(directory / "route_counts.csv", route_columns, routes)
    learned_columns = ["experiment_id", "label", "pair_seed", "epsilon", "context", "agent_seed",
                       "road_ids", "reward_samples", "mean_reward"]
    _write_csv(directory / "learned_routes.csv", learned_columns, map(asdict, result.learned_routes))
    if plots:
        plot_outputs(result, summary, pairs, directory)


def plot_outputs(result: ExperimentRun, summary: list[dict], pairs: list[dict], directory: Path) -> None:
    # The optional plotting dependency uses a disposable cache inside the output directory.
    with TemporaryDirectory(prefix=".plot-cache-", dir=directory) as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt

            _draw_plots(result, summary, pairs, directory / "preliminary_plots.png", plt)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def _draw_plots(result: ExperimentRun, summary: list[dict], pairs: list[dict], path: Path, plt) -> None:
    config = result.config
    fig, grid = plt.subplots(2, 3, figsize=(16.0, 9.4), layout="constrained")
    fig.suptitle("Preliminary synthetic robotaxi routing experiments",
                 fontsize=17, fontweight="bold")
    learning, levels, validation, distribution, completion, paired = grid.flat
    train_rows = [row for row in result.episodes if row.split == "train"]
    test_rows = [row for row in result.episodes if row.split == "test"]
    methods = list(dict.fromkeys(row.method for row in test_rows))
    block_size = len(config.levels) * 4 + 1
    for epsilon in sorted(config.epsilons):
        by_block = defaultdict(list)
        for row in train_rows:
            if row.epsilon == epsilon:
                by_block[(row.episode - 1) // block_size].append(row.modeled_pickup_s)
        blocks = sorted(by_block)
        learning.plot([min((block + 1) * block_size, config.train_episodes) for block in blocks],
                      [_mean(by_block[block]) for block in blocks], marker="o", ms=3,
                      label=f"ε={epsilon:g}")
    learning.set(title="Training learning curves", xlabel="Training episodes per seed",
                 ylabel="Mean modeled pickup time (s)\nCompleted pickups")
    learning.legend(ncols=2, fontsize=8)

    for method in methods:
        color, name = POLICY_COLORS[method], POLICY_NAMES[method]
        points = {row["condition"]: row["mean_pickup_s"] for row in summary
                  if row["split"] == "test" and row["method"] == method and row["pattern"] == "all"}
        levels.plot(range(len(config.levels)), [points.get(level) for level in config.levels],
                    color=color, marker="o", label=name)
        values = sorted(row.modeled_pickup_s for row in test_rows
                        if row.method == method and row.modeled_pickup_s is not None)
        if values:
            distribution.step([values[0], *values],
                              [i / len(values) for i in range(len(values) + 1)],
                              where="post", color=color, label=name)
    levels.set_xticks(range(len(config.levels)), [level.replace("_", " ").title() for level in config.levels])
    levels.set(title="Held-out pickup time by congestion", xlabel="Nominal short-road congestion",
               ylabel="Mean modeled pickup time (s)\nCompleted pickups")
    levels.legend(fontsize=8)
    distribution.set(title="Held-out completed pickup distribution", xlabel="Modeled pickup time (s)",
                     ylabel="Cumulative proportion", ylim=(0, 1.04))
    if distribution.lines:
        distribution.legend(fontsize=8)

    val = sorted(
        (row for row in summary if row["split"] == "validation" and row["method"] == "epsilon_greedy"
         and row["condition"] == "all"), key=lambda row: row["epsilon"],
    )
    validation.plot([row["epsilon"] for row in val], [row["mean_pickup_s"] for row in val],
                    color="#078275", marker="o", label="Completed mean (s)")
    validation.axvline(result.chosen_epsilon, color="#555555", linestyle="--", label="Selected ε")
    val_completion = validation.twinx()
    val_completion.plot([row["epsilon"] for row in val], [100 * row["completion_rate"] for row in val],
                        color="#c65a22", marker="s", linestyle=":", label="Completion (%)")
    validation.set(title="Frozen validation by training ε", xlabel="Training exploration rate ε",
                   ylabel="Mean modeled pickup time (s)")
    val_completion.set(ylabel="Completion (%)", ylim=(0, 105))
    validation.legend(handles=[*validation.lines, *val_completion.lines], fontsize=8, loc="best")

    width = [0.0] * len(methods)
    for status, color in zip(STATUS, ("#078275", "#a7acb2", "#c65a22", "#805199"), strict=True):
        values = [100 * sum(row.method == method and row.status == status for row in test_rows)
                  / sum(row.method == method for row in test_rows) for method in methods]
        completion.barh([POLICY_NAMES[method] for method in methods], values, left=width,
                        label=status.title(), color=color)
        width = [previous + value for previous, value in zip(width, values, strict=True)]
    completion.set(title="Held-out completion and failure", xlabel="Share of all requests (%)", xlim=(0, 100))
    completion.invert_yaxis()
    completion.legend(fontsize=8, ncols=2, loc="best")

    comparison_rows = defaultdict(list)
    for row in pairs:
        if row["delta_pickup_s"] is not None:
            comparison_rows[row["method"], row["reference"]].append(row["delta_pickup_s"])
    if comparison_rows:
        paired.boxplot(list(comparison_rows.values()), orientation="horizontal", showfliers=False,
                       tick_labels=[f"{POLICY_NAMES[left]} − {POLICY_NAMES[right]}"
                                    for left, right in comparison_rows])
        paired.invert_yaxis()
    else:
        paired.text(0.5, 0.5, "No pairs with two completed pickups", ha="center", transform=paired.transAxes)
    paired.axvline(0, linewidth=1, linestyle="--", color="#666666")
    paired.set(title="Held-out paired pickup differences", xlabel="Modeled pickup difference (s)\nBoth policies completed")
    paired.tick_params(axis="y", labelsize=8)
    for axis in (learning, levels, validation, distribution, completion, paired):
        axis.grid(axis="x" if axis in (completion, paired) else "y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.savefig(path, dpi=155)
    plt.close(fig)
