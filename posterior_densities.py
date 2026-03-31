from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from integration import posterior_density


SCORE_FIELDS = [
    "continuous_release",
    "immediate_ignition",
    "barrier_stopped_immediate_ignition",
    "delayed_ignition",
    "barrier_stopped_delayed_ignition",
    "confined_space",
    "pure_h2",
    "gaseous_h2",
    "pressurized_h2",
    "loss_of_containment",
]

INCLUSION_FIELDS = [
    "pure_h2",
    "gaseous_h2",
    "pressurized_h2",
    "loss_of_containment",
]

QUESTION_LABELS = {
    "continuous_release": "Continuous release",
    "immediate_ignition": "Immediate ignition",
    "delayed_ignition": "Delayed ignition",
    "confined_space": "Confined space",
}

VALUE_LABELS = {True: "Yes", False: "No"}


WORKER_STATE: dict[str, Any] = {}


@dataclass(frozen=True)
class BranchSpec:
    slug: str
    question: str
    parent_conditions: dict[str, bool]
    parent_label: str
    yes_node_id: str
    no_node_id: str


BRANCH_SPECS = [
    BranchSpec(
        slug="01_continuous_release",
        question="continuous_release",
        parent_conditions={},
        parent_label="Included pool",
        yes_node_id="cr_yes",
        no_node_id="cr_no",
    ),
    BranchSpec(
        slug="02_immediate_ignition_given_continuous_yes",
        question="immediate_ignition",
        parent_conditions={"continuous_release": True},
        parent_label="Continuous release = Yes",
        yes_node_id="imm_yes_cr_yes",
        no_node_id="imm_no_cr_yes",
    ),
    BranchSpec(
        slug="03_immediate_ignition_given_continuous_no",
        question="immediate_ignition",
        parent_conditions={"continuous_release": False},
        parent_label="Continuous release = No",
        yes_node_id="imm_yes_cr_no",
        no_node_id="imm_no_cr_no",
    ),
    BranchSpec(
        slug="04_delayed_ignition_given_continuous_yes_immediate_no",
        question="delayed_ignition",
        parent_conditions={"continuous_release": True, "immediate_ignition": False},
        parent_label="Continuous release = Yes, Immediate ignition = No",
        yes_node_id="del_yes_cr_yes",
        no_node_id="del_no_cr_yes",
    ),
    BranchSpec(
        slug="05_delayed_ignition_given_continuous_no_immediate_no",
        question="delayed_ignition",
        parent_conditions={"continuous_release": False, "immediate_ignition": False},
        parent_label="Continuous release = No, Immediate ignition = No",
        yes_node_id="del_yes_cr_no",
        no_node_id="del_no_cr_no",
    ),
    BranchSpec(
        slug="06_confined_space_given_continuous_yes_immediate_no_delayed_yes",
        question="confined_space",
        parent_conditions={
            "continuous_release": True,
            "immediate_ignition": False,
            "delayed_ignition": True,
        },
        parent_label="Continuous release = Yes, Immediate ignition = No, Delayed ignition = Yes",
        yes_node_id="conf_yes_cr_yes_del_yes",
        no_node_id="conf_no_cr_yes_del_yes",
    ),
    BranchSpec(
        slug="07_confined_space_given_continuous_no_immediate_no_delayed_yes",
        question="confined_space",
        parent_conditions={
            "continuous_release": False,
            "immediate_ignition": False,
            "delayed_ignition": True,
        },
        parent_label="Continuous release = No, Immediate ignition = No, Delayed ignition = Yes",
        yes_node_id="conf_yes_cr_no_del_yes",
        no_node_id="conf_no_cr_no_del_yes",
    ),
]

FILTER_OPTION_NAMES = [
    "onlyPureH2",
    "onlyGaseousH2",
    "onlyPressurizedH2",
    "onlyLossOfContainment",
    "includeBarrierImmediate",
    "includeBarrierDelayed",
    "useScoreWeights",
]

DEFAULT_FILTER_OPTIONS = {
    "onlyPureH2": True,
    "onlyGaseousH2": True,
    "onlyPressurizedH2": True,
    "onlyLossOfContainment": True,
    "includeBarrierImmediate": False,
    "includeBarrierDelayed": False,
    "useScoreWeights": True,
}


def ensure_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def normalize_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return min(10.0, max(0.0, score))


def score_to_answer_weight(value: Any, use_score_strength: bool = True) -> tuple[float, float]:
    score = normalize_score(value)
    if score is None:
        return 1.0, 0.0
    answer = 1.0 if score >= 5.0 else 0.0
    if score == 5.0:
        return answer, 0.0
    weight = abs(score - 5.0) / 5.0 if use_score_strength else 1.0
    return answer, weight


def average_score(record: dict[str, Any], key: str) -> float | None:
    values = [normalize_score(value) for value in ensure_array(record.get(key))]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def slugify_value(value: bool) -> str:
    return "yes" if value else "no"


def round_float(value: float, digits: int = 8) -> float:
    return round(float(value), digits)


def round_float_list(values: np.ndarray | list[float], digits: int = 8) -> list[float]:
    return [round_float(value, digits=digits) for value in values]


def model_option_name(model_id: str) -> str:
    return f"model:{model_id}"


def option_key(options: dict[str, bool], model_ids: list[str]) -> str:
    parts = [f"{name}:{1 if options[name] else 0}" for name in FILTER_OPTION_NAMES]
    parts.extend(f"{model_option_name(model_id)}:{1 if options[model_option_name(model_id)] else 0}" for model_id in model_ids)
    return "__".join(parts)


def option_names(model_ids: list[str]) -> list[str]:
    return FILTER_OPTION_NAMES + [model_option_name(model_id) for model_id in model_ids]


def option_combo_index(options: dict[str, bool], names: list[str]) -> int:
    value = 0
    for bit_index, name in enumerate(names):
        if options.get(name, False):
            value |= 1 << bit_index
    return value


def default_options(model_ids: list[str]) -> dict[str, bool]:
    options = dict(DEFAULT_FILTER_OPTIONS)
    for model_id in model_ids:
        options[model_option_name(model_id)] = True
    return options


def selected_model_ids_from_options(options: dict[str, bool], model_ids: list[str]) -> list[str]:
    return [model_id for model_id in model_ids if options.get(model_option_name(model_id), False)]


def iter_filter_options(model_ids: list[str]) -> list[dict[str, bool]]:
    combinations: list[dict[str, bool]] = []
    names = option_names(model_ids)
    for bits in range(1 << len(names)):
        options = {}
        for index, name in enumerate(names):
            options[name] = bool((bits >> index) & 1)
        combinations.append(options)
    return combinations


def format_probability(value: float) -> str:
    return f"{value * 100:.1f}%"


def shortest_credible_interval(grid: np.ndarray, pdf: np.ndarray, mass: float = 0.95) -> tuple[float, float]:
    if grid.size < 2:
        return float(grid[0]), float(grid[-1])
    dx = np.diff(grid)
    midpoint_mass = 0.5 * (pdf[:-1] + pdf[1:]) * dx
    cdf = np.concatenate([[0.0], np.cumsum(midpoint_mass)])
    cdf /= cdf[-1]

    best_width = math.inf
    best_pair = (float(grid[0]), float(grid[-1]))
    upper_indices = np.searchsorted(cdf, cdf + mass, side="left")
    for left_index, right_index in enumerate(upper_indices):
        if right_index >= grid.size:
            break
        left = float(grid[left_index])
        right = float(grid[right_index])
        width = right - left
        if width < best_width:
            best_width = width
            best_pair = (left, right)
    return best_pair


def summarize_density(grid: np.ndarray, density: np.ndarray) -> dict[str, float]:
    mean = float(np.trapezoid(grid * density, grid))
    mode = float(grid[int(np.argmax(density))])
    summary: dict[str, float] = {
        "mean": mean,
        "mode": mode,
    }
    for mass in (0.50, 0.68, 0.95, 0.99):
        low, high = shortest_credible_interval(grid, density, mass=mass)
        label = int(round(mass * 100))
        summary[f"hdi_{label}_low"] = low
        summary[f"hdi_{label}_high"] = high
    return summary


def plot_density(
    grid: np.ndarray,
    density: np.ndarray,
    title: str,
    output_path: Path,
    stats: dict[str, float],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.plot(grid, density, color="#1f4e79", linewidth=2.0)

    hdi_low = stats["hdi_95_low"]
    hdi_high = stats["hdi_95_high"]
    mask = (grid >= hdi_low) & (grid <= hdi_high)
    ax.fill_between(grid, density, where=mask, color="#8ecae6", alpha=0.45)
    ax.axvline(stats["mean"], color="#cc5500", linestyle="--", linewidth=1.2)
    ax.axvline(stats["mode"], color="#2a9d8f", linestyle=":", linewidth=1.2)

    annotation = (
        f"Mean {format_probability(stats['mean'])} | "
        f"Mode {format_probability(stats['mode'])} | "
        f"HDI95 [{format_probability(hdi_low)}, {format_probability(hdi_high)}]"
    )
    ax.text(
        0.02,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#c8d3df", "alpha": 0.95},
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Posterior density")
    ax.set_title(title)
    ax.grid(alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def hydrate_record(base_record: dict[str, Any], shared_record: dict[str, Any] | None, model: dict[str, Any]) -> dict[str, Any]:
    record = {}
    if shared_record:
        record.update(shared_record)
    record.update(base_record)
    record["model"] = model.get("model") or model.get("name") or ""
    record["model_name"] = model.get("name") or model.get("model") or ""
    for key in SCORE_FIELDS:
        record[key] = ensure_array(record.get(key))
    return record


def load_aggregated_events(root: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], int, int]:
    manifest = load_json(manifest_path)
    models = manifest.get("models", [])
    shared_events = manifest.get("events", [])
    answers_per_model = int(manifest.get("answers_per_model", 0))
    if not models:
        raise ValueError("No models found in the manifest.")

    shared_by_event_id = {}
    for index, shared_event in enumerate(shared_events):
        event_id = shared_event.get("event_id")
        if event_id is not None:
            shared_by_event_id[str(event_id)] = shared_event
        else:
            shared_by_event_id[f"index:{index}"] = shared_event

    aggregated: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for model in models:
        events_path = root / model["events_path"]
        records = load_json(events_path)
        for index, record in enumerate(records):
            event_id = record.get("event_id")
            key = str(event_id) if event_id is not None else f"index:{index}"
            shared_record = shared_by_event_id.get(key)
            hydrated = hydrate_record(record, shared_record, model)
            if key not in aggregated:
                aggregated[key] = {
                    "event_id": hydrated.get("event_id"),
                    "title": hydrated.get("title") or "",
                    "description": hydrated.get("description") or "",
                    "records_by_model": {},
                    "model_order": [],
                }
                order.append(key)
            aggregated[key]["records_by_model"][hydrated["model"]] = hydrated
            aggregated[key]["model_order"].append(hydrated["model"])

    if not order:
        raise ValueError("No events loaded from model outputs.")

    model_ids = [model.get("model") or model.get("name") or "" for model in models]
    events = []
    for key in order:
        event = aggregated[key]
        missing = [model_id for model_id in model_ids if model_id not in event["records_by_model"]]
        if missing:
            continue
        event["model_ids"] = model_ids
        events.append(event)

    if not events:
        raise ValueError("No events contain a full set of model records.")

    first_record = events[0]["records_by_model"][model_ids[0]]
    inferred_answers = max(len(first_record.get(field, [])) for field in SCORE_FIELDS)
    return events, len(model_ids), answers_per_model or inferred_answers


def event_average_score(event: dict[str, Any], key: str, selected_model_ids: list[str]) -> float | None:
    values: list[float] = []
    for model_id in selected_model_ids:
        record = event["records_by_model"][model_id]
        for value in ensure_array(record.get(key)):
            normalized = normalize_score(value)
            if normalized is not None:
                values.append(normalized)
    if not values:
        return None
    return float(np.mean(values))


def event_majority(event: dict[str, Any], key: str, selected_model_ids: list[str]) -> bool:
    avg = event_average_score(event, key, selected_model_ids)
    return bool(avg is not None and avg > 5.0)


def barrier_immediate_majority(event: dict[str, Any], selected_model_ids: list[str]) -> bool:
    return event_majority(event, "barrier_stopped_immediate_ignition", selected_model_ids)


def barrier_delayed_majority(event: dict[str, Any], selected_model_ids: list[str]) -> bool:
    return event_majority(event, "barrier_stopped_delayed_ignition", selected_model_ids) and not event_majority(
        event, "immediate_ignition", selected_model_ids
    )


def is_included_event(event: dict[str, Any], options: dict[str, bool], selected_model_ids: list[str]) -> bool:
    if not selected_model_ids:
        return False
    if options["onlyPureH2"] and not event_majority(event, "pure_h2", selected_model_ids):
        return False
    if options["onlyGaseousH2"] and not event_majority(event, "gaseous_h2", selected_model_ids):
        return False
    if options["onlyPressurizedH2"] and not event_majority(event, "pressurized_h2", selected_model_ids):
        return False
    if options["onlyLossOfContainment"] and not event_majority(event, "loss_of_containment", selected_model_ids):
        return False
    if not options["includeBarrierImmediate"] and barrier_immediate_majority(event, selected_model_ids):
        return False
    if not options["includeBarrierDelayed"] and barrier_delayed_majority(event, selected_model_ids):
        return False
    return True


def matches_parent_conditions(event: dict[str, Any], conditions: dict[str, bool], selected_model_ids: list[str]) -> bool:
    return all(event_majority(event, key, selected_model_ids) == expected for key, expected in conditions.items())


def build_weight_matrices(
    events: list[dict[str, Any]],
    question: str,
    selected_model_ids: list[str],
    use_score_strength: bool,
) -> tuple[np.ndarray, np.ndarray]:
    s_matrix = np.zeros((len(events), len(selected_model_ids)), dtype=np.float64)
    n_matrix = np.zeros((len(events), len(selected_model_ids)), dtype=np.float64)

    for event_index, event in enumerate(events):
        for model_index, model_id in enumerate(selected_model_ids):
            record = event["records_by_model"][model_id]
            weights = []
            weighted_answers = []
            for score in ensure_array(record.get(question)):
                answer, weight = score_to_answer_weight(score, use_score_strength=use_score_strength)
                weights.append(weight)
                weighted_answers.append(answer * weight)
            s_matrix[event_index, model_index] = float(np.sum(weighted_answers))
            n_matrix[event_index, model_index] = float(np.sum(weights))
    return s_matrix, n_matrix


def mirror_stats(stats: dict[str, float]) -> dict[str, float]:
    mirrored = {
        "mean": 1.0 - stats["mean"],
        "mode": 1.0 - stats["mode"],
        "hdi_95_low": 1.0 - stats["hdi_95_high"],
        "hdi_95_high": 1.0 - stats["hdi_95_low"],
    }
    for label in (50, 68, 95, 99):
        mirrored[f"hdi_{label}_low"] = 1.0 - stats[f"hdi_{label}_high"]
        mirrored[f"hdi_{label}_high"] = 1.0 - stats[f"hdi_{label}_low"]
    return mirrored


def branch_title(
    question: str,
    branch_value: bool,
    parent_label: str,
    parent_event_count: int,
    branch_event_count: int,
    model_count: int,
    answer_count: int,
) -> str:
    return (
        f"{QUESTION_LABELS[question]} = {VALUE_LABELS[branch_value]}\n"
        f"Parent: {parent_label} | Parent events: {parent_event_count} | Branch count: {branch_event_count} | "
        f"Models: {model_count} | Realizations/model: {answer_count}"
    )


def serialize_node_summary(branch_event_count: int, stats: dict[str, float]) -> list[float]:
    return [int(branch_event_count), round_float(stats["mean"], digits=6)]


def init_worker_state(
    events: list[dict[str, Any]],
    answer_count: int,
    p_grid: np.ndarray,
    sample_count: int,
    seed: int,
    chunk_size: int,
    generate_plots: bool,
    plots_dir: str,
    densities_dir: str,
) -> None:
    global WORKER_STATE
    WORKER_STATE = {
        "events": events,
        "answer_count": answer_count,
        "p_grid": p_grid,
        "sample_count": sample_count,
        "seed": seed,
        "chunk_size": chunk_size,
        "generate_plots": generate_plots,
        "plots_dir": Path(plots_dir),
        "densities_dir": Path(densities_dir),
    }


def compute_combo_entry(combo_index: int, options: dict[str, bool], model_ids: list[str]) -> tuple[int, list[Any]]:
    events: list[dict[str, Any]] = WORKER_STATE["events"]
    answer_count: int = WORKER_STATE["answer_count"]
    p_grid: np.ndarray = WORKER_STATE["p_grid"]
    sample_count: int = WORKER_STATE["sample_count"]
    seed: int = WORKER_STATE["seed"]
    chunk_size: int = WORKER_STATE["chunk_size"]
    generate_plots: bool = WORKER_STATE["generate_plots"]
    plots_dir: Path = WORKER_STATE["plots_dir"]
    densities_dir: Path = WORKER_STATE["densities_dir"]

    selected_model_ids = selected_model_ids_from_options(options, model_ids)
    density_file = densities_dir / f"combo_{combo_index:05d}.json"
    included_events = [event for event in events if is_included_event(event, options, selected_model_ids)]
    node_order = [spec.yes_node_id for spec in BRANCH_SPECS] + [spec.no_node_id for spec in BRANCH_SPECS]
    node_index_by_id = {node_id: index for index, node_id in enumerate(node_order)}
    combo_entry: list[Any] = [len(included_events), None]
    if not included_events:
        return combo_index, combo_entry

    density_payload: list[list[float]] = []
    combo_nodes: list[list[float] | None] = [None] * len(node_order)
    for branch_index, spec in enumerate(BRANCH_SPECS, start=1):
        parent_events = [
            event
            for event in included_events
            if matches_parent_conditions(event, spec.parent_conditions, selected_model_ids)
        ]
        if not parent_events:
            continue

        s_matrix, n_matrix = build_weight_matrices(
            parent_events,
            spec.question,
            selected_model_ids,
            use_score_strength=options["useScoreWeights"],
        )
        density_yes = posterior_density(
            s_matrix=s_matrix,
            n_matrix=n_matrix,
            p_grid=p_grid,
            sample_count=sample_count,
            seed=seed + combo_index * 100 + branch_index,
            chunk_size=chunk_size,
        )
        stats_yes = summarize_density(p_grid, density_yes)
        density_no = density_yes[::-1].copy()
        stats_no = mirror_stats(stats_yes)
        density_payload.append(round_float_list(density_yes))

        outputs = [
            (True, spec.yes_node_id, density_yes, stats_yes),
            (False, spec.no_node_id, density_no, stats_no),
        ]
        for branch_value, node_id, density, stats in outputs:
            branch_event_count = sum(
                1
                for event in parent_events
                if event_majority(event, spec.question, selected_model_ids) == branch_value
            )
            output_path = None
            if generate_plots:
                combo_plot_dir = plots_dir / combo_key
                combo_plot_dir.mkdir(parents=True, exist_ok=True)
                output_path = combo_plot_dir / f"{node_id}.png"
                plot_density(
                    grid=p_grid,
                    density=density,
                    title=branch_title(
                        question=spec.question,
                        branch_value=branch_value,
                        parent_label=spec.parent_label,
                        parent_event_count=len(parent_events),
                        branch_event_count=branch_event_count,
                        model_count=len(selected_model_ids),
                        answer_count=answer_count,
                    ),
                    output_path=output_path,
                    stats=stats,
                )

            combo_nodes[node_index_by_id[node_id]] = serialize_node_summary(branch_event_count=branch_event_count, stats=stats)

    with density_file.open("w", encoding="utf-8") as handle:
        json.dump(density_payload, handle, separators=(",", ":"))
        handle.write("\n")
    combo_entry[1] = combo_nodes

    return combo_index, combo_entry


def write_summary_json(summary: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, separators=(",", ":"))
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute posterior plots for every event-tree branch.")
    parser.add_argument("--manifest", default="events-manifest.json", help="Path to the events manifest.")
    parser.add_argument("--output-dir", default="posteriors", help="Directory to write posterior plots into.")
    parser.add_argument("--samples", type=int, default=1024, help="Number of latent-space integration samples.")
    parser.add_argument("--grid-size", type=int, default=401, help="Number of probability grid points.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="Number of p-grid points to evaluate together.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes to use for combination-level parallelism.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for QMC/MC integration.")
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Also save posterior PNGs for every precomputed option combination.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    manifest_path = (root / args.manifest).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    densities_dir = output_dir / "densities"
    densities_dir.mkdir(parents=True, exist_ok=True)
    if args.generate_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    events, model_count, answer_count = load_aggregated_events(root=root, manifest_path=manifest_path)
    if not events:
        raise ValueError("No events were available for posterior generation.")

    model_ids = events[0]["model_ids"]
    model_entries = [
        {
            "id": model_id,
            "name": events[0]["records_by_model"][model_id].get("model_name") or model_id,
        }
        for model_id in model_ids
    ]
    default_option_values = default_options(model_ids)
    all_option_names = option_names(model_ids)
    node_order = [spec.yes_node_id for spec in BRANCH_SPECS] + [spec.no_node_id for spec in BRANCH_SPECS]
    yes_node_order = [spec.yes_node_id for spec in BRANCH_SPECS]
    node_density_map = []
    yes_index_by_id = {node_id: index for index, node_id in enumerate(yes_node_order)}
    for spec in BRANCH_SPECS:
        node_density_map.append([yes_index_by_id[spec.yes_node_id], 0])
    for spec in BRANCH_SPECS:
        node_density_map.append([yes_index_by_id[spec.yes_node_id], 1])
    p_grid = np.linspace(1e-4, 1.0 - 1e-4, args.grid_size)
    summary_payload: dict[str, Any] = {
        "meta": {
            "sample_count": args.samples,
            "grid_size": args.grid_size,
            "chunk_size": max(1, args.chunk_size),
            "workers": max(1, args.workers),
            "model_count": model_count,
            "realizations_per_model": answer_count,
            "grid": round_float_list(p_grid),
            "option_names": all_option_names,
            "available_models": model_entries,
            "default_combo_index": option_combo_index(default_option_values, all_option_names),
            "node_order": node_order,
            "yes_node_order": yes_node_order,
            "node_density_map": node_density_map,
            "generate_plots": bool(args.generate_plots),
        },
        "combinations": [],
    }

    filter_options_list = iter_filter_options(model_ids)
    combo_results: dict[int, list[Any]] = {}
    workers = max(1, args.workers)
    if workers == 1:
        combo_progress = tqdm(range(len(filter_options_list)), desc="Computing posteriors")
        init_worker_state(
            events=events,
            answer_count=answer_count,
            p_grid=p_grid,
            sample_count=args.samples,
            seed=args.seed,
            chunk_size=max(1, args.chunk_size),
            generate_plots=bool(args.generate_plots),
            plots_dir=str(plots_dir),
            densities_dir=str(densities_dir),
        )
        for combo_index in combo_progress:
            returned_index, combo_entry = compute_combo_entry(
                combo_index=combo_index,
                options=filter_options_list[combo_index],
                model_ids=model_ids,
            )
            combo_results[returned_index] = combo_entry
    else:
        ctx = multiprocessing.get_context("spawn")
        combo_progress = tqdm(total=len(filter_options_list), desc="Computing posteriors")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=init_worker_state,
            initargs=(
                events,
                answer_count,
                p_grid,
                args.samples,
                args.seed,
                max(1, args.chunk_size),
                bool(args.generate_plots),
                str(plots_dir),
                str(densities_dir),
            ),
        ) as executor:
            future_map = {
                executor.submit(compute_combo_entry, combo_index, options, model_ids): combo_index
                for combo_index, options in enumerate(filter_options_list)
            }
            for future in concurrent.futures.as_completed(future_map):
                returned_index, combo_entry = future.result()
                combo_results[returned_index] = combo_entry
                combo_progress.update(1)
        combo_progress.close()

    summary_payload["combinations"] = [combo_results[index] for index in range(len(filter_options_list))]

    write_summary_json(summary_payload, output_dir / "posterior_summary.json")


if __name__ == "__main__":
    main()
