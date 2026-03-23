import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.optimize import minimize_scalar
import numpy as np

def create_event_tree(events, show_exclusion = False, filename = "event_tree.png"):
    """
    Build a left-to-right event tree using the boolean answers.

    Layout rules:
    - LOC at the root (or "All events" when showing exclusions).
    - Optional inclusion column before branching if show_exclusion=True.
    - Split on continuous release (Yes/No).
    - Split each branch on immediate ignition.
      - If immediate ignition is Yes, go directly to an outcome.
      - If immediate ignition is No, split on delayed ignition.
    - If delayed ignition is Yes, split on confined space.
    - If delayed ignition is No, go directly to an outcome.
    """
    if not events:
        raise ValueError("No events provided for plotting.")

    exclusion_flags = [
        "exclude_not_pure_h2",
        "exclude_not_gaseous_h2",
        "exclude_no_loc",
        "barrier_stopped_immediate_ignition",
    ]

    def barrier_delayed(event):
        return bool(event.get("barrier_stopped_delayed_ignition")) and not bool(
            event.get("immediate_ignition")
        )
    if show_exclusion:
        events = [
            {
                **event,
                "excluded": any(bool(event.get(flag)) for flag in exclusion_flags)
                or barrier_delayed(event),
            }
            for event in events
        ]
    else:
        events = [
            event
            for event in events
            if all(not bool(event.get(flag)) for flag in exclusion_flags)
            and not barrier_delayed(event)
        ]
        if not events:
            raise ValueError("No includable events after exclusions were applied.")

    def matches(event, conditions):
        return all(bool(event.get(key)) == value for key, value in conditions.items())

    def count_for(conditions):
        return sum(1 for event in events if matches(event, conditions))

    def beta_hdr(alpha, beta_param, gamma=0.95):
        dist = beta(alpha, beta_param)

        # Boundary-mode shortcuts (often the HDR is one-sided)
        if alpha <= 1 and beta_param > 1:
            return 0.0, dist.ppf(gamma)
        if beta_param <= 1 and alpha > 1:
            return dist.ppf(1 - gamma), 1.0
        if alpha <= 1 and beta_param <= 1:
            # U-shaped; HDR is NOT a single interval in general.
            # You need a "highest-density set" that may be two-sided.
            raise ValueError("Beta is U-shaped (alpha<=1 and beta<=1): HDR may be disjoint.")

        # Unimodal case: shortest credible interval = HDR
        def interval_len(a):
            Fa = dist.cdf(a)
            Fb = Fa + gamma
            if Fb >= 1.0:
                return np.inf
            b = dist.ppf(Fb)
            return b - a

        # a must satisfy F(a) <= 1-gamma; a is in [0, ppf(1-gamma)]
        a_max = dist.ppf(1 - gamma)
        res = minimize_scalar(interval_len, bounds=(0.0, a_max), method="bounded")

        a = res.x
        b = dist.ppf(dist.cdf(a) + gamma)
        return a, b

    def format_uncertainty(x, w, plot=False):
        weights = [max(0.0, float(value)) / 10.0 for value in w]
        S = sum(xi * wi for xi, wi in zip(x, weights))
        N = sum(weights)

        if N <= 0:
            return "0.0 - 100.0%"

        alpha = S+1
        beta_param = N-S+1
        a,b = beta_hdr(alpha, beta_param, gamma=0.95)

        interval_label = f"{a*100:.1f} - {b*100:.1f}%"

        if plot:
            dist = beta(alpha, beta_param)
            xs = np.linspace(0.0, 1.0, 400)
            ys = dist.pdf(xs)

            fig, ax = plt.subplots()
            ax.fill_between(xs, ys, color="#c7d7ef", alpha=0.5)
            ax.fill_between(xs, ys, where=(xs >= a) & (xs <= b), color="#ffb878", alpha=0.7, label="95% credible interval")
            ax.plot(xs, ys, color="#1f77b4", lw=2.0, label=f"Beta({alpha:.1f}, {beta_param:.1f})")
            ax.axvline(a, color="#ff7f0e", linestyle="--", lw=1.2)
            ax.axvline(b, color="#ff7f0e", linestyle="--", lw=1.2)
            ax.set_xlabel("Probability")
            ax.set_ylabel("Density")
            ax.set_title("Beta posterior")
            ax.legend()
            fig.tight_layout()
            plt.show()

        return interval_label

    def bold_label(label):
        # Use mathtext bold and preserve spaces.
        escaped = str(label).replace(" ", r"\ ")
        return f"$\\bf{{{escaped}}}$"

    # Ordered leaves establish vertical spacing.
    main_leaf_specs = [
        {
            "id": "leaf_cr_y_imm_y",
            "conditions": {"continuous_release": True, "immediate_ignition": True},
            "parent": "imm_yes_cr_yes",
        },
        {
            "id": "leaf_cr_y_imm_n_del_y_conf_y",
            "conditions": {
                "continuous_release": True,
                "immediate_ignition": False,
                "delayed_ignition": True,
                "confined_space": True,
            },
            "parent": "conf_yes_cr_yes_del_yes",
        },
        {
            "id": "leaf_cr_y_imm_n_del_y_conf_n",
            "conditions": {
                "continuous_release": True,
                "immediate_ignition": False,
                "delayed_ignition": True,
                "confined_space": False,
            },
            "parent": "conf_no_cr_yes_del_yes",
        },
        {
            "id": "leaf_cr_y_imm_n_del_n",
            "conditions": {
                "continuous_release": True,
                "immediate_ignition": False,
                "delayed_ignition": False,
            },
            "parent": "del_no_cr_yes",
        },
        {
            "id": "leaf_cr_n_imm_y",
            "conditions": {"continuous_release": False, "immediate_ignition": True},
            "parent": "imm_yes_cr_no",
        },
        {
            "id": "leaf_cr_n_imm_n_del_y_conf_y",
            "conditions": {
                "continuous_release": False,
                "immediate_ignition": False,
                "delayed_ignition": True,
                "confined_space": True,
            },
            "parent": "conf_yes_cr_no_del_yes",
        },
        {
            "id": "leaf_cr_n_imm_n_del_y_conf_n",
            "conditions": {
                "continuous_release": False,
                "immediate_ignition": False,
                "delayed_ignition": True,
                "confined_space": False,
            },
            "parent": "conf_no_cr_no_del_yes",
        },
        {
            "id": "leaf_cr_n_imm_n_del_n",
            "conditions": {
                "continuous_release": False,
                "immediate_ignition": False,
                "delayed_ignition": False,
            },
            "parent": "del_no_cr_no",
        },
    ]

    if show_exclusion:
        exclusion_leaf_specs = [
            {
                "id": "leaf_excluded",
                "conditions": {"excluded": True},
                "parent": "root",
            },
        ]
        for leaf in main_leaf_specs:
            leaf["conditions"]["excluded"] = False
    else:
        exclusion_leaf_specs = []

    spacing = 2.0
    if show_exclusion:
        horizontal_positions = {
            "root": 0.0,
            "exclude": 2.0,
            "continuous": 4.0,
            "immediate": 6.0,
            "delayed": 8.0,
            "confined": 10.0,
            "outcome": 12.0,
        }
    else:
        horizontal_positions = {
            "root": 0.0,
            "continuous": 2.0,
            "immediate": 4.0,
            "delayed": 6.0,
            "confined": 8.0,
            "outcome": 10.0,
        }

    start_y = (len(main_leaf_specs) - 1) * spacing / 2
    for index, leaf in enumerate(main_leaf_specs):
        leaf["x"] = horizontal_positions["outcome"]
        leaf["y"] = start_y - index * spacing
        leaf["label"] = "XXX"
        leaf["kind"] = "outcome"

    if show_exclusion and exclusion_leaf_specs:
        # Place the exclusion outcomes below the main tree with a small gap.
        main_min_y = min(leaf["y"] for leaf in main_leaf_specs)
        exclusion_start_y = main_min_y - spacing * 2
        for index, leaf in enumerate(exclusion_leaf_specs):
            leaf["x"] = horizontal_positions["exclude"]
            leaf["y"] = exclusion_start_y - index * spacing
            leaf["label"] = "Excluded"
            leaf["kind"] = "outcome"

    outcome_labels = [
        "Jet fire",
        "VCE",
        "Flash fire",
        "Plume",
        "Fireball",
        "VCE",
        "Flash fire",
        "Puff",
    ]
    outcome_idx = 0
    for leaf in main_leaf_specs:
        if leaf["label"] != "XXX":
            continue
        label = outcome_labels[outcome_idx] if outcome_idx < len(outcome_labels) else outcome_labels[-1]
        leaf["label"] = label
        outcome_idx += 1
    leaf_specs = main_leaf_specs + exclusion_leaf_specs

    # Decision nodes.
    root_label = "All events" if show_exclusion else "LOC"
    decision_nodes = [
        {"id": "root", "label": root_label, "conditions": {}, "parent": None, "x": horizontal_positions["root"]},
    ]

    if show_exclusion:
        decision_nodes.append(
            {
                "id": "loc",
                "label": "LOC",
                "conditions": {"excluded": False},
                "parent": "root",
                "x": horizontal_positions["exclude"],
            }
        )

    decision_nodes.extend(
        [
            {
                "id": "cr_yes",
                "label": "Yes",
                "conditions": {"continuous_release": True},
                "parent": "loc" if show_exclusion else "root",
                "x": horizontal_positions["continuous"],
            },
            {
                "id": "cr_no",
                "label": "No",
                "conditions": {"continuous_release": False},
                "parent": "loc" if show_exclusion else "root",
                "x": horizontal_positions["continuous"],
            },
            {
                "id": "imm_yes_cr_yes",
                "label": "Yes",
                "conditions": {"continuous_release": True, "immediate_ignition": True},
                "parent": "cr_yes",
                "x": horizontal_positions["immediate"],
            },
            {
                "id": "imm_no_cr_yes",
                "label": "No",
                "conditions": {"continuous_release": True, "immediate_ignition": False},
                "parent": "cr_yes",
                "x": horizontal_positions["immediate"],
            },
            {
                "id": "imm_yes_cr_no",
                "label": "Yes",
                "conditions": {"continuous_release": False, "immediate_ignition": True},
                "parent": "cr_no",
                "x": horizontal_positions["immediate"],
            },
            {
                "id": "imm_no_cr_no",
                "label": "No",
                "conditions": {"continuous_release": False, "immediate_ignition": False},
                "parent": "cr_no",
                "x": horizontal_positions["immediate"],
            },
            {
                "id": "del_yes_cr_yes",
                "label": "Yes",
                "conditions": {"continuous_release": True, "immediate_ignition": False, "delayed_ignition": True},
                "parent": "imm_no_cr_yes",
                "x": horizontal_positions["delayed"],
            },
            {
                "id": "del_no_cr_yes",
                "label": "No",
                "conditions": {"continuous_release": True, "immediate_ignition": False, "delayed_ignition": False},
                "parent": "imm_no_cr_yes",
                "x": horizontal_positions["delayed"],
            },
            {
                "id": "del_yes_cr_no",
                "label": "Yes",
                "conditions": {"continuous_release": False, "immediate_ignition": False, "delayed_ignition": True},
                "parent": "imm_no_cr_no",
                "x": horizontal_positions["delayed"],
            },
            {
                "id": "del_no_cr_no",
                "label": "No",
                "conditions": {"continuous_release": False, "immediate_ignition": False, "delayed_ignition": False},
                "parent": "imm_no_cr_no",
                "x": horizontal_positions["delayed"],
            },
            {
                "id": "conf_yes_cr_yes_del_yes",
                "label": "Yes",
                "conditions": {
                    "continuous_release": True,
                    "immediate_ignition": False,
                    "delayed_ignition": True,
                    "confined_space": True,
                },
                "parent": "del_yes_cr_yes",
                "x": horizontal_positions["confined"],
            },
            {
                "id": "conf_no_cr_yes_del_yes",
                "label": "No",
                "conditions": {
                    "continuous_release": True,
                    "immediate_ignition": False,
                    "delayed_ignition": True,
                    "confined_space": False,
                },
                "parent": "del_yes_cr_yes",
                "x": horizontal_positions["confined"],
            },
            {
                "id": "conf_yes_cr_no_del_yes",
                "label": "Yes",
                "conditions": {
                    "continuous_release": False,
                    "immediate_ignition": False,
                    "delayed_ignition": True,
                    "confined_space": True,
                },
                "parent": "del_yes_cr_no",
                "x": horizontal_positions["confined"],
            },
            {
                "id": "conf_no_cr_no_del_yes",
                "label": "No",
                "conditions": {
                    "continuous_release": False,
                    "immediate_ignition": False,
                    "delayed_ignition": True,
                    "confined_space": False,
                },
                "parent": "del_yes_cr_no",
                "x": horizontal_positions["confined"],
            },
        ]
    )

    if show_exclusion:
        for node in decision_nodes:
            if node["id"] != "root":
                node["conditions"]["excluded"] = False

    # Compute vertical positions by averaging descendant leaves.
    def leaf_positions_for(node):
        candidates = main_leaf_specs if show_exclusion else leaf_specs

        return [leaf["y"] for leaf in candidates if matches(leaf["conditions"], node["conditions"])]

    for node in decision_nodes:
        ys = leaf_positions_for(node)
        node["y"] = sum(ys) / len(ys) if ys else 0.0
        node["kind"] = "decision"

    nodes = decision_nodes + leaf_specs
    counts = {node["id"]: count_for(node["conditions"]) for node in nodes}
    lookup = {node["id"]: node for node in nodes}
    edges = [(node["parent"], node["id"]) for node in nodes if node["parent"]]

    def added_condition_key(node_id):
        node = lookup[node_id]
        parent_id = node.get("parent")
        if not parent_id:
            return None
        parent = lookup[parent_id]

        parent_conditions = parent.get("conditions", {})
        node_conditions = node.get("conditions", {})
        added_keys = list(set(node_conditions.keys()) - set(parent_conditions.keys()))
        if not added_keys:
            return None

        return sorted(added_keys)[0]

    def uncertainty_inputs_for(node_id):
        key = added_condition_key(node_id)
        if not key:
            return [], []

        node = lookup[node_id]
        parent = lookup[node["parent"]]

        parent_conditions = parent.get("conditions", {})
        node_conditions = node.get("conditions", {})
        target_value = node_conditions[key]

        answers = []
        confidences = []
        confidence_key = f"{key}_confidence"

        for event in events:
            if not matches(event, parent_conditions):
                continue
            answers.append(1 if bool(event.get(key)) == target_value else 0)
            try:
                confidences.append(int(event.get(confidence_key, 0)))
            except (TypeError, ValueError):
                confidences.append(0)

        return answers, confidences

    base_height = 6
    fig_height = base_height + (1.5 if show_exclusion else 0)
    fig, ax = plt.subplots(figsize=(8.5, fig_height), dpi=200)

    # Column titles above each question column.
    column_titles = []
    column_titles.extend(
        [
            (horizontal_positions["continuous"], "Continuous\nrelease"),
            (horizontal_positions["immediate"], "Immediate\nignition"),
            (horizontal_positions["delayed"], "Delayed\nignition"),
            (horizontal_positions["confined"], "Confined\nspace"),
            (horizontal_positions["outcome"], "Outcome"),
        ]
    )
    title_y = start_y + spacing * 1.8
    for x, title in column_titles:
        ax.text(
            x,
            title_y,
            title,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    for parent_id, child_id in edges:
        parent = lookup[parent_id]
        child = lookup[child_id]
        ax.plot(
            [parent["x"], child["x"]],
            [parent["y"], child["y"]],
            color="#424242",
            linewidth=1.4,
            zorder=1,
        )

    for node in nodes:
        count = counts[node["id"]]
        if node["id"] == "root":
            text = f"{bold_label(node['label'])}\nTotal: {count}"
        elif node["kind"] == "outcome":
            text = f"{bold_label(node['label'])}\n{count}"
        else:
            added_key = added_condition_key(node["id"])
            skip_uncertainty = (show_exclusion and node["id"] in {"loc"}) or added_key == "continuous_release"
            if skip_uncertainty:
                text = f"{bold_label(node['label'])} ({count})"
            else:
                answers, confidences = uncertainty_inputs_for(node["id"])
                interval = format_uncertainty(answers, confidences)
                text = f"{bold_label(node['label'])} ({count})\n{interval}"

        ax.text(
            node["x"],
            node["y"],
            text,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.5", "fc": "#f7f7f7", "ec": "#555555", "lw": 1.0},
            zorder=2,
        )

    ax.set_xlim(-0.5, horizontal_positions["outcome"] + 0.6)
    leaf_y_values = [leaf["y"] for leaf in leaf_specs]
    min_leaf_y = min(leaf_y_values) if leaf_y_values else -start_y
    max_leaf_y = max(leaf_y_values) if leaf_y_values else start_y
    top_limit = max(title_y + spacing * 0.6, max_leaf_y + spacing)
    bottom_limit = min_leaf_y - spacing
    ax.set_ylim(bottom_limit, top_limit)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    return fig
