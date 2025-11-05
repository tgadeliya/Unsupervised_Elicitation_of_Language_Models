from typing import Dict, List
import pandas as pd
import seaborn as sns


def plot_truthfulqa_single(data: Dict[str, List[float]], savepath: str) -> None:
    """
    Make a single 'TruthfulQA' subplot as a seaborn bar chart with mean +/- SE.
    - Bars appear in the SAME order as keys in `data`.
    - No x-axis tick labels (legend only).
    - Uses seaborn only (no matplotlib imports).
    """
    # Keep insertion order and drop empty series
    labels = [k for k, v in data.items() if v and len(v) > 0]
    if not labels:
        raise ValueError("`data` is empty or has only empty lists.")

    # Long-form dataframe
    rows = []
    for lab in labels:
        for val in data[lab]:
            rows.append({"label": lab, "value": float(val)})
    df = pd.DataFrame(rows)

    # Colors per the paper (fallbacks handled below)
    color_map = {
        "Zero-shot": "#b276b2",
        "Zero-shot (Chat)": "#b276b2",   # same color; hatch applied below
        "Unsupervised (Ours)": "#55c0d9",
        "Golden Supervision": "#f0b55a",
        "Human Supervision": "#6fbf73",
    }
    # Build a palette list in the same order as `labels`
    default_palette = sns.color_palette(n_colors=len(labels))
    palette = [color_map.get(lab, default_palette[i]) for i, lab in enumerate(labels)]

    # Simple mean function so we don't import numpy
    def _mean(x):
        x = list(x)
        return sum(x) / len(x) if len(x) else float("nan")

    sns.set_theme(style="whitegrid")
    ax = sns.barplot(
        data=df,
        x="label",
        y="value",
        order=labels,
        palette=palette,
        estimator=_mean,
        errorbar="se",        # standard error across seeds
        capsize=0.08,         # narrow caps
        err_kws=dict(color="black", linewidth=1),
        edgecolor="black",
        linewidth=0.8,
    )

    # Figure sizing via seaborn's returned Axes
    fig = ax.get_figure()
    fig.set_size_inches(6, 3.5)

    # Hatch only the "Zero-shot (Chat)" bar, add labels for legend
    for patch, lab in zip(ax.patches, labels):
        if lab == "Zero-shot (Chat)":
            patch.set_hatch("..")
        patch.set_label(lab)  # so legend uses the bar rectangles

    # Title & y-axis formatting
    ax.set_title("TruthfulQA")
    ax.set_ylabel("accuracy (%)")
    
    # No x-axis labels/ticks; legend only
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)
    ax.grid(axis="x", visible=False)

    # Legend above, single row
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=len(labels),
        frameon=True,
        fontsize=9,
        handlelength=1.8,
        columnspacing=1.1,
    )

    fig.tight_layout()
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    fig.clear()


# ---- tiny usage example ----
if __name__ == "__main__":
    example = {
        "Zero-shot": [60.2, 59.8, 60.9],
        "Zero-shot (Chat)": [73.5, 74.1, 73.8],
        "Unsupervised (Ours)": [91.0, 90.6, 91.2],
        "Golden Supervision": [92.5, 92.3, 92.8],
        "Human Supervision": [90.5],  # single value -> no error bar
    }
    plot_truthfulqa_single(example, "truthfulqa.png")
