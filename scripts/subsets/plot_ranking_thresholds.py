
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter


# =====================================================
# Configuration
# =====================================================

thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
eval_type = "offdna"
split_type = "random"

eval_values = ["DDR1", "MAPK14"]
delta_value = True
w_poisson_value = False

train_order = ["DDR1", "DDR2","MAPK14", "MAPK11"]

palette = sns.color_palette("flare", n_colors=len(train_order))
train_colors = dict(zip(train_order, palette))

reference_lines = {
    "DDR1": 0.3660090497737557,
    "MAPK14": 0.4096867838044309
}

# =====================================================
# Load reranking results across thresholds
# =====================================================

all_results = []

for threshold in thresholds:
    for eval_value in eval_values:

        csv_file = (
            f"results/{split_type}/"
            f"reranking_results_{eval_value}_threshold{threshold:.1f}.csv"
        )

        df = pd.read_csv(csv_file)
        df["threshold"] = threshold

        # Convert Boolean columns safely
        for col in ["w_poisson", "delta"]:
            if df[col].dtype == object or str(df[col].dtype) == "str":
                df[col] = (
                    df[col].astype(str)
                    .str.strip()
                    .str.lower()
                    .map({"true": True, "false": False})
                )
            else:
                df[col] = df[col].astype(bool)

        df = df[
            (df["eval"].str.upper() == eval_value) &
            (df["w_poisson"] == w_poisson_value) &
            (df["delta"] == delta_value)
        ].copy()

        all_results.append(df)

results = pd.concat(all_results, ignore_index=True)

# Normalize labels for consistent matching
results["eval"] = results["eval"].str.upper()
results["train"] = results["train"].str.upper()

# =====================================================
# Load training molecule counts
# =====================================================

counts_df = pd.read_csv(
    f"results/{split_type}/training_molecule_counts.csv"
)

counts_df["eval"] = counts_df["eval"].str.upper()
counts_df["train"] = counts_df["train"].str.upper()

counts_df["threshold"] = pd.to_numeric(counts_df["threshold"])
counts_df["n_molecules"] = pd.to_numeric(counts_df["n_molecules"])

# Keep the thresholds and evaluation targets used in this figure
counts_df = counts_df[
    counts_df["eval"].isin(eval_values) &
    counts_df["threshold"].isin(thresholds)
].copy()

# =====================================================
# Create 2 x 2 figure
#
# Rows:
#   Top    = DDR1
#   Bottom = MAPK14
#
# Columns:
#   Left   = Spearman's rho
#   Right  = Training dataset size
# =====================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(5.2, 4.2),
    sharex=True,
    gridspec_kw={"width_ratios": [1, 1]}
)

# =====================================================
# Plot performance and dataset size
# =====================================================

for row, eval_value in enumerate(eval_values):

    # -------------------------------------------------
    # Column 1: Spearman's rho vs threshold
    # -------------------------------------------------

    ax = axes[row, 0]

    data = results[
        results["eval"] == eval_value
    ].copy()

    for train in train_order:

        subset = data[
            data["train"] == train
        ].copy()

        if subset.empty:
            continue

        summary = (
            subset.groupby("threshold", as_index=False)
            .agg(
                result=("result", "mean"),
                result_std=("result_std", "mean")
            )
            .sort_values("threshold")
        )

        x = summary["threshold"].to_numpy(dtype=float)
        y = summary["result"].to_numpy(dtype=float)
        err = summary["result_std"].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=train_colors[train],
            label=train
        )

        ax.fill_between(
            x,
            y - err,
            y + err,
            color=train_colors[train],
            alpha=0.2
        )

    # Reference performance
    ax.axhline(
        reference_lines[eval_value],
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.7
    )

    # -------------------------------------------------
    # Column 2: Training dataset size vs threshold
    # -------------------------------------------------

    ax_size = axes[row, 1]

    size_data = counts_df[
        counts_df["eval"] == eval_value
    ].copy()

    for train in train_order:

        subset = size_data[
            size_data["train"] == train
        ].copy()

        if subset.empty:
            continue

        summary = (
            subset.groupby("threshold", as_index=False)
            .agg(n_molecules=("n_molecules", "mean"))
            .sort_values("threshold")
        )

        ax_size.plot(
            summary["threshold"].to_numpy(dtype=float),
            summary["n_molecules"].to_numpy(dtype=float),
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=train_colors[train],
            label=train
        )

    # DDR1 reranking performance: y-axis ticks to 2 decimal places
    if eval_value == "DDR1":
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # Right column: move y-axis ticks and label to the right
    ax_size.yaxis.tick_right()
    ax_size.yaxis.set_label_position("right")


    # -------------------------------------------------
    # Shared formatting for each row
    # -------------------------------------------------

    # Row labels
    ax.set_ylabel(
        f"{eval_value} Spearman's $\\rho$",
        fontsize=7
    )

    ax_size.set_ylabel(
        "Number of Molecules",
        fontsize=7
    )

    for current_ax in [ax, ax_size]:
        current_ax.set_xticks(thresholds)
        current_ax.set_xlim(0.18, 0.72)
        current_ax.grid(axis="y", alpha=0.3)
        current_ax.set_axisbelow(True)
        current_ax.tick_params(labelsize=7)

    # X-axis labels on bottom row only
    if row == 1:
        ax.set_xlabel(
            "Tanimoto Sim. Threshold",
            fontsize=7
        )

        ax_size.set_xlabel(
            "Tanimoto Sim. Threshold",
            fontsize=7
        )

# =====================================================
# Column titles
# =====================================================

axes[0, 0].set_title(
    "$\\Delta$-Learning Performance",
    fontsize=8, pad=7
)

axes[0, 1].set_title(
    "Training Dataset Size",
    fontsize=8, pad=7
)

# =====================================================
# Shared legend
# =====================================================

handles = [
    plt.Line2D(
        [0], [0],
        color=train_colors[train],
        marker="o",
        linewidth=1.5,
        markersize=4,
        label=train
    )
    for train in train_order
]

fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
    ncol=4,
    fontsize=7,
    frameon=False,
)


fig.align_ylabels(axes[:, 0])
fig.align_ylabels(axes[:, 1])

plt.tight_layout(rect=[0, 0.10, 1, 1])
fig.subplots_adjust(wspace=0.2)

plt.savefig(
    f"threshold_comparison.pdf",
    dpi=600,
    bbox_inches="tight"
)

plt.show()
