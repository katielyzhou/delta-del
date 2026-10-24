import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter

# =====================================================
# Configuration
# =====================================================

dataset_sizes = [10, 20, 28, 40, 50, 60, 70]
split_type = "random"
eval_values = ["DDR1", "MAPK14"]
delta_value = True
w_poisson_value = False

dataset_types = ["same", "random"]

dataset_titles = {
    "same": "Same Dist. Across Seeds",
    "random": "Different Dist. Across Seeds",
}

train_order = [
    "DDR1",
    "DDR2",
    "MAPK14",
    "MAPK11",
]

palette = sns.color_palette("flare", n_colors=len(train_order))
train_colors = dict(zip(train_order, palette))

reference_lines = {
    "DDR1": 0.3660090497737557,
    "MAPK14": 0.4096867838044309,
}

# Results are saved under:
# results/{split_type}/{output_dir}/reranking_results_{TARGET}.csv
results_dir = Path("results") / split_type


# =====================================================
# Load results across dataset sizes and seeds
# =====================================================

all_results = []

for dataset_type in dataset_types:
    for dataset_size in dataset_sizes:
        for eval_value in eval_values:

            evaluation_dir = (
                results_dir
                / str(dataset_size)
            )

            if not evaluation_dir.exists():
                print(f"WARNING: Directory not found: {evaluation_dir}")
                continue

            # Recursively find result files for this evaluation target
            files = sorted(
                evaluation_dir.glob(
                    f"seed_*/{dataset_type}/reranking_results_{eval_value}.csv"
                )
            )

            if not files:
                print(
                    f"WARNING: No files found for "
                    f"dataset={dataset_type}, "
                    f"evaluation={eval_value}, "
                    f"dataset_size={dataset_size} "
                    f"under {evaluation_dir}"
                )
                continue

            print(
                f"{dataset_type}, {eval_value}, N={dataset_size}: "
                f"found {len(files)} seed files"
            )

            for csv_file in files:
                df = pd.read_csv(csv_file)

                # Extract seed from parent directory
                seed_dir = csv_file.parents[1]
                seed_string = seed_dir.name.removeprefix("seed_")

                try:
                    seed = int(seed_string)
                except ValueError as exc:
                    raise ValueError(
                        f"Could not extract seed from directory: "
                        f"{seed_dir}"
                    ) from exc

                df["seed"] = seed
                df["dataset_size"] = dataset_size
                df["dataset_type"] = dataset_type

                # -----------------------------------------
                # Ensure Boolean columns are interpreted
                # correctly
                # -----------------------------------------

                for col in ["w_poisson", "delta"]:
                    if col not in df.columns:
                        raise ValueError(
                            f"Missing column '{col}' in {csv_file}"
                        )

                    if df[col].dtype == object:
                        df[col] = (
                            df[col]
                            .astype(str)
                            .str.strip()
                            .str.lower()
                            .map({
                                "true": True,
                                "false": False,
                            })
                        )
                    else:
                        df[col] = df[col].astype(bool)

                    if df[col].isna().any():
                        raise ValueError(
                            f"Could not parse all values in column "
                            f"'{col}' in {csv_file}"
                        )

                # -----------------------------------------
                # Validate and filter evaluation results
                # -----------------------------------------

                if "eval" not in df.columns:
                    raise ValueError(
                        f"Missing column 'eval' in {csv_file}"
                    )

                df["eval"] = df["eval"].astype(str).str.upper()

                df = df[
                    (df["eval"] == eval_value) &
                    (df["w_poisson"] == w_poisson_value) &
                    (df["delta"] == delta_value)
                ].copy()

                if not df.empty:
                    all_results.append(df)

# =====================================================
# Combine all seed files
# =====================================================

if not all_results:
    raise ValueError(
        f"No results loaded from {results_dir}. "
        "Check the directory structure, filenames, and filters."
    )

results = pd.concat(all_results, ignore_index=True)

# =====================================================
# Validate and clean data
# =====================================================

required_columns = [
    "dataset_type",
    "dataset_size",
    "seed",
    "eval",
    "train",
    "delta",
    "result",
]

missing_columns = [
    col for col in required_columns
    if col not in results.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

results["eval"] = results["eval"].astype(str).str.upper()
results["train"] = results["train"].astype(str).str.upper()

results["result"] = pd.to_numeric(
    results["result"],
    errors="coerce",
)

results = results.dropna(subset=["result"])

# =====================================================
# Calculate mean and standard deviation across seeds
# =====================================================

group_columns = [
    "dataset_type",
    "dataset_size",
    "eval",
    "train",
    "delta",
]

summary_results = (
    results
    .groupby(group_columns, as_index=False)
    .agg(
        result_mean=("result", "mean"),
        result_std=("result", "std"),
        n_seeds=("seed", "nunique"),
    )
    .sort_values(
        ["dataset_type", "eval", "train", "dataset_size"]
    )
)

# =====================================================
# Create 2 x 2 plot
#
# Rows: DDR1, MAPK14
# Columns: same, random
#
# Lines: mean result across seeds
# Shading: mean +/- 1 SD across seeds
# =====================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(5.0, 4.2),
    sharex=True,
    sharey="row",
    gridspec_kw={"width_ratios": [1, 1]},
)

for row, eval_value in enumerate(eval_values):
    for col, dataset_type in enumerate(dataset_types):

        ax = axes[row, col]

        data = summary_results[
            (summary_results["eval"] == eval_value) &
            (summary_results["dataset_type"] == dataset_type)
        ].copy()

        # ---------------------------------------------
        # Plot each training target
        # ---------------------------------------------

        for train in train_order:
            subset = (
                data[data["train"] == train]
                .sort_values("dataset_size")
            )

            if subset.empty:
                continue

            x = subset["dataset_size"].to_numpy(dtype=float)
            y = subset["result_mean"].to_numpy(dtype=float)
            err = subset["result_std"].to_numpy(dtype=float)

            ax.plot(
                x,
                y,
                marker="o",
                markersize=4,
                linewidth=1.5,
                color=train_colors[train],
                label=train,
            )

            # Mean +/- 1 SD across seeds
            valid = np.isfinite(err)

            if valid.any():
                ax.fill_between(
                    x,
                    y - err,
                    y + err,
                    where=valid,
                    color=train_colors[train],
                    alpha=0.2,
                )

        # ---------------------------------------------
        # Reference performance
        # ---------------------------------------------

        ax.axhline(
            reference_lines[eval_value],
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )

        # ---------------------------------------------
        # Column titles
        # ---------------------------------------------

        if row == 0:
            ax.set_title(
                dataset_titles[dataset_type],
                fontsize=7,
                pad=7,
            )

        # ---------------------------------------------
        # Y-axis labels
        # ---------------------------------------------

        if col == 0:
            ax.set_ylabel(
                f"{eval_value} Spearman's $\\rho$",
                fontsize=7,
            )

        # ---------------------------------------------
        # X-axis
        # ---------------------------------------------

        ax.set_xlabel(
            "Dataset Size" if row == 1 else "",
            fontsize=7,
        )

        if row == 0:
            ax.tick_params(labelbottom=False)

        # ---------------------------------------------
        # Formatting
        # ---------------------------------------------

        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)

        if eval_value == "DDR1":
            ax.yaxis.set_major_formatter(
                FormatStrFormatter("%.2f")
            )
        else:
            ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

# =====================================================
# Shared legend
# =====================================================

handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
    ncol=4,
    fontsize=7,
    frameon=False,
)

fig.align_ylabels(axes[:, 0])
fig.align_ylabels(axes[:, 1])

plt.tight_layout(rect=[0, 0.10, 1, 1])

# =====================================================
# Save and display
# =====================================================

plot_file = "dataset_size_matched.pdf"

fig.savefig(
    plot_file,
    bbox_inches="tight",
)

print(f"Plot saved to: {plot_file}")

plt.show()
