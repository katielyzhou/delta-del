import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
import seaborn as sns


eval_type = "offdna"
split_type = "random"

cmap1 = sns.color_palette("crest", as_cmap=True)
cmap2 = sns.color_palette("flare", as_cmap=True)

eval_values = ["DDR1", "MAPK14"]

train_order = ["FLT3", "ABL1", "ABL2", "DDR1", "DDR2", "AURKA", "CDK2", "MAPK14", "MAPK11"]

# ============================
# Plotting function
# ============================

def plot_eval_type(ax, data, eval_type_value, width_type="large"):

    data = data[data["eval_type"] == eval_type_value].copy()

    data = data[
        (data["train_type"] != "chembl_ic50")
        & (data["train_type"] != "off")
    ].copy()

    if data.empty:
        ax.set_title(f"{eval_type_value} (no data)")
        return

    group_cols = ["train", "train_type"]
    bar_cols = ["delta", "w_poisson"]

    data["train"] = pd.Categorical(
        data["train"],
        categories=train_order,
        ordered=True
    )

    groups = (
        data[group_cols]
        .drop_duplicates()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    bars = (
        data[bar_cols]
        .drop_duplicates()
        .sort_values(bar_cols)
        .reset_index(drop=True)
    )

    n_groups = len(groups)
    n_bars = len(bars)

    x = np.arange(n_groups)

    groups_for_color = list(
        bars.itertuples(index=False, name=None)
    )

    colors = {
        groups_for_color[0]: cmap1(0.2),
        groups_for_color[1]: cmap1(0.6),
        groups_for_color[2]: cmap2(0.4),
        groups_for_color[3]: cmap2(0.8),
    }

    width = 0.2

    for i, (_, bar_setting) in enumerate(bars.iterrows()):

        values = []
        errors = []

        for _, group in groups.iterrows():

            subset = data[
                (data["train"] == group["train"]) &
                (data["train_type"] == group["train_type"]) &
                (data["delta"] == bar_setting["delta"]) &
                (data["w_poisson"] == bar_setting["w_poisson"])
            ]

            if len(subset) == 0:
                values.append(np.nan)
                errors.append(0)
            else:
                values.append(subset["result"].iloc[0])
                errors.append(subset["result_std"].iloc[0])

        offset = (i - n_bars / 2) * width + width / 2

        if bar_setting["w_poisson"]:
            label = (
                f"$\\Delta$ = {bar_setting['delta']}, "
                "$\\mathbf{x}$ = [FP, $\\hat{y}_{\\text{DEL}}$]"
            )
        else:
            label = (
                f"$\\Delta$ = {bar_setting['delta']}, "
                "$\\mathbf{x}$ = [FP]"
            )

        if width_type == "large":
            basecolor = to_rgba(
                colors[
                    (bar_setting["delta"], bar_setting["w_poisson"])
                ]
            )

            alpha = 0.9

            facecolor = (
                alpha * basecolor[0] + (1 - alpha),
                alpha * basecolor[1] + (1 - alpha),
                alpha * basecolor[2] + (1 - alpha),
                1.0
            )
        else:
            facecolor = to_rgba(
                colors[
                    (bar_setting["delta"], bar_setting["w_poisson"])
                ]
            )

        dark_factor = 0.7 if width_type == "large" else 0.5

        ecolor = (
            facecolor[0] * dark_factor,
            facecolor[1] * dark_factor,
            facecolor[2] * dark_factor,
            facecolor[3],
        )

        bar_width = width * 0.8 if width_type == "large" else width * 0.3

        ax.bar(
            x + offset,
            values,
            bar_width,
            yerr=errors,
            label=label,
            color=facecolor,
            edgecolor=facecolor,
            alpha=1.0,
            ecolor=ecolor,
            error_kw={
                "elinewidth": 1.0,
                "capsize": (
                    0
                    if (
                        bar_setting["delta"],
                        bar_setting["w_poisson"]
                    ) == (False, False)
                    else 1
                )
            }
        )

    group_labels = [
        f"{row['train']}"
        for _, row in groups.iterrows()
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(
        group_labels,
        rotation=0,
        ha="center",
        fontsize=7
    )

    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)


# ============================
# Generate figure
# ============================


fig, axes = plt.subplots(
    len(eval_values),
    1,
    figsize=(5.2, 3.5),
    sharex=False,
    sharey=False
)

if len(eval_values) == 1:
    axes = [axes]

for ax, eval_value in zip(axes, eval_values):

    csv_file = f"results/{split_type}/reranking_results_{eval_value.upper()}.csv"

    df = pd.read_csv(csv_file)

    df["result_std"] = df["result_std"].fillna(0)

    df_eval = df[df["eval"] == eval_value].copy()

    plot_eval_type(
        ax,
        df_eval,
        eval_type
    )

    ax.set_ylabel(
        "Spearman's $\\rho$",
        fontsize=7
    )

    ax.tick_params(
        axis="y",
        labelsize=7
    )

    if eval_value == "DDR1":
        ax.set_ylim(-0.3, 0.55)
    else:
        ax.set_ylim(-0.4, 0.9)



axes[0].tick_params(
    axis="x",
    which="both",
    bottom=True,
    labelbottom=False
)


handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles[-4:],
    labels[-4:],
    loc="upper center",
    ncols=4,
    bbox_to_anchor=(0.53, 1.04),
    fontsize=6,
    frameon=False
)

plt.tight_layout()
plt.show()