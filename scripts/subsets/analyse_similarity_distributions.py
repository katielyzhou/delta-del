import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BIN_COUNT_RE = re.compile(r"^bin_(\d+)_count$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarise and plot similarity distributions across evaluation references, targets, sizes, and seeds."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/matched_chembl"),
        help="Root directory produced by the sampling script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis output directory. Defaults to <input-dir>/analysis.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["ddr1", "ddr2", "mapk11", "mapk14"],
        help="Target names to include.",
    )
    parser.add_argument(
        "--enforce-same",
        action="store_true",
        help="Read from seed_<seed>/same instead of seed_<seed>/random.",
    )
    return parser.parse_args()


def infer_run_metadata(csv_path, input_root):
    """Infer reference, sample size, seed, and mode from the output path."""
    parts = csv_path.relative_to(input_root).parts

    # evaluation_<reference>/<size>/seed_<seed>/<mode>/similarity_distribution.csv
    if len(parts) < 5:
        raise ValueError(f"Unexpected output path layout: {csv_path}")

    reference = parts[-5]

    try:
        size = int(parts[-4])
    except ValueError as exc:
        raise ValueError(f"Could not parse sample size from {csv_path}") from exc

    seed_match = re.fullmatch(r"seed_(\d+)", parts[-3])
    if not seed_match:
        raise ValueError(f"Could not parse seed from {csv_path}")

    mode = parts[-2]
    return reference, size, int(seed_match.group(1)), mode


def load_reports(input_root, mode):
    report_paths = sorted(
        input_root.glob(
            f"evaluation_*/*/seed_*/{mode}/similarity_distribution.csv"
        )
    )

    if not report_paths:
        raise FileNotFoundError(
            f"No similarity_distribution.csv files found under {input_root}. "
            "Expected evaluation_<reference>/<size>/seed_<seed>/similarity_distribution.csv."
        )

    rows = []
    for path in report_paths:
        reference, size, seed, actual_mode = infer_run_metadata(path, input_root)
        report = pd.read_csv(path)
        required = {"target"}
        missing = required - set(report.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for _, row in report.iterrows():
            record = row.to_dict()
            record.update(
                evaluation_reference=reference,
                requested_size=size,
                seed=seed,
                sampling_mode=actual_mode,
                report_path=str(path),
            )
            rows.append(record)

    data = pd.DataFrame(rows)
    data["requested_size"] = pd.to_numeric(data["requested_size"], errors="raise")
    data["seed"] = pd.to_numeric(data["seed"], errors="raise")
    return data


def bin_columns(data, pattern):
    found = []
    for column in data.columns:
        match = pattern.fullmatch(column)
        if match:
            found.append((int(match.group(1)), column))
    return [column for _, column in sorted(found)]


def plot_histograms_by_size(data, output_dir, targets):
    """For each reference, target, and seed, plot one histogram per dataset size."""
    plot_dir = output_dir / "histograms_by_size"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for reference in sorted(data["evaluation_reference"].unique()):
        for target in targets:
            target_data = data[
                (data["evaluation_reference"] == reference)
                & (data["target"] == target)
            ]

            for seed in sorted(target_data["seed"].unique()):
                seed_data = target_data[target_data["seed"] == seed]
                sizes = sorted(seed_data["requested_size"].unique())

                if not sizes:
                    continue

                ncols = 3
                nrows = int(np.ceil(len(sizes) / ncols))
                fig, axes = plt.subplots(
                    nrows, ncols,
                    figsize=(5 * ncols, 3.8 * nrows),
                    squeeze=False, sharey=True, sharex=True
                )
                axes = axes.ravel()

                for ax, size in zip(axes, sizes):
                    row = seed_data[seed_data["requested_size"] == size].iloc[0]
                    counts = np.array([
                        row[col] for col in bin_columns(data, BIN_COUNT_RE)
                    ], dtype=float)

                    counts = np.clip(counts, 0, None)
                    bin_edges = np.linspace(0, 1, len(counts) + 1)

                    ax.bar(
                        bin_edges[:-1],
                        counts,
                        width=np.diff(bin_edges),
                        align="edge",
                        edgecolor="black",
                    )

                    ax.set(
                        title=f"Dataset size = {size}",
                        xlabel="Tanimoto similarity",
                        ylabel="Count",
                        xlim=(0, 1),
                    )
                    ax.grid(axis="y", alpha=0.25)

                for ax in axes[len(sizes):]:
                    ax.set_visible(False)

                fig.suptitle(
                    f"Evaluation reference: {reference} | Target: {target} | Seed: {seed}",
                    fontsize=14,
                )
                fig.tight_layout()
                fig.savefig(
                    plot_dir / f"{reference}_{target}_seed{seed}.png",
                    dpi=160,
                )
                plt.close(fig)


def plot_histograms_by_seed(data, output_dir, targets):
    """For each reference, target, and size, plot one histogram per seed."""
    plot_dir = output_dir / "histograms_by_seed"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for reference in sorted(data["evaluation_reference"].unique()):
        for target in targets:
            target_data = data[
                (data["evaluation_reference"] == reference)
                & (data["target"] == target)
            ]

            for size in sorted(target_data["requested_size"].unique()):
                size_data = target_data[target_data["requested_size"] == size]
                seeds = sorted(size_data["seed"].unique())

                if not seeds:
                    continue

                ncols = 3
                nrows = int(np.ceil(len(seeds) / ncols))
                fig, axes = plt.subplots(
                    nrows, ncols,
                    figsize=(5 * ncols, 3.8 * nrows),
                    squeeze=False, sharey=True, sharex=True
                )
                axes = axes.ravel()

                for ax, seed in zip(axes, seeds):
                    row = size_data[size_data["seed"] == seed].iloc[0]
                    counts = np.array([
                        row[col] for col in bin_columns(data, BIN_COUNT_RE)
                    ], dtype=float)

                    counts = np.clip(counts, 0, None)
                    bin_edges = np.linspace(0, 1, len(counts) + 1)

                    ax.bar(
                        bin_edges[:-1],
                        counts,
                        width=np.diff(bin_edges),
                        align="edge",
                        edgecolor="black",
                    )

                    ax.set(
                        title=f"Seed = {seed}",
                        xlabel="Tanimoto similarity",
                        ylabel="Count",
                        xlim=(0, 1),
                    )
                    ax.grid(axis="y", alpha=0.25)

                for ax in axes[len(seeds):]:
                    ax.set_visible(False)

                fig.suptitle(
                    f"Evaluation reference: {reference} | Target: {target} | Dataset size: {size}",
                    fontsize=14,
                )
                fig.tight_layout()
                fig.savefig(
                    plot_dir / f"{reference}_{target}_N{size}.png",
                    dpi=160,
                )
                plt.close(fig)


def main():
    args = parse_args()
    input_root = args.input_dir.resolve()
    output_dir = (args.output_dir or (input_root / "analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "same" if args.enforce_same else "random"

    data = load_reports(input_root, mode)
    targets = [target.lower() for target in args.targets]
    data = data[data["target"].isin(targets)].copy()
    if data.empty:
        raise ValueError(f"No rows found for requested targets: {targets}")

    count_cols = bin_columns(data, BIN_COUNT_RE)

    if not count_cols:
        raise ValueError("No bin_*_count columns found in similarity reports.")

    plot_histograms_by_size(data, output_dir, targets)
    plot_histograms_by_seed(data, output_dir, targets)

    print(f"Loaded {len(data)} target/run reports from {input_root}")
    print(f"Evaluation references: {', '.join(sorted(data['evaluation_reference'].unique()))}")
    print(f"Targets: {', '.join(sorted(data['target'].unique()))}")
    print(f"Analysis written to: {output_dir}")


if __name__ == "__main__":
    main()
