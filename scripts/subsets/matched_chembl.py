import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs

import os
import sys
sys.path.append(".")
from delta_del.helpers import compute_ecfp


TARGETS = ["ddr1", "ddr2", "mapk11", "mapk14"]
DEFAULT_SIZES = [10, 20, 28, 40, 50, 60, 70]
DEFAULT_SEEDS = list(range(42, 52))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate nested ChEMBL samples with matched distributions of nearest-evaluation Tanimoto similarity."
        )
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="Directory containing evaluation and ChEMBL CSV files.",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=DEFAULT_SIZES, help="Increasing sample sizes, e.g. --sizes 10 20 28 40 50.",
    )
    parser.add_argument(
        "--seeds", type=int,  nargs="+", default=DEFAULT_SEEDS, help="Random seeds used to generate samples.",
    )
    parser.add_argument(
        "--bins", type=int, default=10, help="Number of equal-width Tanimoto similarity bins.",
    )
    parser.add_argument(
        "--smiles-column", type=str, default="smiles", help="Column containing SMILES.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output root. Defaults to <data-dir>/matched_chembl.",
    )
    parser.add_argument(
        "--enforce-same", action="store_true", help= "Use the same similarity-bin quotas across all seeds. Without this flag, quotas are random per seed."
    )
    return parser.parse_args()


def canonicalize_smiles(smiles):
    """Return canonical SMILES, or None if the input is missing or invalid."""
    if pd.isna(smiles):
        return None

    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def load_dataset(path, smiles_column):
    """Load a CSV, canonicalize SMILES, and discard invalid/duplicate molecules."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if smiles_column not in df.columns:
        raise ValueError(
            f"Missing SMILES column '{smiles_column}' in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["_canonical_smiles"] = df[smiles_column].map(canonicalize_smiles)
    df = df.dropna(subset=["_canonical_smiles"])
    return df.drop_duplicates(subset=["_canonical_smiles"]).reset_index(drop=True)


def find_chembl_file(data_dir, target):
    """Locate a ChEMBL input CSV for a target."""
    preferred_paths = [
        data_dir / f"{target}_chembl_Kd.csv",
        data_dir / f"{target}_chembl.csv",
    ]
    for path in preferred_paths:
        if path.exists():
            return path

    pattern = re.compile(
        rf"{re.escape(target)}_chembl(?:_kd)?\.csv",
        flags=re.IGNORECASE,
    )
    matches = [
        path for path in data_dir.glob("*.csv")
        if pattern.fullmatch(path.name)
    ]

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find a ChEMBL CSV for '{target}' in {data_dir}. "
            f"Expected {target}_chembl_Kd.csv or {target}_chembl.csv."
        )
    raise RuntimeError(f"Multiple possible source files for '{target}': {matches}")


def get_nearest_reference_similarities(fingerprints, reference_fingerprints):
    """Return each molecule's maximum Tanimoto similarity to the reference set."""
    if not reference_fingerprints:
        raise ValueError("The evaluation reference has no valid fingerprints.")

    return np.asarray(
        [
            max(DataStructs.BulkTanimotoSimilarity(fp, reference_fingerprints))
            for fp in fingerprints
        ],
        dtype=float,
    )


def assign_similarity_bins(similarities, n_bins):
    """Assign similarities in [0, 1] to equal-width bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(similarities, edges[1:-1], right=False)
    return np.minimum(bin_ids, n_bins - 1)


def _validate_quota_capacity(counts, n, eligible_targets):
    """Return per-bin shared capacities, raising if sample size is infeasible."""
    capacities = np.asarray(counts, dtype=int).min(axis=0)
    if capacities.sum() < n:
        raise ValueError(
            f"Cannot sample N={n} with shared bin quotas across "
            f"({', '.join(eligible_targets)}). Maximum feasible sample size "
            f"is {int(capacities.sum())}."
        )
    return capacities


def calculate_proportional_quotas(
    count_by_target,
    targets_by_size,
    sizes,
    n_bins,
):
    """
    Calculate nested, shared quotas approximately proportional to pooled
    bin frequencies. The resulting quota vectors are identical across seeds.
    """
    quotas_by_size = {}
    previous_quotas = np.zeros(n_bins, dtype=int)

    for n in sizes:
        eligible = targets_by_size[n]
        counts = np.asarray([count_by_target[t] for t in eligible], dtype=int)
        capacities = _validate_quota_capacity(counts, n, eligible)

        pooled_weights = counts.sum(axis=0).astype(float)
        if pooled_weights.sum() == 0:
            raise ValueError("No molecules available for quota allocation.")
        pooled_weights /= pooled_weights.sum()

        quotas = previous_quotas.copy()
        if np.any(quotas > capacities):
            raise ValueError(
                f"Previous quotas exceed available capacities at N={n}."
            )

        while quotas.sum() < n:
            remaining = n - int(quotas.sum())
            room = capacities - quotas
            active = room > 0
            if not active.any():
                raise RuntimeError("No remaining shared capacity for requested size.")

            weights = pooled_weights * active
            if weights.sum() == 0:
                weights = active.astype(float)

            ideal = remaining * weights / weights.sum()
            floors = np.floor(ideal).astype(int)
            quotas += np.minimum(floors, room)

            if quotas.sum() < n:
                fractions = ideal - floors
                for bin_id in np.argsort(-fractions):
                    if quotas.sum() >= n:
                        break
                    if quotas[bin_id] < capacities[bin_id]:
                        quotas[bin_id] += 1

        quotas_by_size[n] = quotas.copy()
        previous_quotas = quotas.copy()

    return quotas_by_size


def calculate_random_quotas(
    count_by_target,
    targets_by_size,
    sizes,
    n_bins,
    seed,
):
    """
    Generate random, nested similarity-bin quotas for one seed.

    Quotas are shared across eligible targets within each sample size, but
    randomly allocated across bins. Quotas never decrease as size increases.
    """
    rng = np.random.default_rng(seed)
    quotas_by_size = {}
    previous_quotas = np.zeros(n_bins, dtype=int)

    for n in sizes:
        eligible = targets_by_size[n]
        counts = np.asarray([count_by_target[t] for t in eligible], dtype=int)
        capacities = _validate_quota_capacity(counts, n, eligible)

        quotas = previous_quotas.copy()
        if np.any(quotas > capacities):
            raise ValueError(
                f"Previous quotas exceed available capacities at N={n}."
            )

        while quotas.sum() < n:
            available_bins = np.flatnonzero(quotas < capacities)
            if len(available_bins) == 0:
                raise RuntimeError("No remaining bin capacity.")

            chosen_bin = rng.choice(available_bins)
            quotas[chosen_bin] += 1

        quotas_by_size[n] = quotas.copy()
        previous_quotas = quotas.copy()

    return quotas_by_size


def extend_sample(previous_sample, df, quotas, rng, n_bins):
    """Extend a sample while retaining every previously selected molecule."""
    previous_smiles_by_bin = {bin_id: set() for bin_id in range(n_bins)}
    if previous_sample is not None and not previous_sample.empty:
        for bin_id, group in previous_sample.groupby("_similarity_bin"):
            previous_smiles_by_bin[int(bin_id)] = set(
                group["_canonical_smiles"].tolist()
            )

    pieces = []
    for bin_id, quota in enumerate(quotas):
        already_selected = previous_smiles_by_bin[bin_id]
        if len(already_selected) > quota:
            raise ValueError(
                f"New quota for bin {bin_id} ({quota}) is smaller than the "
                f"previous sample's count ({len(already_selected)}). "
                "A nested sample cannot shrink a bin's quota."
            )

        needed = int(quota) - len(already_selected)
        pool = df[
            (df["_similarity_bin"] == bin_id)
            & (~df["_canonical_smiles"].isin(already_selected))
        ]

        if len(pool) < needed:
            raise RuntimeError(
                f"Bin {bin_id} has {len(pool)} unselected molecules available, "
                f"but {needed} are needed."
            )

        if needed:
            selected_indices = rng.choice(
                pool.index.to_numpy(),
                size=needed,
                replace=False,
            )
            pieces.append(df.loc[selected_indices])

        if previous_sample is not None and not previous_sample.empty:
            previous_bin = previous_sample[
                previous_sample["_similarity_bin"] == bin_id
            ]
            if not previous_bin.empty:
                pieces.append(previous_bin)

    sampled = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else df.iloc[0:0].copy()
    )

    if len(sampled) != int(quotas.sum()):
        raise RuntimeError(
            f"Expected {quotas.sum()} molecules after extension, got {len(sampled)}."
        )
    if sampled["_canonical_smiles"].duplicated().any():
        raise RuntimeError("Duplicate molecules detected while extending sample.")

    return sampled.iloc[rng.permutation(len(sampled))].reset_index(drop=True)


def load_reference_fingerprints(data_dir, smiles_column, reference_target):
    """Load one evaluation reference without intersecting or combining datasets."""
    reference_path = data_dir / f"{reference_target}_offdna.csv"
    reference_df = load_dataset(reference_path, smiles_column)
    reference_smiles = sorted(reference_df["_canonical_smiles"].unique())

    fingerprints = [
        fp for fp in (
            compute_ecfp(smiles, return_as="fp")
            for smiles in reference_smiles
        )
        if fp is not None
    ]
    if not fingerprints:
        raise ValueError(
            f"The evaluation reference {reference_path} has no valid fingerprints."
        )

    print(
        f"Evaluation reference '{reference_target}' contains "
        f"{len(fingerprints)} valid fingerprints."
    )
    return fingerprints


def load_target_datasets(data_dir, smiles_column, reference_fingerprints, n_bins):
    """Load target datasets and calculate similarity-bin assignments."""
    datasets = {}
    count_by_target = {}

    for target in TARGETS:
        source_path = find_chembl_file(data_dir, target)
        df = load_dataset(source_path, smiles_column)

        fingerprints = [
            compute_ecfp(smiles, return_as="fp")
            for smiles in df["_canonical_smiles"]
        ]
        valid_indices = [
            index for index, fp in enumerate(fingerprints) if fp is not None
        ]
        df = df.iloc[valid_indices].reset_index(drop=True)
        fingerprints = [fingerprints[index] for index in valid_indices]

        if df.empty:
            raise ValueError(f"No valid molecules with fingerprints for {target}.")

        similarities = get_nearest_reference_similarities(
            fingerprints,
            reference_fingerprints,
        )
        df["_nearest_eval_tanimoto"] = similarities
        df["_similarity_bin"] = assign_similarity_bins(similarities, n_bins)

        count_by_target[target] = np.bincount(
            df["_similarity_bin"].to_numpy(),
            minlength=n_bins,
        )
        datasets[target] = {"path": source_path, "df": df}

        print(
            f"{target}: {len(df)} unique molecules; "
            f"nearest-evaluation similarity mean={similarities.mean():.4f}"
        )

    return datasets, count_by_target


def get_targets_by_size(sizes, datasets):
    """Determine which targets can support each requested sample size."""
    targets_by_size = {}

    for n in sizes:
        eligible = [
            target for target in TARGETS
            if len(datasets[target]["df"]) >= n
        ]
        skipped = [target for target in TARGETS if target not in eligible]

        if not eligible:
            raise ValueError(f"No target has enough molecules for requested N={n}.")

        if skipped:
            print(
                f"N={n}: skipping target(s) with fewer than {n} molecules: "
                f"{', '.join(skipped)}"
            )

        targets_by_size[n] = eligible

    return targets_by_size


def save_sample_and_report(
    sampled,
    target,
    seed,
    n,
    source_path,
    output_dir,
    n_bins,
):
    """Save a sampled dataset and return its similarity-distribution report row."""
    similarities = sampled["_nearest_eval_tanimoto"].to_numpy()
    actual_counts = np.bincount(
        sampled["_similarity_bin"].to_numpy(),
        minlength=n_bins,
    )

    output_df = sampled.drop(
        columns=["_canonical_smiles", "_similarity_bin"]
    ).rename(
        columns={"_nearest_eval_tanimoto": "nearest_eval_tanimoto"}
    )

    output_df.to_csv(output_dir / source_path.name, index=False)

    report = {
        "seed": seed,
        "target": target,
        "dataset_size": n,
        "source_file": source_path.name,
        "n_molecules": len(sampled),
        "mean_nearest_eval_tanimoto": float(similarities.mean()),
        "std_nearest_eval_tanimoto": float(similarities.std()),
    }
    for bin_id in range(n_bins):
        report[f"bin_{bin_id}_count"] = int(actual_counts[bin_id])
        report[f"bin_{bin_id}_proportion"] = float(
            actual_counts[bin_id] / len(sampled)
        )

    return report


def main():
    args = parse_args()

    if not args.sizes or any(size < 1 for size in args.sizes):
        raise ValueError("All --sizes values must be positive.")
    if len(set(args.sizes)) != len(args.sizes):
        raise ValueError("--sizes must not contain duplicates.")
    if not args.seeds:
        raise ValueError("At least one seed must be provided.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates.")
    if args.bins < 1:
        raise ValueError("--bins must be at least 1.")

    sizes = sorted(args.sizes)
    data_dir = args.data_dir
    output_root = args.output_dir or data_dir / "matched_chembl"
    bin_edges = np.linspace(0.0, 1.0, args.bins + 1)

    # Run two independent experiments: one against DDR1 evaluation molecules,
    # and one against MAPK14 evaluation molecules. No reference molecules are
    # intersected or combined, and quotas/samples are independent per run.
    for reference_target in ("ddr1", "mapk14"):
        print(f"\n=== Evaluation reference: {reference_target} ===")
        reference_fingerprints = load_reference_fingerprints(
            data_dir,
            args.smiles_column,
            reference_target,
        )
        datasets, count_by_target = load_target_datasets(
            data_dir,
            args.smiles_column,
            reference_fingerprints,
            args.bins,
        )
        targets_by_size = get_targets_by_size(sizes, datasets)

        # Quotas are calculated independently for each evaluation reference.
        # With --enforce-same, each reference uses fixed quotas across seeds.
        shared_quotas = None
        if args.enforce_same:
            shared_quotas = calculate_proportional_quotas(
                count_by_target,
                targets_by_size,
                sizes,
                args.bins,
            )

        reference_output_root = output_root / f"evaluation_{reference_target}"

        for seed in args.seeds:
            quotas_by_size = (
                shared_quotas
                if args.enforce_same
                else calculate_random_quotas(
                    count_by_target,
                    targets_by_size,
                    sizes,
                    args.bins,
                    seed,
                )
            )

            rng = np.random.default_rng(seed)
            previous_samples = {target: None for target in TARGETS}

            for n in sizes:
                output_dir = reference_output_root / str(n) / f"seed_{seed}" / ("same" if args.enforce_same else "random")
                output_dir.mkdir(parents=True, exist_ok=True)

                quotas = quotas_by_size[n]
                report_rows = []

                for target in targets_by_size[n]:
                    sampled = extend_sample(
                        previous_samples[target],
                        datasets[target]["df"],
                        quotas,
                        rng,
                        args.bins,
                    )
                    previous_samples[target] = sampled.copy()

                    report_row = save_sample_and_report(
                        sampled=sampled,
                        target=target,
                        seed=seed,
                        n=n,
                        source_path=datasets[target]["path"],
                        output_dir=output_dir,
                        n_bins=args.bins,
                    )
                    report_row["evaluation_reference"] = reference_target
                    report_rows.append(report_row)

                pd.DataFrame(report_rows).to_csv(
                    output_dir / "similarity_distribution.csv",
                    index=False,
                )
                pd.DataFrame(
                    {
                        "bin": np.arange(args.bins),
                        "similarity_lower": bin_edges[:-1],
                        "similarity_upper": bin_edges[1:],
                        "shared_quota": quotas,
                    }
                ).to_csv(
                    output_dir / "similarity_bin_quotas.csv",
                    index=False,
                )

                print(
                    f"Reference {reference_target}, seed {seed}: saved nested "
                    f"samples of N={n} ({', '.join(targets_by_size[n])}) "
                    f"to {output_dir}"
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
