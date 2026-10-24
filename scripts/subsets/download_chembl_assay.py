import pandas as pd
from chembl_webresource_client.new_client import new_client

target = "cdk2"
metric = "Kd"

if target == "ddr1":
    chembl_id = "CHEMBL5319"
elif target == "mapk14":
    chembl_id = "CHEMBL260"
elif target == "aurka":
    chembl_id = "CHEMBL4722"
elif target == "cdk2":
    chembl_id = "CHEMBL301"
elif target == "abl1":
    chembl_id = "CHEMBL1862"
elif target == "abl2":
    chembl_id = "CHEMBL4014"
elif target == "mapk11":
    chembl_id = "CHEMBL3961"
elif target == "ddr2":
    chembl_id = "CHEMBL5122"
elif target == "flt3":
    chembl_id = "CHEMBL1974"

activity = new_client.activity

results = activity.filter(
    target_chembl_id=chembl_id,
    standard_type=metric
).only(
    "molecule_chembl_id",
    "canonical_smiles",
    "relation",
    "standard_value",
    "standard_units",
    "assay_chembl_id",
    "potential_duplicate"
)

df = pd.DataFrame(results)

print("Initial activity records:", len(df))

assay = new_client.assay

assay_data = assay.filter(
    target_chembl_id=chembl_id
).only(
    "assay_chembl_id",
    "assay_type",
    "confidence_score"
)

assay_df = pd.DataFrame(assay_data)

df = df.merge(
    assay_df,
    on="assay_chembl_id",
    how="left"
)

df = df[
    (df["assay_type"] == "B") &
    (df["confidence_score"] >= 7) &
    (df["relation"] == "=") &
    df["standard_units"].notna() &
    (df["standard_units"].str.strip() != "")
].copy()

print("After quality filtering:", len(df))

# Remove duplicate records with the same measurement
df = df.drop_duplicates(
    subset=[
        "molecule_chembl_id",
        "standard_value",
        "standard_units",
        "relation"
    ]
)

# Require molecules to have only one remaining measurement
df["is_non_unique_molecule"] = df.duplicated(
    "molecule_chembl_id",
    keep=False
)

df = df[
    ~df["is_non_unique_molecule"]
].copy()

# Remove invalid SMILES
df = df[
    df["canonical_smiles"].notna() &
    df["canonical_smiles"].apply(lambda x: isinstance(x, str)) &
    (df["canonical_smiles"].str.len() > 0)
].copy()

print("After molecule-level filtering:", len(df))

# Select the assay with the largest number of results
# Print assay sizes before and after deduplication

# Count records and unique molecules before deduplication
assay_counts_before = (
    df.groupby("assay_chembl_id")
      .agg(
          n_records_before=("molecule_chembl_id", "size"),
          n_molecules_before=("molecule_chembl_id", "nunique")
      )
)

# Remove duplicate molecules within each assay
df = df.drop_duplicates(
    subset=["assay_chembl_id", "molecule_chembl_id"],
    keep="first"
).copy()

# Count records after deduplication
assay_counts_after = (
    df.groupby("assay_chembl_id")
      .agg(
          n_records_after=("molecule_chembl_id", "size"),
          n_molecules_after=("molecule_chembl_id", "nunique")
      )
)

# Combine before/after statistics
assay_summary = assay_counts_before.join(assay_counts_after)

assay_summary["duplicates_removed"] = (
    assay_summary["n_records_before"] -
    assay_summary["n_records_after"]
)

assay_summary = assay_summary.sort_values(
    "n_molecules_after",
    ascending=False
)

print("\nAssay sizes before and after deduplication:")
print(assay_summary.to_string())


# Select assay with the most unique molecules


if assay_summary.empty:
    raise ValueError("No valid measurements remain after filtering.")

best_assay_id = assay_summary.index[0]

df = df[
    df["assay_chembl_id"] == best_assay_id
].copy()

print("\nSelected assay:", best_assay_id)
print("Unique molecules retained:", len(df))
print("Duplicates removed in selected assay:",
      assay_summary.loc[best_assay_id, "duplicates_removed"])

assay_counts = (
    df.groupby("assay_chembl_id")
      .agg(
          n_measurements=("molecule_chembl_id", "nunique"),
          n_records=("molecule_chembl_id", "size")
      )
      .sort_values(
          ["n_measurements", "n_records"],
          ascending=False
      )
)

print("\nLargest assays:")
print(assay_counts.head(10))

if assay_counts.empty:
    raise ValueError("No valid measurements remain after filtering.")

best_assay_id = assay_counts.index[0]

df = df[
    df["assay_chembl_id"] == best_assay_id
].copy()

print("\nSelected assay:", best_assay_id)
print("Measurements retained:", len(df))

# Prepare output
df = df.rename(
    columns={
        "canonical_smiles": "smiles",
        "standard_value": "kd"
    }
)

df[["smiles_a", "smiles_b", "smiles_c"]] = ""

df.to_csv(
    f"data/{target}_chembl_{metric}-assay.csv",
    index=False
)

print("Saved dataset.")
