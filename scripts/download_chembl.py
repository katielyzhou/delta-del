
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
    'molecule_chembl_id',
    'canonical_smiles',
    'relation',
    'standard_value',
    'standard_units',
    'assay_chembl_id',
    'potential_duplicate'
)


import pandas as pd
df = pd.DataFrame(results)
print(len(df))
assay = new_client.assay

assay_data = assay.filter(
    target_chembl_id=chembl_id
).only(
    'assay_chembl_id',
    'assay_type',
    'confidence_score'
)

assay_df = pd.DataFrame(assay_data)
df = df.merge(assay_df, on="assay_chembl_id", how="left")

df = df[
    (df["assay_type"] == "B") &
    (df["confidence_score"] >= 7) &
    (df["relation"] == "=") &
    (df["potential_duplicate"] == 0) &
    df["standard_units"].notna() &
    (df["standard_units"].str.strip() != "")
]
print(len(df))
# flag duplicates (non-unique molecule IDs)
df["is_non_unique_molecule"] = df.duplicated("molecule_chembl_id", keep=False)

df = df.drop_duplicates(subset=[
    "molecule_chembl_id",
    "standard_value",
    "standard_units",
    "relation"
])

df = df[df["is_non_unique_molecule"] == False]

df = df.rename(columns={
    "canonical_smiles": "smiles",
    "standard_value": "kd"
})

df = df[df["smiles"].notna()]
df = df[df["smiles"].apply(lambda x: isinstance(x, str))]
df = df[df["smiles"].str.len() > 0]

df[["smiles_a", "smiles_b", "smiles_c"]] = ""

print(len(df))
df.to_csv(f"data/{target}_chembl_{metric}.csv", index=False)
