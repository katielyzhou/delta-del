import pandas as pd
import numpy as np
import os
import argparse


def append_train_label(df, target, split_index, split_type, data_path="data"):
    """
    Append split column from target parquet.
    """

    # split file
    df_split = pd.read_parquet(
        os.path.join(data_path, f"{target}_{split_type}.parquet")
    )

    df = df.copy()

    # append split
    df[f"split{split_index}"] = df_split[f"split{split_index}"].values

    return df


def process_parquet_files(file_list, output_file=None,
                          split_type = "random", split_index = 1
                          ):
    all_batches = []
    

    for target_id, file_path in enumerate(file_list):
        print(f"Processing {file_path} as target_id={target_id}")
        df = pd.read_parquet(file_path)

        target_name = os.path.basename(file_path).split('_')[0]

        df = append_train_label(df, target_name, split_index, split_type)

        for rep in range(3):

            batch_target = pd.DataFrame({
                'molecule_hash': df['molecule_hash'],
                'smiles': df['smiles'],
                'target_id': target_id,
                'replicate_id': rep,
                "enrichment": df['target_enrichment'],
                "split": df[f"split{split_index}"],
            })
        
            all_batches.append(batch_target)

    df_all = pd.concat(all_batches, ignore_index=True)
    print(f"Size of final parquet: {len(df_all)}")

    if output_file is None:
        output_file = f"data/processed_{split_type}_s{split_index}.parquet"

    df_all.to_parquet(output_file, engine='pyarrow', index=False)
    print(f"Saved final merged dataset to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Prepare DEL dataset for contrastive learning.")
    parser.add_argument("--input", nargs="+", required=True, help="List of .parquet files, one per target.")
    parser.add_argument("--output", default=None, help="Path to save merged dataset as parquet.")
    parser.add_argument("--split-type", default="random")
    parser.add_argument("--split-index", required=True)
    args = parser.parse_args()

    process_parquet_files(args.input, args.output,
                          split_type = args.split_type, split_index = args.split_index)

if __name__ == "__main__":
    main()