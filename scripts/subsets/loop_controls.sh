#!/usr/bin/env bash
set -euo pipefail

SCRIPT="reranking.py"
DATA_ROOT="data/matched_chembl"

REFERENCES=("ddr1" "mapk14")
SIZES=(10 20 28 40 50 60 70)
SEEDS=(42 43 44 45 46 47 48 49 50 51)
MODES=("random" "same")

for reference in "${REFERENCES[@]}"; do

    # Map evaluation reference to target ID
    case "$reference" in
        ddr1)
            TARGET_ID=0
            ;;
        mapk14)
            TARGET_ID=1
            ;;
        *)
            echo "Unknown reference: $reference" >&2
            exit 1
            ;;
    esac

    for size in "${SIZES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for mode in "${MODES[@]}"; do

                DATA_DIR="${DATA_ROOT}/evaluation_${reference}/${size}/seed_${seed}/${mode}"

                echo "========================================"
                echo "Reference: $reference"
                echo "Target ID: $TARGET_ID"
                echo "Size:      $size"
                echo "Seed:      $seed"
                echo "Mode:      $mode"
                echo "Directory: $DATA_DIR"

                if [[ ! -d "$DATA_DIR" ]]; then
                    echo "Skipping: directory does not exist"
                    continue
                fi

                python "scripts/$SCRIPT" \
                    --data-dir "$DATA_DIR" \
                    --target-id "$TARGET_ID" \
		    --output-dir "${size}/seed_${seed}/${mode}" \
		    --train-targets ddr1 ddr2 mapk11 mapk14

            done
        done
    done
done

echo "All combinations processed."
