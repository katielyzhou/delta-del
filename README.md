# delta-del

This repository contains code to perform post-hoc $\Delta$-learning on DNA-encoded library (DEL) models, using Gaussian processes trained on ChEMBL data. Details can be found in the paper [here](https://openreview.net/forum?id=9F1iwx2O71).

## Installation
You can clone the repository with:
```sh
git clone https://github.com/katielyzhou/delta-del
```
To create a *conda* environment containing the required packages

```sh
conda env create -f environment.yml -n delta-del
```

## Usage
To perform $\Delta$-learning, please run:
  ```sh
  python scripts/reranking.py --target-id [ID]
  # [ID] is 0 for DDR1, 1 for MAPK14
  ```
where results can be visualised using ```plot_reranking.py```.

## Contents
The repository contains the following:

```data```: The ChEMBL datasets have been provided. Please download and place the following [KinDEL data](https://github.com/insitro/kindel) here too:
* Files in the ```heldout``` directory.
* Optionally (for training the DEL model): ```{target}_1M.parquet``` files and files in the ```splits``` directory.

```models```: Contains pre-trained DEL models.

```scripts```: Scripts for preprocessing, training, and evaluation. Other scripts included are:

* ```download_chembl.py``` which can be used to extract the ChEMBL datasets.

* The ```subsets``` subdirectory contains code to run additional analyses, such as

  * ```download_chembl_assay.py``` extracts largest-assay data.

  * ```matched_chembl.py``` to generate same-size datasets for the control experiments. Use flag ```--enforce-same``` to enforce the same similarity distribution across seeds.
  * Visualise the distributions with ```analyse_similarity_distributions.py```.
  * Perform the control re-ranking experiments using ```loop_controls.sh```.
  * Generate figures using ```plot_reranking_controls.py```.

* The ```del_model``` directory contains scripts for training the DEL model.

  * ```prepare_kindel.py``` to reformat the KinDEL data for training the DEL model. Please run:
    ```sh
    python scripts/prepare_kindel.py \
        --input data/ddr1_1M.parquet data/mapk14_1M.parquet \
        --split-index [index] \
        --split-type [random/disynthon]
    ```
    where ```[index]``` is 1, 2, 3, 4, or 5.

  * ```prepare_fingerprints.py``` to pre-compute the fingerprints used for training the DEL model. Please run:
      ```sh
    python scripts/prepare_fingerprints.py --input [input_file]
    ```
    where ```[input_file]``` is the path to one of the parquet files prepared from above.

  * ```training.py``` to train the DEL model. Default config file is located at ```configs/config.yaml```.
  * ```evaluate_regression.py``` to evaluate the DEL model, independent of any $\Delta$-learning.
  
## Citation
If you found the code useful, please cite:
> K. L. Y. Zhou, F. Imrie, and A. Bradley, Cross-Protein $\Delta$-Learning for Re-Ranking DNA-Encoded Libraries, in <i> NeurIPS 2026 Workshop on AI for Drug Discovery: Bridging the Translation Gap</i>, 2026.

Link to: [OpenReview](https://openreview.net/forum?id=9F1iwx2O71)