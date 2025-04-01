# GCSA

Image retrieval re-ranking with side information using *Generalized Contextual Similarity Aggregation* (GCSA).

This repository contains the code for the paper **Visual Re-Ranking with Non-Visual Side Information**, to appear at Scandinavian Conference on Image Analysis (SCIA) 2025.

Project page: https://ghanning.github.io/GCSA/

## Installation

Install the required Python packages as follows:

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

### C extension

An optional C extension to help speed up the 2D field-of-view overlap computations is available, with build instructions below.

#### Ubuntu

```
sudo apt install libgeos-dev
python setup.py build_ext --inplace
```

#### macOS (using [Homebrew](https://brew.sh))

```
brew install geos
python setup.py build_ext --inplace --include-dirs=/opt/homebrew/include/ --library-dirs=/opt/homebrew/lib/
```

## Datasets

Download the [Mapillary Street-level Sequences](https://www.mapillary.com/dataset/places) (SLS) and [LaMAR](https://lamar.ethz.ch/) datasets from their respective web sites and unzip into a subdirectory named "datasets":

```
.
└── datasets
    ├── lamar
    │   ├── CAB
    │   ├── HGE
    │   └── LIN
    └── msls
        ├── test
        └── train_val
```

### Metadata for Mapillary SLS test set

The Mapillary SLS dataset does not include the files **postprocessed.csv** and **raw.csv**, which contain GPS positions and heading angles, for the cities in the test set. When positional affinity is used our model requires this information and these files were therefore reconstructed from [Mapillary's public API](https://www.mapillary.com/developer/api-documentation). The resulting CSV files are checked into this repository and can be unzipped with the command

```bash
unzip data/msls_test_meta.zip -d datasets/msls/
```

### Labels for LaMAR

For LaMAR we establish ground truth labels based on the Sampson score between query and database images. These labels have been precomputed and are stored in the "data" folder.

## Preprocessing

### Image descriptors

Use the provided scripts to extract NetVLAD descriptors with [hloc](https://github.com/cvg/Hierarchical-Localization):

```
./scripts/msls_netvlad.sh
./scripts/lamar_netvlad.sh
```

#### Other descriptors

Only NetVLAD is supported out-of-the-box. To use another global image descriptor:

* Generate HDF5 files containing the descriptors, with the same structure as in hloc.
* Create new training configs by copying the existing ones and changing **desc_fn** and **input_dim**.

*Note*: The ground truth labels for LaMAR were generated for the top k database matches according to the NetVLAD descriptor similarity and would therefore need to be recomputed when using another descriptor.

## Training

Training is done in two stages. First the linear projection $W$ is pre-trained, followed by training of the full network (with the weights of $W$ frozen).

*Tip*: Pass the `--wandb_project <PROJECT>` argument to the training script to log the results to [Weights & Biases](https://wandb.ai).

### Mapillary SLS

Run the pre-training with

```
python -m gcsa.train --name <PRETRAIN_NAME> --conf configs/msls_pretrain.yaml
```

where `<PRETRAIN_NAME>` is the name of the run (for example "msls-pretrain").

Next train the rest of the network by running

```
python -m gcsa.train --name <NAME> --conf configs/msls_train.yaml train.checkpoint=outputs/training/<PRETRAIN_NAME>/checkpoint_10.pth.tar
```

Here the last checkpoint of the pre-training is specified, from which the model weights are initialized.

The trained GCSA network should reach a mAP@10 of around 60.27% on the validation set (as opposed to 32.64% without re-ranking).

### LaMAR

Run the two training stages with

```
python -m gcsa.train --name <PRETRAIN_NAME> --conf configs/lamar_pretrain.yaml
```

and

```
python -m gcsa.train --name <NAME> --conf configs/lamar_train.yaml train.checkpoint=outputs/training/<PRETRAIN_NAME>/checkpoint_10.pth.tar
```

respectively. The network should achieve a validation mAP@10 of approximately 62.48% after the second training stage (compared to 45.08% without re-ranking).

## Evaluation

### Mapillary SLS

Evaluation on the test set can be performed by running

```
python -m gcsa.msls.test --checkpoint <CHECKPOINT_PATH> --output <OUTPUT_PATH>
```

and submitting the zipped .csv file to the [MSLS Place recognition challenge](https://codalab.lisn.upsaclay.fr/competitions/865).

### LaMAR

The code for localizing the query images in the test set is not available at this point.

## Pre-trained weights

Checkpoints for our full model - using positional, heading (Mapillary SLS) and radio (LaMAR) affinity - can be found in the "checkpoints" directory.

## Demo

Try out the re-ranking with the Jupyter notebook [demo.ipynb](demo.ipynb).

## BibTex Citation

```
@inproceedings{hanning2025visual,
  title={{Visual Re-Ranking with Non-Visual Side Information}},
  author={Hanning, Gustav and Flood, Gabrielle and Larsson, Viktor},
  booktitle={Scandinavian Conference on Image Analysis (SCIA)},
  year={2025}
}
```

