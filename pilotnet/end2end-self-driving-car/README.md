# Binary-encoded labels for end-to-end autonomous driving

## Introduction

This is the official code for binary-encoded labels on Sully Chen's Driving Dataset. We develop an implementation based on [1] and use PilotNet as our network architecture. 

# Environment setup
This code is developed using Python 3.8.3 and PyTorch 1.5.1 on Ubuntu 18.04 with NVIDIA RTX 2080 Ti GPUs.

## Conda
Create the conda environment by:

```
conda env create -f environment.yml
```

# Data

The dataset is provided by Sully Chen at [2]
Please use the provided `download_dataset.sh` script to download the data and unzip it into end2end-self-driving-car/driving_dataset

We provide train/test splits for the dataset in `train.csv` and `val.csv`. 

Your directory structure should look like:

```
pilotnet
 ┣ end2end-self-driving-car
 ┃ ┣ ckpt/
 ┃ ┣ config
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┗ defaults.py
 ┃ ┣ data
 ┃ ┃ ┣ datasets
 ┃ ┃ ┃ ┣ __init__.py
 ┃ ┃ ┃ ┣ baladmobile.py
 ┃ ┃ ┃ ┗ driving_data.py
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┗ build.py
 ┃ ┣ driving_dataset
 ┃ ┃ ┣ .DS_Store
 ┃ ┃ ┣ data.txt
 ┃ ┃ ┣ train.csv
 ┃ ┃ ┣ val.csv
 ┃ ┃ ┗ vis.csv
 ┃ ┣ encodings/
 ┃ ┣ model
 ┃ ┃ ┣ engine
 ┃ ┃ ┃ ┣ __init__.py
 ┃ ┃ ┃ ┣ evaluation.py
 ┃ ┃ ┃ ┣ pruning.py
 ┃ ┃ ┃ ┣ trainer.py
 ┃ ┃ ┃ ┗ visualization.py
 ┃ ┃ ┣ layer
 ┃ ┃ ┃ ┣ __init__.py
 ┃ ┃ ┃ ┗ feed_forward.py
 ┃ ┃ ┣ meta_arch
 ┃ ┃ ┃ ┣ __init__.py
 ┃ ┃ ┃ ┣ backward_pilot_net.py
 ┃ ┃ ┃ ┣ pilot_net.py
 ┃ ┃ ┃ ┗ pilot_net_analytical.py
 ┃ ┃ ┣ solver
 ┃ ┃ ┃ ┣ __init__.py
 ┃ ┃ ┃ ┗ build.py
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┣ build.py
 ┃ ┃ ┣ conversion_helper.py
 ┃ ┃ ┗ transforms.py
 ┃ ┣ util
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┣ logger.py
 ┃ ┃ ┣ prepare_driving_dataset.py
 ┃ ┃ ┣ save_image.py
 ┃ ┃ ┗ visdom_plots.py
 ┃ ┣ .gitignore
 ┃ ┣ drive.py
 ┃ ┣ get_min_max.py
 ┃ ┗ main.py
 ┣ .gitignore
 ┣ README.md
 ┣ download_dataset.sh
 ┗ environment.yml
```

Please also ensure CODINGS_HOME in `main.py` points to the correct encodings (see age_estimation for more details)
# Testing

Models can be downloaded from [this link][https://drive.google.com/drive/folders/1ZLKYmucHP5DlNXux5_wEecHKl-XOh5iQ?usp=sharing]

You can run them according to their configuration:

TRANSFORM
* u (unary code)
* j (Johnson code)
* e1 (B1JDJ code)
* e2 (B2JDJ code)
* h (hexJ code)
* had (Hadamard code)
* cnt (continuous)
* mc (multiclass classification)

REVERSE_TRANSFORM
* cor (GEN/correlation matrix decoding)
* ex (GENEX/expected value decoding)
* mc (Multiclass classification, only used with multiclass classification encoding)
* sf (special function for unary and johnson codes)

Since there is no need to supply a loss for evaluation, it does not need to be passed. 

```
python main.py --mode test --transform TRANSFORM --reverse-transform REVERSE_TRANSFORM MODEL.WEIGHTS <model-file>
```

For example

```
python main.py --mode test --transform j --reverse-transform cor MODEL.WEIGHTS j_cor_bce.pth.tar
```

# Training

If you would like to train our models, please use the command below. We describe the losses available.

* bce: available for all encodings, all decodings (except multiclass classification)
* ce: available for all encodings
* mae: available for GENEX
* mse: available for GENEX

```
python main.py --transform TRANSFORM --reverse-transform REVERSE_TRANSFORM --loss LOSS
```

For example

```
python main.py --transform j --reverse-transform cor --loss bce
```
# Trained models
Trained models are available at [https://drive.google.com/drive/folders/1ZLKYmucHP5DlNXux5_wEecHKl-XOh5iQ?usp=sharing](https://drive.google.com/drive/folders/1ZLKYmucHP5DlNXux5_wEecHKl-XOh5iQ?usp=sharing)

# References
[1] M. Fathi, MahanFathi/end2end-self-driving-car. 2020. Available: https://github.com/MahanFathi/end2end-self-driving-car
[2]S. Chen, SullyChen/driving-datasets. 2021. Available: https://github.com/SullyChen/driving-datasets
