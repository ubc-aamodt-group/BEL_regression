# Binary encoded labels for age estimation

## Introduction
This is the official code for binary-encoded labels on the AFAD and MORPH-II datasets. We develop our own implementation based on [1]. We adopt ResNet-50 for age estimation. 

# Environment setup
This code is developed using Python 3.8.3 and PyTorch 1.5.1 on Ubuntu 18.04 with NVIDIA RTX 2080 Ti GPUs.

## Conda
Create the conda environment by:
```
conda env create -f environment.yml
```
Install the age estimation module:
```
pip install -e .
```

# Data
Please download images (AFAD, MORPH-II). We provide the train/valid/test splits used in our experiments in the `dataset` directory

## AFAD

Download AFAD dataset from [this link][https://afad-dataset.github.io/]

Unzip the images into the appropriate folder, then modify `train_main_afad.py` with the base directory. 

## MORPH-II
Acquire the MORPH-II dataset from [this link][https://ebill.uncw.edu/C20231_ustores/web/store_main.jsp?STOREID=4]

Unzip the images into the appropriate folder and run `preprocess-morph2.py` on that folder. 

Final structure should look like:

To run the code, update the BASE_DIR directory in `train_main_afad.py`, `test_main_afad.py`, `train_main_morph2.py`, `test_main_morph2.py`, and `transforms.py`. 

```
age_estimation
 ┣ age_estimation
 ┃ ┣ train.py
 ┃ ┣ train_main_afad.py
 ┃ ┣ train_main_iw.py
 ┃ ┗ train_main_morph2.py
 ┣ ageresnet
 ┃ ┣ data
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┣ afad.py
 ┃ ┃ ┣ imdb_wiki.py
 ┃ ┃ ┣ logger.py
 ┃ ┃ ┣ morph2.py
 ┃ ┃ ┣ record.py
 ┃ ┃ ┗ transforms.py
 ┃ ┣ models
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┣ age_resnet.py
 ┃ ┃ ┣ age_resnet_quant.py
 ┃ ┃ ┣ pruning.py
 ┃ ┃ ┗ quant_layers.py
 ┃ ┣ utils
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┗ function.py
 ┃ ┗ __init__.py
 ┣ dataset
 ┃ ┣ MORPH2-preprocess
 ┃ ┃ ┗ preprocess_morph2.py
 ┃ ┣ afad
 ┃ ┃ ┣ AFAD-FULL/
 ┃ ┃ ┣ afad_test.csv
 ┃ ┃ ┣ afad_train.csv
 ┃ ┃ ┗ afad_valid.csv
 ┃ ┣ morph2
 ┃ ┃ ┣ morph2-aligned/
 ┃ ┃ ┣ morph2_test.csv
 ┃ ┃ ┣ morph2_train.csv
 ┃ ┃ ┗ morph2_valid.csv
 ┃ ┗ wiki.csv
 ┣ encodings/
 ┣ ckpt/
 ┣ README.md
 ┣ environment.yml
 ┣ requirements.txt
 ┗ setup.py
```
# Trained models
Trained models can be downloaded from [https://drive.google.com/drive/folders/1rNu43ENmrtf0ZxSjQlgg7n2MN8rtqtDS?usp=sharing](https://drive.google.com/drive/folders/1rNu43ENmrtf0ZxSjQlgg7n2MN8rtqtDS?usp=sharing)

# Inference
We can run inference via: `CUDA_VISIBLE_DEVICES=0 python train.py  --transform cnt --dataset morph2 --gpus 0 --loss cnt --reverse-transform cnt --ckpt models/morph2_cnt_cnt_cnt.pth.tar`

More details are described in the code.

# Training
We can run training via: `CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --mode train --transform cnt --dataset morph2 --gpus 0 --loss cnt --reverse-transform cnt`
