
## Binary-encoded labels for facial labdmark detection


### Introduction 
This is the official code for binary-encoded labels: BEL-J and ORBC.  We evaluate our method on three datasets: COFW, 300W, WFLW, and AFLW.
We use  official code implementation provided by [1] and modify for our proposed approach. We adopt **HRNetV2-W18** for facial landmark detection.

#### Environment setup
This code is developed using on Python 3.6.12 and PyTorch 1.4.0 on Ubuntu 18.04 with NVIDIA GPUs. 

1. Install PyTorch 1.4.0 following the [official instructions](https://pytorch.org/)
2. Install dependencies
````
python -m pip install -r requirements.txt
````


#### Data

Please download images (COFW, 300W, AFLW, WFLW) from official websites and then put them into `images` folder for each dataset.

##### COFW:

Download the \*.mat from [this link](https://drive.google.com/file/d/1Z5KyYqRbymlvtQ7bqP74AgVpm7TEv2z_/view?usp=sharing) and [this link](https://drive.google.com/file/d/1ACitXQigMq7Y3x5fkoXU6eUQIOwT0AqR/view?usp=sharing). 

##### 300W: 
[part1](https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.001), [part2](https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.002), [part3](https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.003),  [part4](https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.004)
Please note that the database is simply split into 4 smaller parts for easier download. In order to create the database you have to unzip part1 (i.e., 300w.zip.001)

##### AFLW:
Please visit [the official website](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/) for download instructions. 
Unzip aflw-images-{0,2,3}.tar.gz (images). 

##### WFLW:
Download the images from [this link](https://drive.google.com/file/d/1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC/view) and unzip the images. 

#### HRNetV2-W18 pretrained model

Download the ImageNet pretarined model to ``hrnetv2_pretrained`` from [this link](https://drive.google.com/file/d/1TWSQEdhGwKQrfYjIiWUvkckRVqI4tyXT/view?usp=sharing)


#### BEL trained models

Downlaod the trained models for BEL to  ``trained_models`` directory from [this link](https://drive.google.com/file/d/1V_80cbSGWh5bmbxh0XIjl08Bomu0DOi_/view?usp=sharing) 

Downlaod the trained models for direct regression to  ``direct_regression/trained_models`` directory from [this link](https://drive.google.com/file/d/1ixQ-QThmuF7sR0mBx6Jf9z14Lw1MW4u7/view?usp=sharing) 

*download.sh provides usefule commands to download the trained models, dataset WFLW/COFW/300LP, and HRNetV2-W18 ImageNet pretrained modes.*
*gdown is a python package.*


Facial_detection director structure should look like this:
````
facial_detection/
├── code_configs
│   ├── *_256_code.pkl
│   ├── *_256_tensor.pkl
├── configs
│   ├── 10_0p0003_16_WFLW_30_V1.yaml
│   ├── 10_0p0007_16_300W_30_V1.yaml
│   ├── 30_0p0005_16_AFLW_30_V1.yaml
│   └── 30_0p0005_16_COFW_30_V1.yaml
├── README.md
├── requirements.txt
├── lib
│   ├── config
│   ├── core
│   ├── datasets
│   ├── __init__.py
│   ├── models
│   └── utils
├── tools
├── trained_models
|    └── *.pth
└── data
    ├── cofw
    |   ├── COFW_test_color.mat
    |   └── COFW_train_color.mat
    ├── 300w
    │   ├── face_landmarks_300w_test.csv
    │   ├── face_landmarks_300w_train.csv
    │   ├── face_landmarks_300w_valid_challenge.csv
    │   ├── face_landmarks_300w_valid_common.csv
    │   ├── face_landmarks_300w_valid.csv
    │   └── images
    ├── aflw
    │   ├── face_landmarks_aflw_test.csv
    │   ├── face_landmarks_aflw_test_frontal.csv
    │   ├── face_landmarks_aflw_train.csv
    │   └── images
    └── wflw
        ├── face_landmarks_wflw_test_blur.csv
        ├── face_landmarks_wflw_test.csv
        ├── face_landmarks_wflw_test_expression.csv
        ├── face_landmarks_wflw_test_illumination.csv
        ├── face_landmarks_wflw_test_largepose.csv
        ├── face_landmarks_wflw_test_makeup.csv
        ├── face_landmarks_wflw_test_occlusion.csv
        ├── face_landmarks_wflw_train.csv
        └── images


````

#### Test

Run following commands to set the output directory location and generate codematrix for hadamrd code
````
export TMPDIR="."
cd code_configs
python generate_had.py
cd ../
````

COFW dataset
````
python tools/test_prune_hrnet6.py --cfg configs/30_0p0005_16_COFW_30_V1.yaml --suf "test" --loss "ce" --code code_configs/belhexj.yaml --model_file trained_models/COFW_BELHEXJ_CE.pth

python tools/test_prune_hrnet6.py --cfg configs/30_0p0005_16_COFW_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belu.yaml --model_file trained_models/COFW_BELU.pth

python tools/test_prune_hrnet6.py --cfg configs/30_0p0005_16_COFW_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belj.yaml --model_file trained_models/COFW_BELJ.pth

python tools/test_prune_hrnet6.py --cfg configs/30_0p0005_16_COFW_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belb1jdj.yaml --model_file trained_models/COFW_BELB1JDJ.pth

python tools/test_prune_hrnet6.py --cfg configs/30_0p0005_16_COFW_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belb2jdj.yaml --model_file trained_models/COFW_BELB2JDJ.pth

python tools/test_prune_hrnet6.py --cfg configs/30_0p0005_16_COFW_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belhexj.yaml --model_file trained_models/COFW_BELHEXJ.pth

python tools/test_prune_hrnet6.py --cfg configs/30_0p0005_16_COFW_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belhad.yaml --model_file trained_models/COFW_BELHAD.pth
````
300W dataset
````
python tools/test_prune_hrnet6.py --cfg configs/10_0p0007_16_300W_30_V1.yaml --suf "test" --loss "ce" --code code_configs/belj.yaml --model_file trained_models/300W_BELJ_CE.pth

python tools/test_prune_hrnet6.py --cfg configs/10_0p0007_16_300W_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belu.yaml --model_file trained_models/300W_BELU.pth

python tools/test_prune_hrnet6.py --cfg configs/10_0p0007_16_300W_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belj.yaml --model_file trained_models/300W_BELJ.pth

python tools/test_prune_hrnet6.py --cfg configs/10_0p0007_16_300W_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belb1jdj.yaml --model_file trained_models/300W_BELB1JDJ.pth

python tools/test_prune_hrnet6.py --cfg configs/10_0p0007_16_300W_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belb2jdj.yaml --model_file trained_models/300W_BELB2JDJ.pth

python tools/test_prune_hrnet6.py --cfg configs/10_0p0007_16_300W_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belhexj.yaml --model_file trained_models/300W_BELHEXJ.pth

python tools/test_prune_hrnet6.py --cfg configs/10_0p0007_16_300W_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belhad.yaml --model_file trained_models/300W_BELHAD.pth
````
WFLW dataset
````
python tools/test_prune_hrnet6.py --cfg configs/10_0p0003_16_WFLW_30_V1.yaml --suf "test" --loss "ce" --code code_configs/belb2jdj.yaml --model_file trained_models/WFLW_BELB2JDJ_CE.pth

python tools/test_prune_hrnet6.py --cfg configs/10_0p0003_16_WFLW_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belu.yaml --model_file trained_models/WFLW_BELU.pth

python tools/test_prune_hrnet6.py --cfg configs/10_0p0003_16_WFLW_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belj.yaml --model_file trained_models/WFLW_BELJ.pth

python tools/test_prune_hrnet6.py --cfg configs/10_0p0003_16_WFLW_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belb1jdj.yaml --model_file trained_models/WFLW_BELB1JDJ.pth

python tools/test_prune_hrnet6.py --cfg configs/10_0p0003_16_WFLW_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belb2jdj.yaml --model_file trained_models/WFLW_BELB2JDJ.pth

python tools/test_prune_hrnet6.py --cfg configs/10_0p0003_16_WFLW_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belhexj.yaml --model_file trained_models/WFLW_BELHEXJ.pth

python tools/test_prune_hrnet6.py --cfg configs/10_0p0003_16_WFLW_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belhad.yaml --model_file trained_models/WFLW_BELHAD.pth

````
AFLW dataset
````
python tools/test_prune_hrnet6.py --cfg configs/30_0p0005_16_AFLW_30_V1.yaml --suf "test" --loss "ce" --code code_configs/belj.yaml --model_file trained_models/AFLW_BELJ_CE.pth

python tools/test_prune_hrnet6.py --cfg configs/30_0p0005_16_AFLW_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belu.yaml --model_file trained_models/AFLW_BELU.pth

python tools/test_prune_hrnet6.py --cfg configs/30_0p0005_16_AFLW_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belj.yaml --model_file trained_models/AFLW_BELJ.pth

python tools/test_prune_hrnet6.py --cfg configs/30_0p0005_16_AFLW_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belb1jdj.yaml --model_file trained_models/AFLW_BELB1JDJ.pth

python tools/test_prune_hrnet6.py --cfg configs/30_0p0005_16_AFLW_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belb2jdj.yaml --model_file trained_models/AFLW_BELB2JDJ.pth

python tools/test_prune_hrnet6.py --cfg configs/30_0p0005_16_AFLW_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belhexj.yaml --model_file trained_models/AFLW_BELHEXJ.pth

python tools/test_prune_hrnet6.py --cfg configs/30_0p0005_16_AFLW_30_V1.yaml --suf "test" --loss "bce" --code code_configs/belhad.yaml --model_file trained_models/AFLW_BELHAD.pth
````

Training code: (Please make sure to downlaod HRNetV2-W18 ImageNet pretrained model)
````
python tools/train_prune_hrnet6_cross.py --cfg configs/30_0p0005_16_COFW_30_V1.yaml --loss "ce" --code code_configs/belj.yaml --suf "vnew"

Modify the cfg, code, and loss arguments for different dataset, encoding, and loss functions. 
````
## Multiclass classification
Training
````
python tools/train_multiclass.py --cfg configs/30_0p0005_16_COFW_30_V1.yaml --suf "testV1"  --code code_configs/belufull.yaml --loss ce
python tools/train_multiclass.py --cfg configs/10_0p0007_16_300W_30_V1.yaml --suf "testV1"  --code code_configs/belufull.yaml --loss ce
python tools/train_multiclass.py --cfg configs/10_0p0003_16_WFLW_30_V1.yaml --suf "testV1"  --code code_configs/belufull.yaml --loss ce
python tools/train_multiclass.py --cfg configs/30_0p0005_16_AFLW_30_V1.yaml --suf "testV1"  --code code_configs/belufull.yaml --loss ce
````
Change code argument to code_config/belu.yaml to use two fully connected layers after the feature extractor. 
## Direct regression
Setup
```` 
cd direct_regression
bash setup.sh
````
Training:
````
CUDA_VISIBLE_DEVICES=0 python tools/train_prune_hrnet6_L1.py --cfg configs/30_0p0003_16_COFW_30_V1.yaml --suf "testV1"  --code code_configs/direct_10.yaml

CUDA_VISIBLE_DEVICES=1 python tools/train_prune_hrnet6_L1.py --cfg configs/10_0p0003_16_300W_30_V1.yaml --suf "testV1"  --code code_configs/direct_10.yaml

CUDA_VISIBLE_DEVICES=2 python tools/train_prune_hrnet6_L1.py --cfg configs/10_0p0003_16_WFLW_30_V1.yaml --suf "testV1"  --code code_configs/direct_10.yaml

CUDA_VISIBLE_DEVICES=3 python tools/train_prune_hrnet6_L1.py --cfg configs/30_0p0003_16_AFLW_30_V1.yaml --suf "testV1"  --code code_configs/direct_10.yaml

````
Use direct_10_2fc to train with 2 fully connected layer for the regressor.  
Test with trained models:
````
python tools/test_prune_hrnet6.py --cfg configs/30_0p0003_16_COFW_30_V1.yaml --suf "test" --code code_configs/direct_10.yaml --model_file trained_models/COFW.pth
python tools/test_prune_hrnet6.py --cfg configs/10_0p0003_16_300W_30_V1.yaml --suf "test" --code code_configs/direct_10.yaml --model_file trained_models/300W.pth
python tools/test_prune_hrnet6.py --cfg configs/10_0p0003_16_WFLW_30_V1.yaml --suf "test" --code code_configs/direct_10.yaml --model_file trained_models/WFLW.pth
python tools/test_prune_hrnet6.py --cfg configs/30_0p0003_16_AFLW_30_V1.yaml --suf "test" --code code_configs/direct_10.yaml --model_file trained_models/AFLW.pth
````
## Reference
[1] Deep High-Resolution Representation Learning for Visual Recognition. Jingdong Wang, Ke Sun, Tianheng Cheng, 
    Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, Bin Xiao. Accepted by TPAMI.  [download](https://arxiv.org/pdf/1908.07919.pdf)

Official implementation: [https://github.com/HRNet/HRNet-Facial-Landmark-Detection](https://github.com/HRNet/HRNet-Facial-Landmark-Detection)
Licensed under the MIT License.