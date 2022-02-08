## Binary-encoded labels for head pose estimation (RAFA-Net)


### Introduction 
This is the official code for binary-encoded labels.  We evaluate our method on three datasets: 300LP/AFLW2000 and BIWI. RAFA-Net [1] feature extractor is used for this implmentation. 
We use  official code implementation provided by [1] and modify the regressor for our proposed approach BEL. 

#### Environment setup
This code is developed using on Python 3.6.5 and Keras 2.2.4 and Tensorflow 1.13.1 on Ubuntu 18.04 with NVIDIA GPUs. 

1. Create a virtual environment and activate with python==3.6.5 (**optional to use virtual environment) 
2. Install dependencies
````
python -m pip install -r requirements.txt
````


#### Data

1. You need to download images AFLW2000 and BIWI from official websites and then put them into `Datasets/*` folder for each dataset. 

Useful links:
AFLW2000: [Download link](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip)
````
cd Datasets
wget http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip
unzip AFLW2000-3D.zip
````
BIWI: [Contact information](https://icu.ee.ethz.ch/research/datsets.html)

#### Trained models

Please download the trained models from [this link](https://drive.google.com/file/d/1PmRd72Oy1djQ2wY47ez8SWSPdrONaZbQ/view?usp=sharing) and [this link](https://drive.google.com/file/d/1Usg1cHFp2pgd7k6ck3589L-6zU_mUuhw/view?usp=sharing
) to trained_models directory.


*download.sh provides useful commands to download the trained models and dataset AFLW2000.*
*gdown is a python package.*

Your directory should look like this:

````
hpe_rafanet/
├── requirements.txt
├── 300wcrop.txt
├── 300w_euler_cls.txt
├── Aflw2000crop.txt
├── Aflw2000_euler_cls.txt
├── biwi_angle_30.txt
├── biwi_rect_30.txt
├── biwi_angle_30_s3.txt
├── biwi_rect_30_s3.txt
├── codes
│   ├── *_360_code.pkl
│   ├── *_360_tensor.pkl
├── trained_models
|   ├── AFLW_BELB1JDJ_50
|   ├── AFLW_BELB2JDJ_50
|   ├── AFLW_BELHAD_50
|   ├── AFLW_BELHEXJ_50
|   ├── AFLW_BELJ_10
|   ├── AFLW_BELU_10
|   ├── BIWI_BELB1JDJ_50
|   ├── BIWI_BELB2JDJ_50
|   ├── BIWI_BELHAD_50
|   ├── BIWI_BELHEXJ_50
|   └── BIWI_BELJ_10
|   └── BIWI_BELJ_10_3
|   └── BIWI_BELU_10
├── Datasets
|   ├── AFLW2000
|   │   ├── imgx.jpg
|   │   ├── imgx.mat
|   └── BIWI
|       └── hpdb
|           ├── 01
	:  
            └── 24
├── biwi_test_rafa-net.py
├── pose_data_augmentor.py
├── custom_validate_callback.py
├── attentional_spatial_pooling.py
├── SelfAttention.py
├── SpectralNormalizationKeras.py
└── test_rafa-net.py
````  

#### Test

Protocol 1:
````
python test_rafa-net.py 10 codes/belu_360_code.pkl codes/belu_360_tensor.pkl 360 belu 0.3 trained_models/AFLW_BELU_10

python test_rafa-net.py 10 codes/belj_360_code.pkl codes/belj_360_tensor.pkl 180 belj 0.3 trained_models/AFLW_BELJ_10

python test_rafa-net.py 50 codes/b1jdj_360_code.pkl codes/b1jdj_360_tensor.pkl 91 belb1jdj 0.3 trained_models/AFLW_BELB1JDJ_50 

python test_rafa-net.py 50 codes/b2jdj_360_code.pkl codes/b2jdj_360_tensor.pkl 47 belb2jdj 0.3 trained_models/AFLW_BELB2JDJ_50

python test_rafa-net.py 50 codes/hexj_360_code.pkl codes/hexj_360_tensor.pkl 17 belhexj 0.3 trained_models/AFLW_BELHEXJ_50 

python test_rafa-net.py 50 codes/had_360_code.pkl codes/had_360_tensor.pkl 512 belhad 0.3 trained_models/AFLW_BELHAD_50 

Use this script to run all the test
bash run_p1.sh
````

Protocol 2
````
python biwi_test_rafa-net.py 10 codes/belu_360_code.pkl codes/belu_360_tensor.pkl 360 belu 0.1 trained_models/BIWI_BELU_10

python biwi_test_rafa-net.py 10 codes/belj_360_code.pkl codes/belj_360_tensor.pkl 180 belj 0.1 trained_models/BIWI_BELJ_10

python biwi_test_rafa-net.py 10 codes/belj_360_code.pkl codes/belj_360_tensor.pkl 180 belj3 0.1 trained_models/BIWI_BELJ_10_3

python biwi_test_rafa-net.py 50 codes/b1jdj_360_code.pkl codes/b1jdj_360_tensor.pkl 91 belb1jdj 0.1 trained_models/BIWI_BELB1JDJ_50 

python biwi_test_rafa-net.py 50 codes/b2jdj_360_code.pkl codes/b2jdj_360_tensor.pkl 47 belb2jdj 0.1 trained_models/BIWI_BELB2JDJ_50

python biwi_test_rafa-net.py 50 codes/hexj_360_code.pkl codes/hexj_360_tensor.pkl 17 belhexj 0.1 trained_models/BIWI_BELHEXJ_50 

python biwi_test_rafa-net.py 50 codes/had_360_code.pkl codes/had_360_tensor.pkl 512 belhad 0.1 trained_models/BIWI_BELHAD_50

Use this script to run all the tests.
bash run_p2.sh
````

 
## Reference
[1] A. Behera, Z. Wharton, P. Hewage, and S. Kumar, “Rotation axis focused attention network
314 (rafa-net) for estimating head pose,” in Computer Vision – ACCV 2020, 2021, pp. 223–240.

Official code implementaion: https://github.com/ArdhenduBehera/RAFA-Net/