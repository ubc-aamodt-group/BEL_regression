cd datasets/AFLW2000
wget http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip
unzip AFLW2000-3D.zip
mv AFLW2000/* .
bash run_crop_AFLW2000.sh
cd ../../

cd datasets/300W_LP/
gdown https://drive.google.com/uc?id=0B7OEHD3T4eCkVGs0TkhUWFN6N1k
unzip 300W-LP.zip
mv 300W_LP/* .
bash run_crop_300lp.sh
cd ../../

mkdir trained_models
cd trained_models
gdown https://drive.google.com/uc?id=1qM7pBzm7c0i0Sgm0ucE5IYFki5W8ExHi
unzip hpe_resnet_models.zip
gdown https://drive.google.com/uc?id=1SoV50xxfzHm6LhMAn8Ne8JTYlOqdfxjP
unzip ResNet_direct.zip
cd ../

