ln -s ../hrnetv2_pretrained hrnetv2_pretrained
ln -s ../data data

mkdir trained_models
cd trained_models
gdown https://drive.google.com/uc?id=1V_80cbSGWh5bmbxh0XIjl08Bomu0DOi_
unzip facial_landmark_models_direct.zip
cd ../

