
mkdir trained_models
cd trained_models
gdown https://drive.google.com/uc?id=1V_80cbSGWh5bmbxh0XIjl08Bomu0DOi_
unzip facial_landmark_models.zip
cd ../
cd direct_regression
mkdir trained_models
cd trained_models
gdown https://drive.google.com/uc?id=1ixQ-QThmuF7sR0mBx6Jf9z14Lw1MW4u7
unzip facial_direct.zip
cd ../../

mkdir hrnetv2_pretrained
cd hrnetv2_pretrained
gdown https://drive.google.com/uc?id=1TWSQEdhGwKQrfYjIiWUvkckRVqI4tyXT
cd ../

cd data/wflw
gdown https://drive.google.com/uc?id=1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC
tar –xvzf WFLW_images.tar.gz
cd ../../

cd data/cofw
gdown https://drive.google.com/uc?id=1Z5KyYqRbymlvtQ7bqP74AgVpm7TEv2z_
gdown https://drive.google.com/uc?id=1ACitXQigMq7Y3x5fkoXU6eUQIOwT0AqR
cd ../../
