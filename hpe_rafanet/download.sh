cd Datasets/AFLW2000
wget http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip
unzip AFLW2000-3D.zip
mv AFLW2000/* .
cd ../../


mkdir trained_models
cd trained_models
gdown https://drive.google.com/uc?id=1PmRd72Oy1djQ2wY47ez8SWSPdrONaZbQ
unzip hpe_rafanet_models.zip
gdown https://drive.google.com/uc?id=1Usg1cHFp2pgd7k6ck3589L-6zU_mUuhw
unzip RAFANet_multiclass.zip
https://drive.google.com/file/d/1Usg1cHFp2pgd7k6ck3589L-6zU_mUuhw/view?usp=sharing
cd ../