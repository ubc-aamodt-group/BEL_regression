python 300LP_AFLW/resnet_test.py --data_dir datasets/AFLW2000/ --filename_list datasets/AFLW2000/overall_filelist_clean.txt --test_filename_list datasets/AFLW2000/overall_filelist_clean.txt --output_string test --dataset Pose_300W_LP_random_ds --batch_size 8 --lr 0.0001 --num_epochs 50 --arch resnet --num_bits 10 --gpu 0 --model_file trained_models/BCE_CE_AFLW_BELU.pth

python belj_aflw2000/resnet_test.py --data_dir datasets/AFLW2000/ --filename_list datasets/AFLW2000/overall_filelist_clean.txt --test_filename_list datasets/AFLW2000/overall_filelist_clean.txt --output_string test --dataset Pose_300W_LP_random_ds --batch_size 8 --lr 0.0001 --num_epochs 50 --arch resnet --num_bits 10 --gpu 0 --model_file trained_models/BCE_CE_AFLW_BELJ.pth


python belu_biwi/resnet_test.py --data_dir datasets/biwi/hpdb --filename_list datasets/biwi/70_train.txt --test_filename_list datasets/biwi/30_test.txt --output_string test --dataset BIWI --batch_size 8 --lr 0.0001 --num_epochs 50 --arch resnet --num_bits 10 --gpu 0 --model_file trained_models/BCE_CE_BIWI_BELU.pth

python belj_biwi/resnet_test.py --data_dir datasets/biwi/hpdb --filename_list datasets/biwi/70_train.txt --test_filename_list datasets/biwi/30_test.txt --output_string test --dataset BIWI --batch_size 8 --lr 0.0001 --num_epochs 50 --arch resnet --num_bits 10 --gpu 0 --model_file trained_models/BCE_CE_BIWI_BELJ.pth
