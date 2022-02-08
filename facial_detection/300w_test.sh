python tools/test_prune_hrnet6.py --cfg configs/10_0p0007_16_300W_30_V1.yaml --suf "test" --code code_configs/belu.yaml --model_file trained_models/300W_BELU.pth

python tools/test_prune_hrnet6.py --cfg configs/10_0p0007_16_300W_30_V1.yaml --suf "test" --code code_configs/belj.yaml --model_file trained_models/300W_BELJ.pth

python tools/test_prune_hrnet6.py --cfg configs/10_0p0007_16_300W_30_V1.yaml --suf "test" --code code_configs/belb1jdj.yaml --model_file trained_models/300W_BELB1JDJ.pth

python tools/test_prune_hrnet6.py --cfg configs/10_0p0007_16_300W_30_V1.yaml --suf "test" --code code_configs/belb2jdj.yaml --model_file trained_models/300W_BELB2JDJ.pth

python tools/test_prune_hrnet6.py --cfg configs/10_0p0007_16_300W_30_V1.yaml --suf "test" --code code_configs/belhexj.yaml --model_file trained_models/300W_BELHEXJ.pth

python tools/test_prune_hrnet6.py --cfg configs/10_0p0007_16_300W_30_V1.yaml --suf "test" --code code_configs/belhad.yaml --model_file trained_models/300W_BELHAD.pth
