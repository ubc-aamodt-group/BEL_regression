#!/bin/bash

for dataset in 'afad' 'morph2'; do
	for transform in 'cnt' 'temp' 'nby2hot'; do
		sbatch -p batch --export=DATASET=${dataset},TRANSFORM=${transform} --gres=gpu:1 train.slurm
	done	
done


CUDA_VISIBLE_DEVICES=3 python train.py --num-epochs 50 --transform u --dataset morph2 --gpus 0 --mode train --reverse-transform cor --loss bce
CUDA_VISIBLE_DEVICES=3 python train.py --num-epochs 50 --transform j --dataset morph2 --gpus 0 --mode train --reverse-transform cor --loss bce
CUDA_VISIBLE_DEVICES=3 python train.py --num-epochs 50 --transform e1 --dataset morph2 --gpus 0 --mode train --reverse-transform cor --loss bce
CUDA_VISIBLE_DEVICES=3 python train.py --num-epochs 50 --transform e2 --dataset morph2 --gpus 0 --mode train --reverse-transform cor --loss bce
CUDA_VISIBLE_DEVICES=3 python train.py --num-epochs 50 --transform h --dataset morph2 --gpus 0 --mode train --reverse-transform cor --loss bce

CUDA_VISIBLE_DEVICES=2 python train.py --num-epochs 50 --transform u --dataset morph2 --gpus 0 --mode train --reverse-transform cor --loss ce
CUDA_VISIBLE_DEVICES=2 python train.py --num-epochs 50 --transform j --dataset morph2 --gpus 0 --mode train --reverse-transform cor --loss ce
CUDA_VISIBLE_DEVICES=2 python train.py --num-epochs 50 --transform e1 --dataset morph2 --gpus 0 --mode train --reverse-transform cor --loss ce
CUDA_VISIBLE_DEVICES=2 python train.py --num-epochs 50 --transform e2 --dataset morph2 --gpus 0 --mode train --reverse-transform cor --loss ce
CUDA_VISIBLE_DEVICES=2 python train.py --num-epochs 50 --transform h --dataset morph2 --gpus 0 --mode train --reverse-transform cor --loss ce

CUDA_VISIBLE_DEVICES=1 python train.py --num-epochs 50 --transform u --dataset afad --gpus 0 --mode train --reverse-transform cor --loss bce
CUDA_VISIBLE_DEVICES=1 python train.py --num-epochs 50 --transform j --dataset afad --gpus 0 --mode train --reverse-transform cor --loss bce
CUDA_VISIBLE_DEVICES=1 python train.py --num-epochs 50 --transform e1 --dataset afad --gpus 0 --mode train --reverse-transform cor --loss bce
CUDA_VISIBLE_DEVICES=1 python train.py --num-epochs 50 --transform e2 --dataset afad --gpus 0 --mode train --reverse-transform cor --loss bce
CUDA_VISIBLE_DEVICES=1 python train.py --num-epochs 50 --transform h --dataset afad --gpus 0 --mode train --reverse-transform cor --loss bce

CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform u --dataset afad --gpus 0 --mode train --reverse-transform cor --loss ce
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform j --dataset afad --gpus 0 --mode train --reverse-transform cor --loss ce
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform e1 --dataset afad --gpus 0 --mode train --reverse-transform cor --loss ce
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform e2 --dataset afad --gpus 0 --mode train --reverse-transform cor --loss ce
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform h --dataset afad --gpus 0 --mode train --reverse-transform cor --loss ce

CUDA_VISIBLE_DEVICES=1 python train.py --num-epochs 50 --transform u --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss bce
CUDA_VISIBLE_DEVICES=1 python train.py --num-epochs 50 --transform j --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss bce
CUDA_VISIBLE_DEVICES=1 python train.py --num-epochs 50 --transform e1 --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss bce
CUDA_VISIBLE_DEVICES=1 python train.py --num-epochs 50 --transform e2 --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss bce
CUDA_VISIBLE_DEVICES=1 python train.py --num-epochs 50 --transform h --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss bce

CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform u --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss ce
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform j --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss ce
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform e1 --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss ce
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform e2 --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss ce
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform h --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss ce

CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform u --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss mae
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform j --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss mae
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform e1 --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss mae
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform e2 --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss mae
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform h --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss mae

CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform u --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss mse
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform j --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss mse
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform e1 --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss mse
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform e2 --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss mse
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform h --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss mse

CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform u --dataset morph2 --gpus 0 --mode train --reverse-transform sf --loss bce
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform j --dataset morph2 --gpus 0 --mode train --reverse-transform sf --loss bce
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform u --dataset morph2 --gpus 0 --mode train --reverse-transform sf --loss ce
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform j --dataset morph2 --gpus 0 --mode train --reverse-transform sf --loss ce

for transform in 'u' 'j' 'e1' 'e2' 'h'; do
	for dataset in 'afad' 'morph2'; do
		# cor
		for loss in 'bce' 'ce'; do
			qsub -v DATASET=${dataset},TRANSFORM=${transform},REVERSE_TRANSFORM="cor",LOSS=${loss} launch.pbs
		done
		# ex
		for loss in 'bce' 'ce' 'mae' 'mse'; do
			qsub -v DATASET=${dataset},TRANSFORM=${transform},REVERSE_TRANSFORM="ex",LOSS=${loss} launch.pbs
		done
	done
done

for transform in 'u' 'j'; do
	for dataset in 'afad' 'morph2'; do
		for loss in 'bce' 'ce'; do
			qsub -v DATASET=${dataset},TRANSFORM=${transform},REVERSE_TRANSFORM="sf",LOSS=${loss} launch.pbs
		done
	done
done