#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Ackesnal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:tesla-smx2:2
#SBATCH --mem-per-cpu=10G
#SBATCH -o VOLO_D0_shuffle_6.92M+2.08G_100epoch_out.txt
#SBATCH -e VOLO_D0_shuffle_6.92M+2.08G_100epoch_err.txt

srun ./distributed_train.sh 2 ../data/imagenet/ --model volo_d0_shuffle --img-size 224 -b 256 --lr 1e-3 --warmup-lr 1e-6 --min-lr 1e-6 --drop-path 0.1 --apex-amp --token-label --token-label-size 14 --token-label-data ../data/tlt/label_top5_train_nfnet/ --shuffle true --epochs 100
