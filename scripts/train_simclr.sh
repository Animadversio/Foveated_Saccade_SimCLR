#!/bin/bash
#BSUB -n 4
#BSUB -q general
#BSUB -G compute-crponce
#BSUB -J 'simclr_fast_train[1-2]'
#BSUB -gpu "num=1:gmodel=TeslaV100_SXM2_32GB"
#BSUB -R 'gpuhost'
#BSUB -R 'select[mem>32G]'
#BSUB -R 'rusage[mem=32GB]'
#BSUB -M 32G
#BSUB -u binxu.wang@wustl.edu
#BSUB -o  /scratch1/fs1/crponce/simclr_fast_train.%J.%I
#BSUB -a 'docker(pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.9)'

echo "$LSB_JOBINDEX"

param_list=\
'--out_dim 256 --run_label proj256_eval_test
--out_dim 512 --run_label proj512_eval_test
'

export extra_param="$(echo "$param_list" | head -n $LSB_JOBINDEX | tail -1)"
echo "$extra_param"

cd ~/SimCLR-torch/
python run.py -data $SCRATCH1/Datasets -dataset-name stl10 --workers 12 --epochs 100 --ckpt_every_n_epocs 10  $extra_param