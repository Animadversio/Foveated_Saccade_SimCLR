#!/bin/bash
#BSUB -n 8
#BSUB -q general
#BSUB -G compute-crponce
#BSUB -J 'simclr_fast_sal_train[7-8]'
#BSUB -gpu "num=1:gmodel=TeslaV100_SXM2_32GB:mode=exclusive_process"
#BSUB -R 'gpuhost'
#BSUB -R 'select[mem>64G]'
#BSUB -R 'rusage[mem=64GB]'
#BSUB -M 64GB
#BSUB -u binxu.wang@wustl.edu
#BSUB -o  /scratch1/fs1/crponce/simclr_fast_salienc_train.%J.%I
#BSUB -a 'docker(pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.9)'

# export LSF_DOCKER_SHM_SIZE=16g
# export LSF_DOCKER_VOLUMES="$HOME:$HOME $SCRATCH1:$SCRATCH1 $STORAGE1:$STORAGE1"

echo "$LSB_JOBINDEX"

param_list=\
'--out_dim 128 --run_label proj128_eval_sal_T3.0_pad --crop_temperature 3.0
--out_dim 128 --run_label proj128_eval_sal_T3.0      --crop_temperature 3.0 --pad_img False
--out_dim 128 --run_label proj128_eval_sal_T0.2_pad --crop_temperature 0.2
--out_dim 128 --run_label proj128_eval_sal_T0.2     --crop_temperature 0.2 --pad_img False
--out_dim 128 --run_label proj128_eval_sal_T0.1_pad --crop_temperature 0.1
--out_dim 128 --run_label proj128_eval_sal_T0.1     --crop_temperature 0.1 --pad_img False
--out_dim 128 --run_label proj128_eval_sal_T0.02_pad --crop_temperature 0.02
--out_dim 128 --run_label proj128_eval_sal_T0.02     --crop_temperature 0.02 --pad_img False
'

export extra_param="$(echo "$param_list" | head -n $LSB_JOBINDEX | tail -1)"
echo "$extra_param"

cd ~/SimCLR-torch/
python run_salcrop.py -data $SCRATCH1/Datasets -dataset-name stl10 --workers 16 --ckpt_every_n_epocs 5 --epochs 100  --batch-size 256  $extra_param 