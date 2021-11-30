#!/bin/bash
#BSUB -n 4
#BSUB -q general
#BSUB -G compute-crponce
#BSUB -J 'simclr_fast_sal_new_train[10-18]'
#BSUB -gpu "num=1:gmodel=TeslaV100_SXM2_32GB:mode=exclusive_process"
#BSUB -R 'gpuhost'
#BSUB -R 'select[mem>48G]'
#BSUB -R 'rusage[mem=48GB]'
#BSUB -M 48GB
#BSUB -u binxu.wang@wustl.edu
#BSUB -o  /scratch1/fs1/crponce/simclr_fast_salienc_debug_train.%J.%I
#BSUB -a 'docker(pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.9)'

# export LSF_DOCKER_SHM_SIZE=16g
# export LSF_DOCKER_VOLUMES="$HOME:$HOME $SCRATCH1:$SCRATCH1"
# export LSF_DOCKER_VOLUMES="$HOME:$HOME $SCRATCH1:$SCRATCH1 $STORAGE1:$STORAGE1"

echo "$LSB_JOBINDEX"

param_list=\
'--out_dim 256 --run_label proj256_eval_sal_new_T0.01    --crop_temperature 0.01  --pad_img False
--out_dim 256 --run_label proj256_eval_sal_new_T0.1     --crop_temperature 0.1   --pad_img False
--out_dim 256 --run_label proj256_eval_sal_new_T1.0     --crop_temperature 1.0   --pad_img False
--out_dim 256 --run_label proj256_eval_sal_new_T10.0   --crop_temperature 10.0  --pad_img False
--out_dim 256 --run_label proj256_eval_sal_new_T0.3     --crop_temperature 0.3   --pad_img False
--out_dim 256 --run_label proj256_eval_sal_new_T3.0     --crop_temperature 3.0   --pad_img False
--out_dim 256 --run_label proj256_eval_sal_new_flat     --sal_control True       --pad_img False
--out_dim 256 --run_label proj256_eval_sal_new_T30.0    --crop_temperature 30.0  --pad_img False
--out_dim 256 --run_label proj256_eval_sal_new_T100.0    --crop_temperature 100.0  --pad_img False
--out_dim 256 --run_label proj256_eval_sal_new_flat     --sal_control       
--out_dim 256 --run_label proj256_eval_sal_new_T0.01    --crop_temperature 0.01  
--out_dim 256 --run_label proj256_eval_sal_new_T0.1     --crop_temperature 0.1   
--out_dim 256 --run_label proj256_eval_sal_new_T1.0     --crop_temperature 1.0   
--out_dim 256 --run_label proj256_eval_sal_new_T10.0   --crop_temperature 10.0  
--out_dim 256 --run_label proj256_eval_sal_new_T0.3     --crop_temperature 0.3   
--out_dim 256 --run_label proj256_eval_sal_new_T3.0     --crop_temperature 3.0   
--out_dim 256 --run_label proj256_eval_sal_new_T30.0    --crop_temperature 30.0  
--out_dim 256 --run_label proj256_eval_sal_new_T100.0    --crop_temperature 100.0  
--out_dim 256 --run_label proj256_eval_sal_new_T0.7     --crop_temperature 0.7  
--out_dim 256 --run_label proj256_eval_sal_new_T1.5     --crop_temperature 1.5  
--out_dim 256 --run_label proj256_eval_sal_new_T2.5     --crop_temperature 2.5  
--out_dim 256 --run_label proj256_eval_sal_new_T4.5     --crop_temperature 4.5  
'

export extra_param="$(echo "$param_list" | head -n $LSB_JOBINDEX | tail -1)"
echo "$extra_param"

cd ~/SimCLR-torch/
python run_salcrop.py -data $SCRATCH1/Datasets -dataset-name stl10 --workers 16 --ckpt_every_n_epocs 5 --epochs 100 --batch-size 256 $extra_param
# don't break line without line breaker in bash! 