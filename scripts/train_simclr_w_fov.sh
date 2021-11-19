#!/bin/bash
#BSUB -n 8
#BSUB -q general
#BSUB -G compute-crponce
#BSUB -J 'simclr_fast_fov_debug_exps[8-17]'
#BSUB -gpu "num=1:gmodel=TeslaV100_SXM2_32GB:mode=exclusive_process"
#BSUB -R 'gpuhost'
#BSUB -R 'select[mem>48G]'
#BSUB -R 'rusage[mem=48GB]'
#BSUB -M 48GB
#BSUB -u binxu.wang@wustl.edu
#BSUB -o  /scratch1/fs1/crponce/simclr_fast_fov_debug_exps.%J.%I
#BSUB -a 'docker(pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.9)'

# export LSF_DOCKER_SHM_SIZE=16g
# export LSF_DOCKER_VOLUMES="$HOME:$HOME $SCRATCH1:$SCRATCH1 $STORAGE1:$STORAGE1"

echo "$LSB_JOBINDEX"

param_list=\
'--run_label proj256_eval_fov_sal_exp  --disable_crop False --blur True  --foveation False   
--run_label proj256_eval_fov_sal_ctrl  --sal_control True   --blur True  --foveation False   
--run_label proj256_eval_fov_orig_crop  --orig_cropper True  --blur True  --foveation False   
--run_label proj256_eval_fov_nocrop_blur  --disable_crop True  --blur True  --foveation False   
--run_label proj256_eval_fov_nocrop_fvr0_01-0_5_slp006  --disable_crop True  --blur False --foveation True --fov_area_rng 0.01  0.5  --kerW_coef  0.06  
--run_label proj256_eval_fov_nocrop_fvr0_01-0_1_slp006  --disable_crop True  --blur False --foveation True --fov_area_rng 0.01  0.1  --kerW_coef  0.06  
--run_label proj256_eval_fov_nocrop_fvr0_10-0_5_slp006  --disable_crop True  --blur False --foveation True --fov_area_rng 0.10  0.5  --kerW_coef  0.06  
--run_label proj256_eval_fov_sal_exp 
--run_label proj256_eval_fov_sal_ctrl  --sal_control    
--run_label proj256_eval_fov_orig_crop  --orig_cropper   
--run_label proj256_eval_fov_nocrop_blur  --disable_crop   
--run_label proj256_eval_fov_fvr0_01-0_5_slp006  --disable_blur --foveation --fov_area_rng 0.01  0.5  --kerW_coef  0.06  
--run_label proj256_eval_fov_fvr0_01-0_1_slp006  --disable_blur --foveation --fov_area_rng 0.01  0.1  --kerW_coef  0.06  
--run_label proj256_eval_fov_fvr0_10-0_5_slp006  --disable_blur --foveation --fov_area_rng 0.10  0.5  --kerW_coef  0.06  
--run_label proj256_eval_fov_nocrop_fvr0_01-0_5_slp006  --disable_crop   --disable_blur --foveation --fov_area_rng 0.01  0.5  --kerW_coef  0.06  
--run_label proj256_eval_fov_nocrop_fvr0_01-0_1_slp006  --disable_crop   --disable_blur --foveation --fov_area_rng 0.01  0.1  --kerW_coef  0.06  
--run_label proj256_eval_fov_nocrop_fvr0_10-0_5_slp006  --disable_crop   --disable_blur --foveation --fov_area_rng 0.10  0.5  --kerW_coef  0.06  
'

export extra_param="$(echo "$param_list" | head -n $LSB_JOBINDEX | tail -1)"
echo "$extra_param"

cd ~/SimCLR-torch/
python run_salcrop.py -data $SCRATCH1/Datasets -dataset-name stl10 --workers 16 --ckpt_every_n_epocs 5 --epochs 100  --batch-size 256  --out_dim 256  $extra_param
# don't break line without line breaker in bash! 