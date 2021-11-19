#!/bin/bash
#BSUB -n 4
#BSUB -q general
#BSUB -G compute-crponce
#BSUB -J 'simclr_fast_magnif_exps[37]'
#BSUB -gpu "num=1:gmodel=TeslaV100_SXM2_32GB:mode=exclusive_process"
#BSUB -R 'gpuhost'
#BSUB -R 'select[mem>48G]'
#BSUB -R 'rusage[mem=48GB]'
#BSUB -M 48GB
#BSUB -u binxu.wang@wustl.edu
#BSUB -o  /scratch1/fs1/crponce/simclr_fast_magnif_exps.%J.%I
#BSUB -a 'docker(pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.9)'

# export LSF_DOCKER_SHM_SIZE=16g
# export LSF_DOCKER_VOLUMES="$HOME:$HOME $SCRATCH1:$SCRATCH1 $STORAGE1:$STORAGE1"
# export LSF_DOCKER_VOLUMES="$HOME:$HOME $SCRATCH1:$SCRATCH1"
# export CUDA_LAUNCH_BLOCKING=1
echo "$LSB_JOBINDEX"

param_list=\
'--run_label proj256_eval_magnif_bsl --crop  
--run_label proj256_eval_magnif_cvr_0_05-0_70 --magnif --cover_ratio 0.05 0.70  --fov_size 20  --K  20  --sampling_bdr 16  
--run_label proj256_eval_magnif_cvr_0_05-0_35 --magnif --cover_ratio 0.05 0.35  --fov_size 20  --K  20  --sampling_bdr 16 
--run_label proj256_eval_magnif_cvr_0_01-0_35 --magnif --cover_ratio 0.01 0.35  --fov_size 20  --K  20  --sampling_bdr 16 
--run_label proj256_eval_magnif_cvr_0_01-1_50 --magnif --cover_ratio 0.01 1.50  --fov_size 20  --K  20  --sampling_bdr 16
--run_label proj256_eval_magnif_exp_cvr_0_05-1_00_slp_1_50      --magnif --gridfunc_form radial_exp --cover_ratio 0.05  1.5  --slope_C 1.5 --sampling_bdr 16
--run_label proj256_eval_magnif_exp_cvr_0_05-1_00_slp_0_75-3_00 --magnif --gridfunc_form radial_exp --cover_ratio 0.05  1.5  --slope_C 0.75 3.0 --sampling_bdr 16
--run_label proj256_eval_magnif_exp_cvr_0_05-1_00_slp_1_50      --magnif --gridfunc_form radial_exp --cover_ratio 0.05  1.0  --slope_C 1.5 --sampling_bdr 16
--run_label proj256_eval_magnif_exp_cvr_0_05-1_00_slp_0_75-3_00 --magnif --gridfunc_form radial_exp --cover_ratio 0.05  1.0  --slope_C 0.75 3.0 --sampling_bdr 16
--run_label proj256_eval_magnif_exp_cvr_0_05-0_50_slp_1_50      --magnif --gridfunc_form radial_exp --cover_ratio 0.05  0.5  --slope_C 1.5 --sampling_bdr 16
--run_label proj256_eval_magnif_exp_cvr_0_05-0_50_slp_0_75-3_00 --magnif --gridfunc_form radial_exp --cover_ratio 0.05  0.5  --slope_C 0.75 3.0 --sampling_bdr 16
--run_label proj256_eval_magnif_salmap_T1_0_cvr_0_01-0_35 --magnif --sal_sample --sample_temperature 1.0 --cover_ratio 0.01 0.35  --fov_size 20  --K  20  --sampling_bdr 16
--run_label proj256_eval_magnif_salmap_T0_7_cvr_0_01-0_35 --magnif --sal_sample --sample_temperature 0.7 --cover_ratio 0.01 0.35  --fov_size 20  --K  20  --sampling_bdr 16
--run_label proj256_eval_magnif_salmap_T1_5_cvr_0_01-0_35 --magnif --sal_sample --sample_temperature 1.5 --cover_ratio 0.01 0.35  --fov_size 20  --K  20  --sampling_bdr 16
--run_label proj256_eval_magnif_salmap_T0_3_cvr_0_01-0_35 --magnif --sal_sample --sample_temperature 0.3 --cover_ratio 0.01 0.35  --fov_size 20  --K  20  --sampling_bdr 16
--run_label proj256_eval_magnif_salmap_T3_0_cvr_0_01-0_35 --magnif --sal_sample --sample_temperature 3.0 --cover_ratio 0.01 0.35  --fov_size 20  --K  20  --sampling_bdr 16
--run_label proj256_eval_magnif_salmap_T10_0_cvr_0_01-0_35 --magnif --sal_sample --sample_temperature 10.0 --cover_ratio 0.01 0.35  --fov_size 20  --K  20  --sampling_bdr 16
--run_label proj256_eval_magnif_exp_cvr_0_01-0_35_slp_1_50      --magnif --gridfunc_form radial_exp --cover_ratio 0.01  0.35  --slope_C 1.5 --sampling_bdr 16
--run_label proj256_eval_magnif_exp_cvr_0_01-0_35_slp_0_75-3_00 --magnif --gridfunc_form radial_exp --cover_ratio 0.01  0.35  --slope_C 0.75 3.0 --sampling_bdr 16
--run_label proj256_eval_magnif_exp_cvr_0_01-0_15_slp_1_50      --magnif --gridfunc_form radial_exp --cover_ratio 0.01  0.15  --slope_C 1.5 --sampling_bdr 16
--run_label proj256_eval_magnif_exp_cvr_0_01-0_15_slp_0_75-3_00 --magnif --gridfunc_form radial_exp --cover_ratio 0.01  0.15  --slope_C 0.75 3.0 --sampling_bdr 16
--run_label proj256_eval_magnif_exp_cvr_0_05-1_50_slp_1_50      --magnif --gridfunc_form radial_exp --cover_ratio 0.05  1.50  --slope_C 1.5 --sampling_bdr 16
--run_label proj256_eval_magnif_exp_cvr_0_05-1_50_slp_0_75-3_00 --magnif --gridfunc_form radial_exp --cover_ratio 0.05  1.50  --slope_C 0.75 3.0 --sampling_bdr 16
--run_label proj256_eval_magnif_bsl_RND --randomize_seed --crop
--run_label proj256_eval_magnif_cvr_0_05-0_35_RND --randomize_seed --magnif --cover_ratio 0.05 0.35  --fov_size 20  --K  20  --sampling_bdr 16
--run_label proj256_eval_magnif_exp_cvr_0_01-0_35_slp_0_75      --magnif --gridfunc_form radial_exp --cover_ratio 0.01  0.35  --slope_C 0.75  --sampling_bdr 16
--run_label proj256_eval_magnif_exp_cvr_0_01-0_35_slp_3_00      --magnif --gridfunc_form radial_exp --cover_ratio 0.01  0.35  --slope_C 3.00  --sampling_bdr 16
--run_label proj256_eval_magnif_exp_cvr_0_01-0_35_slp_0_75-1_50 --magnif --gridfunc_form radial_exp --cover_ratio 0.01  0.35  --slope_C 0.75  1.5 --sampling_bdr 16
--run_label proj256_eval_magnif_exp_cvr_0_01-0_35_slp_0_25-1_00 --magnif --gridfunc_form radial_exp --cover_ratio 0.01  0.35  --slope_C 0.25  1.0 --sampling_bdr 16
--run_label proj256_eval_magnif_salmap_T0_1_cvr_0_01-0_35 --magnif --sal_sample --sample_temperature 0.1 --cover_ratio 0.01 0.35  --fov_size 20  --K  20  --sampling_bdr 16
--run_label proj256_eval_magnif_salmap_T0_01_cvr_0_01-0_35 --magnif --sal_sample --sample_temperature 0.01 --cover_ratio 0.01 0.35  --fov_size 20  --K  20  --sampling_bdr 16
--run_label proj256_eval_magnif_salmap_T4_5_cvr_0_01-0_35 --magnif --sal_sample --sample_temperature 4.5 --cover_ratio 0.01 0.35  --fov_size 20  --K  20  --sampling_bdr 16
--run_label proj256_eval_magnif_salmap_T6_0_cvr_0_01-0_35 --magnif --sal_sample --sample_temperature 6.0 --cover_ratio 0.01 0.35  --fov_size 20  --K  20  --sampling_bdr 16
--run_label proj256_eval_magnif_salmap_T30_0_cvr_0_01-0_35 --magnif --sal_sample --sample_temperature 30.0 --cover_ratio 0.01 0.35  --fov_size 20  --K  20  --sampling_bdr 16
--run_label proj256_eval_magnif_salmap_T100_0_cvr_0_01-0_35 --magnif --sal_sample --sample_temperature 100.0 --cover_ratio 0.01 0.35  --fov_size 20  --K  20  --sampling_bdr 16
--run_label proj256_eval_magnif_salmap_T0_03_cvr_0_01-0_35 --magnif --sal_sample --sample_temperature 0.03 --cover_ratio 0.01 0.35  --fov_size 20  --K  20  --sampling_bdr 16
--run_label proj256_eval_magnif_salmap_flat_cvr_0_01-0_35 --magnif --sal_sample --sal_control --cover_ratio 0.01 0.35  --fov_size 20  --K  20  --sampling_bdr 16
'

export extra_param="$(echo "$param_list" | head -n $LSB_JOBINDEX | tail -1)"
echo "$extra_param"

cd ~/SimCLR-torch/
python run_magnif.py -data $SCRATCH1/Datasets -dataset-name stl10 --workers 16 \
	--ckpt_every_n_epocs 5 --epochs 100  --batch-size 256  --out_dim 256  $extra_param
# don't break line without line breaker in bash! 