# Foveated_Saccade_SimCLR
Official code base for the poster "On the use of Cortical Magnification and Saccades as Biological Proxies for Data Augmentation" published in NeurIPS 2021 Workshop Shared Visual Representations in Human and Machine Intelligence(SRVHM)

In this work, we 


The experimental pipeline is heavily based on the pytorch SimCLR implemented by [sthalles](https://github.com/sthalles/SimCLR) and by [Spijkervet](https://github.com/Animadversio/SimCLR-2). This new code base supports our biologically inspired data augmentations, visualization and *post hoc* data analysis. 

## Usage
For running a quick demo, replace the `$Datasets_path` with the parent folder of `stl10_binary` (e.g. `.\Datasets`), where you could download and extract from [here](https://cs.stanford.edu/~acoates/stl10/). 
```bash
python run_magnif.py -data $Datasets_path -dataset-name stl10 --workers 16 \
	--ckpt_every_n_epocs 5 --epochs 100  --batch-size 256  --out_dim 256  \
	--run_label proj256_eval_magnif_cvr_0_05-0_35 --magnif \
	--cover_ratio 0.05 0.35  --fov_size 20  --K  20  --sampling_bdr 16 
```

## Structure of Repo
* Main command line interface
	* `run.py` Running baseline training pipeline without bio-inspired augmentations. 
	* `run_salcrop.py` Running training pipeline with options for foveation transforms and saliency based sampling. 
	* `run_magnif.py` Running training pipeline with options for foveation transforms and saliency based sampling. 
* `data_aug\`, implementation of our bio-inspired augmentations
* `posthoc\`, analysis code for training result. 
* `scripts\`, scripts that run experiments on cluster. 


Inquiries: binxu_wang@hms.harvard.edu