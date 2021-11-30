"""Read in event file written by tensorboard and perform post hoc comparison. """
import os
from os.path import join
import yaml
from glob import glob
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
pd.set_option('display.width', 200)
pd.set_option("max_colwidth", 60)
pd.set_option('display.max_columns', None)
import matplotlib.pylab as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
rootdir = Path(r"E:\Cluster_Backup\SimCLR-runs")
figdir = r"E:\OneDrive - Harvard University\SVRHM2021\Figures"
outdir = 'E:\\OneDrive - Harvard University\\SVRHM2021\\Tables'
SIMCLR_LEN = 390
EVAL_LEN = 22
INVALIDTIME = -99999
def split_record(timestep, value, standardL=EVAL_LEN):
    """ Split events of different threads in a same record file """
    Tnum = len(timestep)
    threadN = np.ceil(Tnum / standardL,).astype(np.int)
    thread_mask = []
    cnt_per_thread = [0 for _ in range(threadN)]
    thread_pointer = 0
    last_event = None
    for i, T in enumerate(timestep):
        if T == last_event:
            thread_pointer += 1
        else:
            thread_pointer = 0
        thread_mask.append(thread_pointer)
        cnt_per_thread[thread_pointer] += 1
        last_event = T

    assert len(thread_mask) == len(timestep) == len(value)
    timestep = np.array(timestep)
    value = np.array(value)
    thread_mask = np.array(thread_mask)
    time_threads = [timestep[thread_mask==i]  for  i  in  range(threadN)]
    val_threads = [value[thread_mask==i]  for  i  in  range(threadN)]
    return [np.concatenate((time_arr,
                    INVALIDTIME * np.ones(standardL-len(time_arr), dtype=time_arr.dtype)))
                    for  time_arr  in  time_threads], \
           [np.concatenate((val_arr,
                    np.nan * np.ones(standardL-len(val_arr), dtype=val_arr.dtype)))
                    for  val_arr  in  val_threads]

# keys = ["cover_ratio"]
def load_format_exps(expdirs, cfgkeys=["cover_ratio"],
                     eval_idx=-2, train_idx=-2, ema_alpha=0.6):
    train_acc_col = [] # 22
    test_acc_col = []  # 22
    simclr_acc_col = [] # 390
    param_list = []
    expnm_list = []
    for ei, expdir in enumerate(expdirs):
        expfp = rootdir/expdir
        fns = glob(str(expfp/"events.out.tfevents.*"))
        assert len(fns) == 1, ("%s folder has %d event files, split them" % (expfp, len(fns)))
        event_acc = EventAccumulator(str(expfp))
        event_acc.Reload()
        cfgargs = yaml.load(open(expfp / "config.yml", 'r'), Loader=yaml.Loader)
        # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
        _, eval_step_test, test_acc_val = zip(*event_acc.Scalars('eval/test_acc'))
        _, eval_step_train, train_acc_val = zip(*event_acc.Scalars('eval/train_acc'))
        _, train_step_nums, simclr_acc_val = zip(*event_acc.Scalars('acc/top1'))
        # step2epc_map = {step: epc for _, step, epc in event_acc.Scalars('epoch')}
        # step2epc_map[-1] = -1
        # Split threads of the same experiments
        eval_step_test_thrs, test_acc_val_thrs = split_record(eval_step_test, test_acc_val, EVAL_LEN)
        eval_step_train_thrs, train_acc_val_thrs = split_record(eval_step_train, train_acc_val, EVAL_LEN)
        train_step_nums_thrs, simclr_acc_val_thrs = split_record(train_step_nums, simclr_acc_val, SIMCLR_LEN)
        thread_num = len(eval_step_test_thrs)
        assert len(test_acc_val_thrs) == len(train_acc_val_thrs) == len(simclr_acc_val_thrs)
        train_acc_col.extend(train_acc_val_thrs)
        test_acc_col.extend(test_acc_val_thrs)
        simclr_acc_col.extend(simclr_acc_val_thrs)
        try:
            cfgdict = {k:cfgargs.__getattribute__(k) for k in cfgkeys}
        except AttributeError:
            print("Keys should be from this list:\n", list(cfgargs.__dict__.keys()))
            print(cfgargs.__dict__)
            raise AttributeError
        param_list.extend([cfgdict] * thread_num)
        expnm_list.extend([expdir] * thread_num)

    eval_timestep = np.array([-1, *range(1,100,5), 100])
    simclr_timestep = np.array([*range(0,39000,100)])
    assert(len(eval_timestep) == EVAL_LEN)
    assert(len(simclr_timestep) == SIMCLR_LEN)
    train_acc_arr = np.array(train_acc_col)
    test_acc_arr = np.array(test_acc_col)
    simclr_acc_arr = np.array(simclr_acc_col)
    simclr_acc_arr_ema = np.array([ema_vectorized(simclr_vec, alpha=ema_alpha)
                                   for simclr_vec in simclr_acc_col])
    param_table = pd.DataFrame(param_list)
    param_table["expdir"] = expnm_list
    param_table["train_acc"] = train_acc_arr[:, eval_idx]
    param_table["test_acc"] = test_acc_arr[:, eval_idx]
    param_table["simclr_acc"] = simclr_acc_arr[:, train_idx]
    param_table["simclr_acc_ema"] = simclr_acc_arr_ema[:, train_idx]
    print("Report linear evaluation acc at %d epc[%d]\n Report simclr acc at %d step[%d]"%
          (eval_timestep[eval_idx], eval_idx, simclr_timestep[train_idx], train_idx))
    print(param_table.drop("expdir", axis=1))

    return  train_acc_arr, test_acc_arr, simclr_acc_arr, \
            eval_timestep, simclr_timestep, param_table

from PIL import Image
def vis_exist_samples(expdir):
    img = plt.imread(rootdir/expdir/"sample_data_augs.png")
    Image.fromarray((255*img).astype(np.uint8)).show()


def ema_vectorized(data, alpha=0.6):
    """https://newbedev.com/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm-mean"""
    if type(data) is list:
        data = np.array(data)
    # alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha

    scale = 1/alpha_rev
    n = data.shape[0]

    r = np.arange(n)
    scale_arr = scale**r
    offset = data[0]*alpha_rev**(r+1)
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def bootstrap_diff_mean(group1,group2,q=[0.025,0.975],trials=1000):
    N1, N2 = len(group1), len(group2)
    samps1 = np.array([np.random.choice(group1, N1, replace=True) for _ in range(trials)])
    samps2 = np.array([np.random.choice(group2, N1, replace=True) for _ in range(trials)])
    dom_arr = samps1.mean(axis=1) - samps2.mean(axis=1)
    qtls = np.quantile(dom_arr, q)
    return dom_arr.mean(), qtls

#%%
runnms = os.listdir(rootdir)

#%% Experiment 1: Evaluate the Pure foveation transform
runnms = os.listdir(rootdir)
exp_fov_expdirs = [*filter(lambda nm:"proj256_eval_fov_" in nm and "Oct06_03-" not in nm, runnms)]
train_acc_arr, test_acc_arr, simclr_acc_arr, eval_timestep, simclr_timestep, \
    param_table = load_format_exps(exp_fov_expdirs, cfgkeys=["foveation", "disable_crop", "sal_control", "fov_area_rng", "kerW_coef", "blur", "pad_img", "crop_temperature"])
param_table.to_csv(join(outdir, "Exp1_pure_fov_vs_crop_result.csv"))
#%%
plt.figure(figsize=[6,5])
plt.plot(simclr_timestep/390, simclr_acc_arr.T, alpha=0.6)
plt.show()
#%%
explabels = param_table.expdir.to_list()
explabels = [lab.split("_Oct")[0].split("proj256_eval_fov_")[1] for lab in explabels]
#%%
simclr_acc_arr_ema = np.array([ema_vectorized(simclr_vec, alpha=0.4)
                                   for simclr_vec in simclr_acc_arr])
figh = plt.figure(figsize=[5, 5])
plt.plot(simclr_timestep/390, simclr_acc_arr_ema.T, alpha=0.5)
plt.ylabel("simclr accuracy")
plt.xlabel("epochs")
plt.legend(explabels)
figh.savefig(join(figdir, "train_curve", "foveblur_simclr_curve.png"))
figh.savefig(join(figdir, "train_curve", "foveblur_simclr_curve.pdf"))
plt.show()
#%%
figh = plt.figure(figsize=[5, 5])
plt.plot(eval_timestep, train_acc_arr.T, alpha=0.5)
plt.plot(eval_timestep, test_acc_arr.T, alpha=0.5, linestyle="-.")
plt.ylabel("eval accuracy")
plt.xlabel("epochs")
plt.legend(explabels+explabels)
figh.savefig(join(figdir, "train_curve", "foveblur_eval_curve.png"))
figh.savefig(join(figdir, "train_curve", "foveblur_eval_curve.pdf"))
plt.show()
#%% Experiment 2: Evaluate Magnification transfomr (Quad)
runnms = os.listdir(rootdir)
exp_magnif_expdirs = [*filter(lambda nm:"proj256_eval_magnif_cvr_" in nm or "proj256_eval_magnif_bsl_" in nm, runnms)]
train_acc_arr, test_acc_arr, simclr_acc_arr, eval_timestep, simclr_timestep, \
    param_table = load_format_exps(exp_magnif_expdirs, cfgkeys=["magnif", "crop", "blur", "cover_ratio", "fov_size", "K", "seed"])
param_table.to_csv(join(outdir, "Exp2_pure_magnif_vs_crop_result.csv"))

#%% compare baseline with magnification
magnif_msk = param_table.expdir.str.contains("proj256_eval_magnif_cvr_0_05-0_35_Oct")
bsl_msk = param_table.expdir.str.contains("proj256_eval_magnif_bsl_Oct")
for gname, msk in zip(["Magnificat", "Baseline"], [magnif_msk, bsl_msk]):
    print(f"{gname}\ttrain_acc {100*param_table.train_acc[msk].mean():.2f}+-{100*param_table.train_acc[msk].std():.2f}\t"
          f"test_acc {100*param_table.test_acc[msk].mean():.2f}+-{100*param_table.test_acc[msk].std():.2f}\t"
          f"simclr_acc {param_table.simclr_acc[msk].mean():.2f}+-{param_table.simclr_acc[msk].std():.2f}")

bootstrap_diff_mean(100*param_table.test_acc[magnif_msk],100*param_table.test_acc[bsl_msk])
#%%
plt.figure(figsize=[10,10])
plt.plot(test_acc_arr.T,)
plt.legend(param_table.expdir.to_list())
plt.show()







#%% Experiment 3: Temperature effect on Random Crops
exp_Tcrop_expdirs = [*filter(lambda nm:("proj256_eval_sal_new_T" in nm)
       and ("Oct08" in nm or "Oct09" in nm), runnms)]
train_acc_arr, test_acc_arr, simclr_acc_arr, eval_timestep, simclr_timestep, \
    param_table = load_format_exps(exp_Tcrop_expdirs, cfgkeys=["crop_temperature", "sal_control", ])#"blur"
param_table = param_table.sort_values("crop_temperature")
param_table.reset_index(inplace=True)
param_table.to_csv(join(outdir, "Exp3_salmap_randcrop_sampling.csv"))

exp_Tcrop_bsl_expdirs = [*filter(lambda nm:"proj256_eval_sal_new_flat" in nm and ("Oct08" in nm or "Oct09" in nm), runnms)]
_, _, _, _, _, \
    param_table_bsl = load_format_exps(exp_Tcrop_bsl_expdirs, cfgkeys=["crop_temperature", "sal_control", ])
param_table_bsl.to_csv(join(outdir, "Exp3_salmap_randcrop_sampling_baseline.csv"))

#%%
figh = plt.figure(figsize=[4,5])
plt.plot(param_table.crop_temperature, param_table.train_acc, marker="o", label="train_acc")
plt.plot(param_table.crop_temperature, param_table.test_acc, marker="o", label="test_acc")
plt.semilogx()
plt.hlines(param_table_bsl.train_acc.mean(), 0, 100, color="darkblue", linestyles=":", label="Train (Uniform Sampling)")
plt.hlines(param_table_bsl.test_acc.mean(), 0, 100, color="red", linestyles=":", label="Test (Uniform Sampling)")
plt.xlabel("Sampling Temperature")
plt.ylabel("Linear Eval Accuracy")
plt.title("Visual Repr Evaluation\nEpoch %d"%eval_timestep[-2])
plt.legend()
plt.show()
figh.savefig(join(figdir, "RandCrop_evalAcc-temperature_curve.png"))
figh.savefig(join(figdir, "RandCrop_evalAcc-temperature_curve.pdf"))
# sort_table = param_table.sort_values("sample_temperature")

#%%
figh = plt.figure(figsize=[4,5])
plt.plot(param_table.crop_temperature, param_table.simclr_acc_ema, marker="o")
plt.semilogx()
plt.hlines(param_table_bsl.simclr_acc_ema.mean(), 0, 100, color="darkblue", linestyles=":", label="Test (Uniform Sampling)")
plt.xlabel("Sampling Temperature")
plt.ylabel("Simclr Train Accuracy")
plt.title("Training Objective Accuracy\nStep %d Epoch 99"%simclr_timestep[-2])
plt.show()
figh.savefig(join(figdir, "RandCrop_simclrAcc-temperature_curve.png"))
figh.savefig(join(figdir, "RandCrop_simclrAcc-temperature_curve.pdf"))







#%% Experiment 4: Magnification with saliency maps: Effect of temperature
runnms = os.listdir(rootdir)
exp_magnif_T_expdirs = [*filter(lambda nm:"proj256_eval_magnif_salmap" in nm and "flat" not in nm, runnms)]
train_acc_arr, test_acc_arr, simclr_acc_arr, eval_timestep, simclr_timestep, \
    param_table = load_format_exps(exp_magnif_T_expdirs, cfgkeys=["cover_ratio", "sample_temperature", "blur"])
param_table = param_table.sort_values("sample_temperature")
param_table.reset_index(inplace=True)
param_table.to_csv(join(outdir, "Exp4_salmap_magnif_sampling.csv"))
#%%
exp_magnif_bsl_expdirs = [*filter(lambda nm:"proj256_eval_magnif_salmap_flat_cvr_0_01-0_35" in nm, runnms)]
_, _, _, _, _, param_table_bsl = load_format_exps(exp_magnif_bsl_expdirs, cfgkeys=["cover_ratio", "sal_sample", "blur"])
param_table_bsl.to_csv(join(outdir, "Exp4_salmap_magnif_sampling_baseline.csv"))

#%%
figh = plt.figure(figsize=[3.5,3.])
plt.plot(param_table.sample_temperature, param_table.train_acc, marker="o", label="train_acc")
plt.plot(param_table.sample_temperature, param_table.test_acc, marker="o", label="test_acc")
plt.semilogx()
plt.hlines(param_table_bsl.train_acc.mean(), 0, 100, color="darkblue", linestyles=":", label="Train (Uniform Sampling)")
plt.hlines(param_table_bsl.test_acc.mean(), 0, 100, color="red", linestyles=":", label="Test (Uniform Sampling)")
plt.xlabel("Sampling Temperature")
plt.ylabel("Linear Eval Accuracy")
plt.title("Visual Repr Evaluation\nEpoch %d"%eval_timestep[-2])
plt.legend()
plt.show()
figh.savefig(join(figdir, "Magnif_evalAcc-temperature_curve_2.png"))
figh.savefig(join(figdir, "Magnif_evalAcc-temperature_curve_2.pdf"))
# sort_table = param_table.sort_values("sample_temperature")

#%%
figh = plt.figure(figsize=[3.5,3.])
plt.plot(param_table.sample_temperature, param_table.simclr_acc_ema, marker="o")
plt.semilogx()
plt.hlines(param_table_bsl.simclr_acc_ema.mean(), 0, 100, color="darkblue", linestyles=":", label="Test (Uniform Sampling)")
plt.xlabel("Sampling Temperature")
plt.ylabel("Simclr Train Accuracy")
plt.title("Training Objective Accuracy\nStep %d Epoch 99"%simclr_timestep[-2])
plt.show()
figh.savefig(join(figdir, "Magnif_simclrAcc-temperature_curve_2.png"))
figh.savefig(join(figdir, "Magnif_simclrAcc-temperature_curve_2.pdf"))




#%%


#%%




#%% Comparison of Magnif models
quad_magnif_expdirs = ["proj256_eval_magnif_cvr_0_01-1_50_Oct07_05-06-53",
                     "proj256_eval_magnif_cvr_0_01-1_50_Oct07_19-46-40",
                     "proj256_eval_magnif_cvr_0_05-0_70_Oct07_05-11-35",
                     "proj256_eval_magnif_cvr_0_05-0_70_Oct07_19-46-29",
                     "proj256_eval_magnif_cvr_0_05-0_35_Oct07_05-06-55",
                     "proj256_eval_magnif_cvr_0_05-0_35_Oct07_19-46-40",
                     "proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-24",
                     "proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-24-SPLIT",
                     "proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-26",
                     "proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-31",
                     "proj256_eval_magnif_cvr_0_01-0_35_Oct07_19-46-40",
                     "proj256_eval_magnif_cvr_0_01-0_35_Oct07_05-06-57",]

train_acc_arr, test_acc_arr, simclr_acc_arr, eval_timestep, simclr_timestep, \
    param_table = load_format_exps(quad_magnif_expdirs, cfgkeys=["cover_ratio"])
#%%
for ei in range(param_table.shape[0]):
    print("cover_ratio [%.2f, %.2f] trainACC %.4f  testACC %.4f  simclrACC %.4f"%(*param_table.cover_ratio[ei], train_acc_arr[ei,-2], test_acc_arr[ei,-2], simclr_acc_arr[ei,-2]))
#%% Temperature
Tcrop_expdirs = ["proj256_eval_sal_new_T0.01_Oct09_00-57-35",
                "proj256_eval_sal_new_T0.1_Oct09_00-57-38",
                "proj256_eval_sal_new_T0.3_Oct09_00-57-38",
                "proj256_eval_sal_new_T0.7_Oct08_08-44-40",
                "proj256_eval_sal_new_T1.0_Oct09_00-57-39",
                "proj256_eval_sal_new_T1.5_Oct08_08-44-40",
                "proj256_eval_sal_new_T2.5_Oct08_08-44-40",
                "proj256_eval_sal_new_T3.0_Oct09_00-57-34",
                "proj256_eval_sal_new_T4.5_Oct08_08-49-50",
                "proj256_eval_sal_new_T10.0_Oct09_00-57-38",
                "proj256_eval_sal_new_T30.0_Oct09_00-57-34",
                "proj256_eval_sal_new_T100.0_Oct09_00-58-34",
                "proj256_eval_sal_new_flat_Oct09_00-57-33",]

train_acc_arr, test_acc_arr, simclr_acc_arr, eval_timestep, simclr_timestep, \
    param_table = load_format_exps(Tcrop_expdirs, cfgkeys=["crop_temperature", "sal_control"])

for ei in range(param_table.shape[0]):
    print("crop_temperature %.1f %s trainACC %.4f  testACC %.4f  simclrACC %.4f"%
        (param_table.crop_temperature[ei], "Control" if param_table.sal_control[ei] else "", train_acc_arr[ei,-2], test_acc_arr[ei,-2], simclr_acc_arr[ei,-2]))

#%% Visualize temperature effect on training
T_arr = param_table.crop_temperature
epoc_id = -2
figh = plt.figure(figsize=(4, 5))
plt.plot(T_arr[:-1], train_acc_arr[:-1, epoc_id], label="Train Set", marker="o")
plt.plot(T_arr[:-1], test_acc_arr[:-1, epoc_id], label="Test Set", marker="o")
plt.hlines(train_acc_arr[-1, epoc_id], 0, 100, color="darkblue", linestyles=":",
           label="Train (Uniform Sampling)")
plt.hlines(test_acc_arr[-1, epoc_id], 0, 100, color="red", linestyles=":",
           label="Test (Uniform Sampling)")
plt.semilogx()
plt.xlim([0, 100])
plt.legend()
plt.xlabel("Sampling Temperature")
plt.ylabel("Linear Eval Accuracy")
plt.title("Visual Repr Evaluation\nEpoch %d"%eval_timestep[epoc_id])
plt.show()
figh.savefig(join(figdir, "randcrop_evalAcc-temperature_curve.png"))
figh.savefig(join(figdir, "randcrop_evalAcc-temperature_curve.pdf"))
#%%
step_id = -2
figh2 = plt.figure(figsize=(4,5))
plt.plot(T_arr[:-1], simclr_acc_arr[:-1, step_id,], label="Simclr Acc", marker="o")
plt.hlines(simclr_acc_arr[-1, step_id], 0, 100,color="darkblue", linestyles=":",
           label="Simclr Acc (Uniform Sampling)")
plt.semilogx()
plt.xlim([0, 100])
plt.legend()
plt.xlabel("Sampling Temperature")
plt.ylabel("Unlabeled Set Simclr Accuracy")
plt.title("Training Objective Accuracy\nStep %d Epoch 99"%simclr_timestep[step_id])
plt.show()
figh2.savefig(join(figdir, "randcrop_simclrAcc-temperature_curve.png"))
figh2.savefig(join(figdir, "randcrop_simclrAcc-temperature_curve.pdf"))


#%% Comparison of using foveation vs crop
foveacrop_expdirs = [# "proj256_eval_fov_orig_crop_Oct06_03-29-24",#no cfg
                    "proj256_eval_fov_orig_crop_Oct06_17-56-31",
                    "proj256_eval_fov_sal_ctrl_Oct06_17-56-29",
                    # "proj256_eval_fov_sal_exp_Oct06_03-30-35",
                    "proj256_eval_fov_sal_exp_Oct06_17-56-30",
                    "proj256_eval_fov_fvr0_01-0_5_slp006_Oct06_17-58-16",
                    "proj256_eval_fov_fvr0_10-0_5_slp006_Oct06_17-58-17",
                    # "proj256_eval_fov_nocrop_blur_Oct06_03-29-24",
                    "proj256_eval_fov_nocrop_blur_Oct06_17-56-28",
                    # "proj256_eval_fov_nocrop_fvr0_01-0_1_slp006_Oct06_03-29-24",
                    # "proj256_eval_fov_nocrop_fvr0_10-0_5_slp006_Oct06_03-30-31",
                    "proj256_eval_fov_nocrop_fvr0_01-0_1_slp006_Oct06_17-56-29",
                    "proj256_eval_fov_nocrop_fvr0_10-0_5_slp006_Oct06_18-17-43",
                    "proj256_eval_fov_nocrop_fvr0_01-0_5_slp006_Oct06_17-58-17",]

train_acc_arr, test_acc_arr, simclr_acc_arr, eval_timestep, simclr_timestep, \
    param_table = load_format_exps(foveacrop_expdirs, cfgkeys=["disable_crop", "blur", "orig_cropper", "sal_control"])
#%
for ei in range(param_table.shape[0]):
    explabel = param_table.index[ei].split("_Oct")[0]
    print("%s:\t%s %s %s %s trainACC %.4f  testACC %.4f  simclrACC %.4f"%
        (explabel, "no crop" if param_table.disable_crop[ei] else "do crop",
         "Blur" if param_table.blur[ei] else "",
         "orig_cropper" if param_table.orig_cropper[ei] else "",
         "Control" if param_table.sal_control[ei] else "",
         train_acc_arr[ei,-2], test_acc_arr[ei,-2], simclr_acc_arr[ei,-2]))

#%%
runnms = os.listdir(rootdir)
exp_magnif_expdirs = [*filter(lambda nm:"proj256_eval_magnif_exp" in nm, runnms)]
train_acc_arr, test_acc_arr, simclr_acc_arr, eval_timestep, simclr_timestep, \
    param_table = load_format_exps(exp_magnif_expdirs, cfgkeys=["cover_ratio", "slope_C", "crop", "blur"])
#%%
for ei in range(param_table.shape[0]):
    print("cover_ratio %s slope_C %s trainACC %.4f  testACC %.4f  simclrACC %.4f"%\
          (param_table.cover_ratio[ei], param_table.slope_C[ei], train_acc_arr[ei,-2], test_acc_arr[ei,-2], simclr_acc_arr[ei,-2]))
#%%
exp_line = param_table.iloc[1]
vis_exist_samples(exp_line.expdir)
print(exp_line)



#%%  Effect of  Randomized seeds
runnms = os.listdir(rootdir)
exp_magnif_expdirs = [*filter(lambda nm:"RND" in nm, runnms)]
train_acc_arr, test_acc_arr, simclr_acc_arr, eval_timestep, simclr_timestep, \
    param_table = load_format_exps(exp_magnif_expdirs, cfgkeys=["cover_ratio", "crop", "gridfunc_form"])

#%% Magnification parameters

#%% Quad Magnification experiments
exp_magnif_expdirs = [*filter(lambda nm:"proj256_eval_magnif_cvr_" in nm, runnms)]
train_acc_arr, test_acc_arr, simclr_acc_arr, eval_timestep, simclr_timestep, \
    param_table = load_format_exps(exp_magnif_expdirs, cfgkeys=["cover_ratio", "crop", "fov_size", "K", "seed"])

#%%
expdir_col = ["proj256_eval_sal_new_T0.01_Oct06_19-02-51",
        "proj256_eval_sal_new_T0.1_Oct06_19-02-51",
        "proj256_eval_sal_new_T0.3_Oct06_19-10-25",
        "proj256_eval_sal_new_T0.7_Oct08_08-44-40",
        "proj256_eval_sal_new_T1.0_Oct06_19-02-50",
        "proj256_eval_sal_new_T1.5_Oct08_08-44-40",
        "proj256_eval_sal_new_T2.5_Oct08_08-44-40",
        "proj256_eval_sal_new_T3.0_Oct06_19-10-25",
        "proj256_eval_sal_new_T4.5_Oct08_08-49-50",
        "proj256_eval_sal_new_T10.0_Oct06_19-10-25",
        "proj256_eval_sal_new_T30.0_Oct06_19-10-25",
        "proj256_eval_sal_new_T100.0_Oct06_19-05-06",
        "proj256_eval_sal_new_flat_Oct06_19-02-51", # use the flat saliency map as substitute.keep the sampling mechanism.
        ]

T_arr = []
train_acc_arr = np.ones((22, len(expdir_col),)) * np.nan
test_acc_arr = np.ones((22, len(expdir_col),)) * np.nan
simclr_acc_arr = np.ones((392, len(expdir_col),)) * np.nan
for ei, expdir in enumerate(expdir_col):
    expfp = rootdir/expdir
    fns = glob(str(expfp/"events.out.tfevents.*"))
    assert len(fns) == 1
    event_acc = EventAccumulator(str(expfp))
    event_acc.Reload()
    step2epc_map = {step: epc for _, step, epc in event_acc.Scalars('epoch')}
    step2epc_map[-1] = -1
    # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    _, eval_step_nums, test_acc_val = zip(*event_acc.Scalars('eval/test_acc'))
    _, eval_step_nums, train_acc_val = zip(*event_acc.Scalars('eval/train_acc'))
    _, train_step_nums, simclr_acc_val = zip(*event_acc.Scalars('acc/top1'))
    epocs = np.array(eval_step_nums)//390
    cfgargs = yaml.load(open(expfp / "config.yml", 'r'), Loader=yaml.Loader)
    temperature = cfgargs.crop_temperature
    sal_control = cfgargs.sal_control
    T_arr.append(temperature)
    train_acc_arr[:len(train_acc_val), ei] = np.array(train_acc_val)
    test_acc_arr[:len(test_acc_val), ei] = np.array(test_acc_val)
    simclr_acc_arr[:len(simclr_acc_val), ei] = np.array(simclr_acc_val)

#%%
import matplotlib.pylab as plt
epoc_id = 1
figh = plt.figure(figsize=(4,5))
plt.plot(T_arr[:-1], train_acc_arr[epoc_id,:-1], label="Train Set")
plt.plot(T_arr[:-1], test_acc_arr[epoc_id,:-1], label="Test Set")
plt.hlines(train_acc_arr[epoc_id, -1],0,100,color="darkblue",linestyles=":",
           label="Train (Uniform Sampling)")
plt.hlines(test_acc_arr[epoc_id, -1],0,100,color="red",linestyles=":",
           label="Test (Uniform Sampling)")
plt.semilogx()
plt.xlim([0,100])
plt.legend()
plt.xlabel("Sampling Temperature")
plt.ylabel("Linear Eval Accuracy")
plt.title("Visual Repr Evaluation")
plt.show()
#%%
step_id = -20
figh = plt.figure(figsize=(4,5))
plt.plot(T_arr[:-1], simclr_acc_arr[step_id,:-1], label="Simclr Acc")
plt.hlines(simclr_acc_arr[step_id, -1],0,100,color="darkblue",linestyles=":",
           label="Simclr Acc (Uniform Sampling)")
plt.semilogx()
plt.xlim([0,100])
plt.legend()
plt.xlabel("Sampling Temperature")
plt.ylabel("Unlabeled Set Simclr Accuracy")
plt.show()
# Show all tags in the log file
# print(event_acc.Tags())
# 'scalars': ['eval/train_loss', 'eval/train_acc', 'eval/test_loss', 'eval/test_acc', 'epoch', 'loss', 'acc/top1', 'acc/top5', 'learning_rate'],
#%%
"proj256_eval_magnif_bsl_Oct07_05-11-35"
"proj256_eval_magnif_bsl_Oct07_19-46-29"
"proj256_eval_magnif_cvr_0_01-1_50_Oct07_05-06-53"
"proj256_eval_magnif_cvr_0_05-0_35_Oct07_05-06-55"
"proj256_eval_magnif_cvr_0_01-0_35_Oct07_05-06-57"
"proj256_eval_magnif_cvr_0_05-0_70_Oct07_05-11-35"
"proj256_eval_magnif_cvr_0_05-0_70_Oct07_19-46-29"
"proj256_eval_magnif_cvr_0_01-0_35_Oct07_19-46-40"
"proj256_eval_magnif_cvr_0_05-0_35_Oct07_19-46-40"
"proj256_eval_magnif_cvr_0_01-1_50_Oct07_19-46-40"
"proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-24"
"proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-26"
"proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-31"
"proj256_eval_magnif_exp_cvr_0_05-1_00_slp_1_50_Oct07_19-21-47"
"proj256_eval_magnif_exp_cvr_0_05-1_00_slp_0_75-3_00_Oct07_19-38-20"
"proj256_eval_magnif_exp_cvr_0_05-1_00_slp_0_75-3_00_Oct07_19-22-11"
"proj256_eval_magnif_exp_cvr_0_05-0_50_slp_1_50_Oct07_19-25-55"
"proj256_eval_magnif_exp_cvr_0_05-0_50_slp_0_75-3_00_Oct07_19-22-28"

#%% Baseline Distribution Fixed Seed
expdirs = ["proj256_eval_magnif_bsl_Oct07_05-11-35",
        "proj256_eval_magnif_bsl_Oct07_19-46-29",
        "proj256_eval_magnif_bsl_Oct08_07-44-26",
        "proj256_eval_magnif_bsl_Oct08_07-44-27",
        "proj256_eval_magnif_bsl_Oct08_07-44-30",
        "proj256_eval_magnif_bsl_Oct08_07-45-39",
        "proj256_eval_magnif_bsl_Oct08_07-45-41",
        "proj256_eval_magnif_bsl_Oct08_07-45-43",
        "proj256_eval_magnif_bsl_Oct08_07-45-48",
        ]

#%% 
expdirs = [
"proj256_eval_magnif_salmap_T0_3_cvr_0_01-0_35_Oct08_07-19-53",
"proj256_eval_magnif_salmap_T0_7_cvr_0_01-0_35_Oct08_07-18-58",
"proj256_eval_magnif_salmap_T1_0_cvr_0_01-0_35_Oct08_07-18-59",
"proj256_eval_magnif_salmap_T1_5_cvr_0_01-0_35_Oct08_07-18-59",
"proj256_eval_magnif_salmap_T10_0_cvr_0_01-0_35_Oct08_07-17-52",
]

#%% baseline experiments
bslexpdirs = ["proj256_eval_magnif_bsl_Oct07_05-11-35",
            "proj256_eval_magnif_bsl_Oct07_19-46-29",
            "proj256_eval_magnif_bsl_Oct08_07-44-26",
            "proj256_eval_magnif_bsl_Oct08_07-44-27",
            "proj256_eval_magnif_bsl_Oct08_07-44-30",
            "proj256_eval_magnif_bsl_Oct08_07-45-39",
            "proj256_eval_magnif_bsl_Oct08_07-45-41",
            "proj256_eval_magnif_bsl_Oct08_07-45-43",
            "proj256_eval_magnif_bsl_Oct08_07-45-48",]

train_acc_arr_bsl, test_acc_arr_bsl, simclr_acc_arr_bsl, eval_timestep, simclr_timestep, \
    param_table_bsl = load_format_exps(bslexpdirs, cfgkeys=["cover_ratio", "crop"])

for ei in range(param_table_bsl.shape[0]):
    print("crop %s trainACC %.4f  testACC %.4f  simclrACC %.4f"%\
          (param_table_bsl.crop[ei], train_acc_arr_bsl[ei,-2], test_acc_arr_bsl[ei,-2], simclr_acc_arr_bsl[ei,-2]))
#%%
