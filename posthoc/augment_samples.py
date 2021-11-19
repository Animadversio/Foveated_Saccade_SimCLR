
import os
from os.path import join
import yaml
from glob import glob
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pylab as plt
from PIL import Image
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
pd.set_option('display.width', 200)
pd.set_option("max_colwidth", 60)
pd.set_option('display.max_columns', None)
rootdir = Path(r"E:\Cluster_Backup\SimCLR-runs")
figdir = r"E:\OneDrive - Harvard University\SVRHM2021\Figures"
outdir = 'E:\\OneDrive - Harvard University\\SVRHM2021\\Tables'

def np2pil(img):
    return Image.fromarray((255 * img).astype(np.uint8))

def pil_showimg(img):
    Image.fromarray((255 * img).astype(np.uint8)).show()
    
def vis_exist_samples(expdir, show=True):
    img = plt.imread(rootdir/expdir/"sample_data_augs.png")
    if show: pil_showimg(img)
    return img
#%%
runnms = os.listdir(rootdir)
#%%
def make_grid_np(img_arr, nrow=8, padding=2, pad_value=0):
    if type(img_arr) is list:
        try:
            img_tsr = np.stack(tuple(img_arr), axis=3)
            img_arr = img_tsr
        except ValueError:
            raise ValueError("img_arr is a list and its elements do not have the same shape as each other.")
    nmaps = img_arr.shape[3]
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = int(img_arr.shape[0] + padding), int(img_arr.shape[1] + padding)
    grid = np.zeros((height * ymaps + padding, width * xmaps + padding, 3), dtype=img_arr.dtype)
    grid.fill(pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[y * height + padding: (y + 1) * height, x * width + padding: (x + 1) * width, :] = img_arr[:,:,:,k]
            k = k + 1
    return grid

def subsample_montage(mtgimg, rows=[1,2,5], cols="rand", colsamps=4, pad=4, imgsize=96):
    H,W,_ = mtgimg.shape
    nrows = H // imgsize
    ncols = W // imgsize

    def extract_tile(i, j):
        rbeg = i * (imgsize + pad) + pad
        cbeg = j * (imgsize + pad) + pad
        return mtgimg[rbeg:rbeg+imgsize, cbeg:cbeg+imgsize, :]

    tile_col = []
    for ri in rows:
        tile_row = []
        if cols == "rand":
            samps = np.random.choice(ncols, size=colsamps, replace=False)
            for ci in samps:
                tile_col.append(extract_tile(ri, ci))
        elif type(cols) in [list, tuple]:
            for ci in cols:
                tile_col.append(extract_tile(ri, ci))
    new_mtg = make_grid_np(tile_col, nrow=colsamps if cols == "rand" else len(cols), padding=2, pad_value=0)
    return tile_col, new_mtg


#%%
expnms = [*filter(lambda nm: "proj256_eval_fov_nocrop_fvr0_01-0_5" in nm, runnms)]
mtgimg = vis_exist_samples(expnms[0], False)
tiles, new_mtg = subsample_montage(mtgimg, rows=[5, 6, 7, 11], cols="rand", colsamps=4, pad=4)
pil_showimg(new_mtg)
np2pil(new_mtg).save(join(figdir, "augment_distrib", "pure_fov_dist_fvr0_01-0_5.png"))
#%%
expnms = ["proj256_eval_magnif_bsl_Oct08_07-45-48"]
mtgimg = vis_exist_samples(expnms[0], False)
tiles, new_mtg = subsample_montage(mtgimg, rows=[5, 6, 7, 11], cols="rand", colsamps=4, pad=4)
pil_showimg(new_mtg)
np2pil(new_mtg).save(join(figdir, "augment_distrib", "orig_crop_baseline_magnif.png"))
#%%
expnms = ["proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-24"]
mtgimg = vis_exist_samples(expnms[0], False)
tiles, new_mtg = subsample_montage(mtgimg, rows=[5, 6, 7, 11], cols="rand", colsamps=4, pad=4)
pil_showimg(new_mtg)
np2pil(new_mtg).save(join(figdir, "augment_distrib", "pure_magnif_dist_cvr_0_05-0_35-2.png"))
#%%
expnms = ["proj256_eval_sal_new_T0.7_Oct08_08-44-40"]
mtgimg = vis_exist_samples(expnms[0], False)
tiles, new_mtg = subsample_montage(mtgimg, rows=[5, 6, 7, 11], cols="rand", colsamps=4, pad=4)
pil_showimg(new_mtg)
np2pil(new_mtg).save(join(figdir, "augment_distrib", "crop_w_salmap_dist_T1.0.png"))
#%%
expnms = ["proj256_eval_magnif_salmap_T1_0_cvr_0_01-0_35_Oct08_07-18-59"]
mtgimg = vis_exist_samples(expnms[0], False)
tiles, new_mtg = subsample_montage(mtgimg, rows=[5, 6, 7, 11], cols="rand", colsamps=4, pad=4)
pil_showimg(new_mtg)
np2pil(new_mtg).save(join(figdir, "augment_distrib", "magnif_w_salmap_dist_T1.0.png"))
#%%
expnms = ["proj256_eval_magnif_salmap_T0_7_cvr_0_01-0_35_Oct08_07-18-58"]
mtgimg = vis_exist_samples(expnms[0], False)
tiles, new_mtg = subsample_montage(mtgimg, rows=[5, 6, 7, 11], cols="rand", colsamps=4, pad=4)
pil_showimg(new_mtg)
np2pil(new_mtg).save(join(figdir, "augment_distrib", "magnif_w_salmap_dist_T0.7.png"))
