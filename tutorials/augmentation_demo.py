"""Demo the two types of foveation for the schematics in the paper"""
import numpy as np
from os.path import join
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage, ToTensor
from torch.nn.functional import interpolate
from data_aug.aug_utils import send_to_clipboard
from data_aug.foveation import FoveateAt, randomFoveated, FoveateAt_demo
figdir = r"E:\OneDrive - Harvard University\SVRHM2021\Figures"
#%%
img = plt.imread("data_aug\\Gigi_resize.jpg")
img_tsr = torch.tensor(img / 255.0).permute([2, 0, 1]).unsqueeze(0).float()
img_tsr_rsz = interpolate(img_tsr, scale_factor=0.5)
# fov_img = FoveateAt(img_tsr, pnt=(200, 400), kerW_coef=0.04, e_o=1, \
#     spacing=0.5, demo=True)
#%%
views = randomFoveated(img_tsr_rsz, 9, bdr=16, kerW_coef=0.04,  spacing=0.2, fov_area_rng=0.05)
ToPILImage()(make_grid(views.squeeze(1), nrow=3)).show()
#%%
views = FoveateAt(img_tsr_rsz, (50, 120), kerW_coef=0.04,  spacing=0.2, e_o=0.6)
mtg = ToPILImage()(make_grid(views.squeeze(1), nrow=3))
mtg.show()
mtg.save(join(figdir,"cat_blur_pure.png"))
#%%
views = FoveateAt(img_tsr_rsz, (160, 70), kerW_coef=0.04,  spacing=0.2, e_o=0.05)
mtg = ToPILImage()(make_grid(views.squeeze(1), nrow=3))
mtg.show()
mtg.save(join(figdir,"cat_blur_pure2.png"))
#%%
finalimg, mask_col, blurimg_col, multiply_col = FoveateAt_demo(img_tsr_rsz,
                       (50, 120), kerW_coef=0.04, spacing=0.4, e_o=0.6)
#%
for i in range(len(mask_col)):
    plt.imsave(join(figdir, "foveation_demo\\mask_%02d.png"%i), mask_col[i], cmap="gray")
    ToPILImage()(blurimg_col[i]).save(join(figdir, "foveation_demo\\blurimg_%02d.png"%i))
    ToPILImage()(multiply_col[i]).save(join(figdir, "foveation_demo\\multimg_%02d.png"%i))

#%%
from data_aug.cort_magnif_tfm import img_cortical_magnif_tsr, radial_exp_isotrop_gridfun, radial_quad_isotrop_gridfun, \
            cortical_magnif_tsr_demo
#%%
fighm, img_cm, img = cortical_magnif_tsr_demo(img_tsr_rsz, (50, 120),
            lambda img, pnt: radial_quad_isotrop_gridfun(img,
                     pnt, fov=20, K=20, cover_ratio=0.35), subN=4)
fighm.savefig(join(figdir,"cat_magnif.png"))
fighm.savefig(join(figdir,"cat_magnif.pdf"))
ToPILImage()(img_cm).save(join(figdir,"cat_magnif_pure.png"))
#%%
fighm, img_cm, img = cortical_magnif_tsr_demo(img_tsr_rsz, (160, 70),
            lambda img, pnt: radial_quad_isotrop_gridfun(img,
                     pnt, fov=20, K=20, cover_ratio=0.20), subN=4)
fighm.savefig(join(figdir,"cat_magnif2.png"))
fighm.savefig(join(figdir,"cat_magnif2.pdf"))
ToPILImage()(img_cm).save(join(figdir,"cat_magnif_pure2.png"))

#%%
from data_aug.dataset_w_salmap import Contrastive_STL10_w_CortMagnif
train_dataset = Contrastive_STL10_w_CortMagnif(r"E:\Datasets")
#%%
from data_aug.visualize_aug_dataset import visualize_saliency_maps_w_imgs
figh, _ = visualize_saliency_maps_w_imgs(train_dataset, True, bdr=16,
                 idxs=[1973,], temperature=[0.1, 0.3, 0.7, 1, 1.5, 3, 6, 10], )
figh.savefig(join(figdir, "salmap_demo", "STl10_salmap_density_id01973.png"))
figh.savefig(join(figdir, "salmap_demo", "STl10_salmap_density_id01973.pdf"))
#%%
figh, _ = visualize_saliency_maps_w_imgs(train_dataset, True, bdr=16,
                 idxs=[58202,], temperature=[0.1, 0.3, 0.7, 1, 1.5, 3, 6, 10], )
figh.savefig(join(figdir, "salmap_demo", "STl10_salmap_density_id58202.png"))
figh.savefig(join(figdir, "salmap_demo", "STl10_salmap_density_id58202.pdf"))
#%%
figh, _ = visualize_saliency_maps_w_imgs(train_dataset, True, bdr=16,
                 idxs=[77493, 131, 58202, 29975,], temperature=[0.7, 1], )
figh.savefig(join(figdir, "augment_distrib", "STl10_salmap_density_id_77493_131_58202_29975.png"))
figh.savefig(join(figdir, "augment_distrib", "STl10_salmap_density_id_77493_131_58202_29975.pdf"))

