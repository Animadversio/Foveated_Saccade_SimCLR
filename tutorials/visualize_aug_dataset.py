""" Scripts to visualize augmentation distribution for the salmap dataset. """
import numpy as np
import matplotlib.pylab as plt
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.transforms import RandomResizedCrop
from data_aug.dataset_w_salmap import Contrastive_STL10_w_salmap
from data_aug.saliency_random_cropper import RandomResizedCrop_with_Density, RandomCrop_with_Density

def visualize_saliency_maps_w_imgs(train_dataset, plot_density=False, temperature=3, bdr=16,
    idxs=(96659, 54019, 88327, 81148, 98469, 77493, 131, 58202, 66666, 65017, 1973, 29975),
    ):
    idx_col = [] if idxs is None else idxs
    if type(temperature) in [tuple, list]:
        nTemp = len(temperature)
    else:
        nTemp = 1
        temperature = [temperature]
    ncols = 2 + plot_density * nTemp
    nrows = min(len(idx_col), 12)
    figh, axs = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.6), squeeze=False)
    for i in range(nrows):
        if idxs is None:
            idx = np.random.randint(1E5)
            idx_col.append(idx)
        else:
            idx = idxs[i]

        img, label = train_dataset.dataset.__getitem__(idx)  # img is PIL.Image, label is xxxx
        salmap = train_dataset.salmaps[idx, :, :, :].astype('float')
        axs[i, 0].imshow(np.array(img))
        axs[i, 0].axis("off")
        axs[i, 1].imshow(salmap.squeeze())
        axs[i, 1].axis("off")
        if plot_density:
            for k, Temp in enumerate(temperature):
                density = np.exp((salmap - salmap.max()) / Temp)
                if bdr > 0:  # set the border density to be 0,
                    density_mat = np.zeros_like(density[0, :, :])
                    density_mat[bdr: -bdr, bdr: -bdr] = density[0, bdr:-bdr, bdr:-bdr]
                else:
                    density_mat = density[0]
                axs[i, 2 + k].imshow(density_mat)
                axs[i, 2 + k].axis("off")

    figh.show()
    return figh, idx_col

def visualize_augmented_dataset(train_dataset, n_views=10,
        idxs=(96659, 54019, 88327, 81148, 98469, 77493, 131, 58202, 66666, 65017, 1973, 29975), ):
    train_dataset.n_views = n_views
    idx_col = [] if idxs is None else idxs
    img_col = []
    for i in range(12):
        if idxs is None:
            idx = np.random.randint(1E5)
            idx_col.append(idx)
        else:
            idx = idxs[i]
        imgs, _ = train_dataset[idx]
        img_col.extend(imgs)

    mtg = make_grid(img_col, nrow=n_views, padding=4)
    train_dataset.n_views = 2
    return ToPILImage()(mtg)


def visualize_samples(train_dataset, idxs=None):
    imgs, _ = train_dataset[1]
    figh, axs = plt.subplots(len(imgs), 10, figsize=(15, len(imgs) * 1.6))
    idx_col = [] if idxs is None else idxs
    for i in range(10):
        if idxs is None:
            idx = np.random.randint(1E5)
            idx_col.append(idx)
        else:
            idx = idxs[i]
        imgs , _ = train_dataset[idx]
        for j in range(len(imgs)):
            axs[j, i].imshow(imgs[j].permute([1,2,0]))
            axs[j, i].axis("off")

    figh.show()
    return figh, idx_col

#%%
if __name__ == '__main__':
    crop_temperature = 1.5
    pad_img = True
    dataset_dir = r"E:\Datasets"
    cropper = RandomResizedCrop_with_Density(96, temperature=crop_temperature, pad_if_needed=pad_img)
    train_dataset = Contrastive_STL10_w_salmap(dataset_dir=dataset_dir,
               density_cropper=cropper, split="unlabeled")  # imgv1, imgv2 =  saldataset[10]

    #%%
    idxs = [96659, 54019, 88327, 81148, 98469, 77493, 131, 58202, 66666, 65017]
    #%%
    # crop_temperature = 1.5
    # pad_img = True
    cropper = RandomResizedCrop_with_Density(96, \
            temperature=crop_temperature, pad_if_needed=pad_img)
    cropper.pad_if_needed = False
    cropper.temperature = 15
    train_dataset.n_views = 7
    train_dataset.density_cropper = cropper
    _, idxs = visualize_samples(train_dataset, idxs)
    #%%

    # figh.savefig("/scratch1/fs1/crponce/Datasets/example%03d.png"%np.random.randint(1E3))
    #%% The baseline
    rndcropper = RandomResizedCrop(96,)
    bsl_cropper = lambda img, salmap: rndcropper(img)
    train_dataset.density_cropper = bsl_cropper
    _, idxs = visualize_samples(train_dataset, idxs)

    #%%
    cropper = RandomResizedCrop_with_Density(96,
                    temperature=0.1, pad_if_needed=False)

    train_dataset = Contrastive_STL10_w_salmap(dataset_dir=r"E:\Datasets",
                density_cropper=cropper, split="unlabeled", salmap_control=False,
                disable_crop=True)  # imgv1, imgv2 =  saldataset[10]
    #%%
    train_dataset.disable_crop = True
    train_dataset.n_views = 7
    train_dataset.transform = train_dataset.get_simclr_post_crop_transform(96,
                            blur=True, foveation=True,
                            kerW_coef=0.06, fov_area_rng=[0.1, 0.5], bdr=12)
    idxs = [96659, 54019, 88327, 81148, 98469, 77493, 131, 58202, 66666, 65017]
    _, idxs = visualize_samples(train_dataset, idxs)
    #%%
    from data_aug.dataset_w_salmap import Contrastive_STL10_w_CortMagnif, get_RandomMagnifTfm
    train_dataset = Contrastive_STL10_w_CortMagnif(dataset_dir=r"E:\Datasets",
        transform=None, split="unlabeled", n_views=2,
        crop=False, magnif=True, sal_sample=False, )
    train_dataset.magnifier = get_RandomMagnifTfm(grid_generator="radial_quad_isotrop",
                  bdr=16, fov=20, K=20, slope_C=0.012,
                  sal_sample=False, sample_temperature=1.5,)
    #%%
    visualize_saliency_maps_w_imgs(train_dataset, temperature=3, bdr=16)