
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision import datasets, transforms, utils
import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pylab as plt
from data_aug.gaussian_blur import GaussianBlur
from data_aug.view_generator import ContrastiveLearningViewGenerator

class STL10_w_salmap(Dataset):
    """ Return STL image with saliency maps """

    def __init__(self, dataset_dir=r"/scratch1/fs1/crponce/Datasets", transform=None, split="unlabeled"):
        """
        Args:
            dataset_dir (string): Directory with all the images. E:\Datasets
            transform (callable, optional): Optional transform to be applied
                on a sample.
            split: "unlabeled"
        """
        self.dataset = datasets.STL10(dataset_dir, split=split, download=True,
                                 transform=None,)
        self.salmaps = np.load(join(dataset_dir, "stl10_unlabeled_salmaps_salicon.npy")) # stl10_unlabeled_saliency.npy
        assert len(self.dataset) == self.salmaps.shape[0]
        # transforms.Compose([transforms.ToTensor(),
        #                     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
        #                                          std=(0.2023, 0.1994, 0.2010))])
        self.root_dir = dataset_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset.__getitem__(idx) # img is PIL.Image, label is xxxx 
        salmap = self.salmaps[idx, :, :, :].astype('float') # numpy.ndarray
        if self.transform:
            img = self.transform(img)
        # salmap_tsr = torch.tensor(salmap).unsqueeze(0).float()
        return (img, salmap), label  # labels can be dropped.

from .foveation import get_RandomFoveationTfm
from .saliency_random_cropper import RandomResizedCrop_with_Density, RandomCrop_with_Density
class Contrastive_STL10_w_salmap(Dataset):
    """ Return Crops of STL10 images with saliency maps """

    def __init__(self, dataset_dir=r"/scratch1/fs1/crponce/Datasets", \
        density_cropper=RandomResizedCrop_with_Density((96, 96),), \
        transform_post_crop=None, split="unlabeled", n_views=2,
        salmap_control=False, disable_crop=False):
        """
        Args:
            dataset_dir (string): Directory with all the images. E:\Datasets
            transform (callable, optional): Optional transform to be applied
                on a sample.
            split: "unlabeled"
        """
        self.dataset = datasets.STL10(dataset_dir, split=split, download=True,
                                 transform=None,)
        self.salmap_control = salmap_control 
        if self.salmap_control: # if true, use flat maps. We can implement random maps in the future. 
            print("Use control saliency map, instead of real ones, data not loading. Temperature disabled.")
        else:
            self.salmaps = np.load(join(dataset_dir, "stl10_unlabeled_salmaps_salicon.npy")) # stl10_unlabeled_saliency.npy
            assert len(self.dataset) == self.salmaps.shape[0]

        self.root_dir = dataset_dir
        self.density_cropper = density_cropper
        self.disable_crop = disable_crop

        if transform_post_crop is not None:
            self.transform = transform_post_crop
        else:
            self.transform = self.get_simclr_post_crop_transform(96, s=1, blur=True, foveation=False)  # default transform pipeline.
        self.n_views = n_views

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset.__getitem__(idx) # img is PIL.Image, label is xxxx 
        if self.salmap_control:
            salmap_tsr = torch.ones(1, 1, 96, 96).float() # flat saliency map, size hard coded. 
        else:
            salmap = self.salmaps[idx, :, :, :].astype('float')  # numpy.ndarray
            salmap_tsr = torch.tensor(salmap).unsqueeze(0).float()  #F.interpolate(, [96, 96])

        if self.disable_crop:
            sal_crops = [img for i in range(self.n_views)]
        else:
            sal_crops = [self.density_cropper(img, salmap_tsr) for i in range(self.n_views)]

        if self.transform:
            imgs = [self.transform(cropview) for cropview in sal_crops]
            return imgs, -1
        else:
            return sal_crops, -1 


    @staticmethod
    def get_simclr_post_crop_transform(size, s=1, blur=True, foveation=False,
                               kerW_coef=0.06, fov_area_rng=(0.01, 0.5), bdr=12):
        """Return a set of data augmentation transformations as described in the SimCLR paper.
        kerW_coef: =0.06,
        fov_area_rng: =(0.01, 0.5),
        bdr: =12

        kerW_coef, fov_area_rng, bdr these parameters will be disabled if foveation=False
        """
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        tfm_list = [transforms.RandomHorizontalFlip(),
                  transforms.RandomApply([color_jitter], p=0.8),
                  transforms.RandomGrayscale(p=0.2),
                  transforms.ToTensor()
                  ]  # hard to do foveation without having a tensor

        if foveation:
            tfm_list.append(get_RandomFoveationTfm(kerW_coef=kerW_coef, fov_area_rng=fov_area_rng, bdr=bdr))

        if blur:
            tfm_list.append(GaussianBlur(kernel_size=int(0.1 * size), return_PIL=False))

        data_transforms = transforms.Compose(tfm_list)
        # transforms.Compose([transforms.ToTensor(),
        #                     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
        #                                          std=(0.2023, 0.1994, 0.2010))])
        return data_transforms


from .cort_magnif_tfm import get_RandomMagnifTfm
class Contrastive_STL10_w_CortMagnif(Dataset):
    """ Return Crops of STL10 images with saliency maps """

    def __init__(self, dataset_dir=r"/scratch1/fs1/crponce/Datasets", \
        transform=None, split="unlabeled", n_views=2,
        crop=False, magnif=False, sal_sample=False, sal_control=False):
        """
        Args:
            dataset_dir (string): Directory with all the images. E:\Datasets
            transform (callable, optional): Optional transform to be applied
                on a sample.
            split: "unlabeled"
        """
        self.dataset = datasets.STL10(dataset_dir, split=split, download=True,
                                 transform=None,)

        self.salmaps = np.load(join(dataset_dir, "stl10_unlabeled_salmaps_salicon.npy")) # stl10_unlabeled_saliency.npy
        assert len(self.dataset) == self.salmaps.shape[0]
        self.root_dir = dataset_dir
        self.crop = crop
        self.magnif = magnif
        self.magnifier = None 
        self.sal_sample = sal_sample  # used in magnifier, not used here ! can be omited
        self.sal_control = sal_control

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.get_simclr_pre_magnif_transform(96, s=1, blur=True, crop=self.crop, )  # default transform pipeline.
        self.n_views = n_views

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset.__getitem__(idx) # img is PIL.Image, label is xxxx
        salmap = self.salmaps[idx, :, :, :].astype('float')  # numpy.ndarray
        salmap_tsr = torch.tensor(salmap).unsqueeze(0).float()  #F.interpolate(, [96, 96])
        if self.sal_control: 
            print("Use flat salincy map as control")
            salmap_tsr = torch.ones([1,1,96,96]).float()

        views = [img for i in range(self.n_views)]

        if self.transform:
            imgs = [self.transform(cropview) for cropview in views]
        else:
            imgs = views

        if self.magnif and self.magnifier is not None:
            finalviews = [self.magnifier(img, salmap_tsr) for img in imgs]
        else:
            finalviews = imgs

        return finalviews, -1

    @staticmethod
    def get_simclr_pre_magnif_transform(size, s=1, crop=False, blur=True, ):
        """Return a set of data augmentation transformations as described in the SimCLR paper.
        """
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        tfm_list = []
        if crop:
            tfm_list += [transforms.RandomResizedCrop(96)]
        tfm_list += [transforms.RandomHorizontalFlip(),
                     transforms.RandomApply([color_jitter], p=0.8),
                     transforms.RandomGrayscale(p=0.2),
                     transforms.ToTensor()
                     ]  # hard to do foveation without having a tensor
        if blur:
            tfm_list.append(GaussianBlur(kernel_size=int(0.1 * size), return_PIL=False))
        data_transforms = transforms.Compose(tfm_list)
        # transforms.Compose([transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
        #                                          std=(0.2023, 0.1994, 0.2010))])
        return data_transforms

    @staticmethod
    def get_simclr_magnif_transform(size, s=1, crop=False, blur=True,
                                    magnif=True, gridfunc_form="radial_quad",
                                    bdr=16, fov=20, K=0, slope_C=1.5, cover_ratio=(0.05, 1),
                                    sal_sample=False, sample_temperature=1.5,):
        """Return a set of data augmentation transformations as described in the SimCLR paper.
        OBSOLETE, USE the above function instead
        bdr
        fov
        K
        slope_C : Slope in the radial exp mapping
        cover_ratio
        """
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        tfm_list = []
        if crop:
            tfm_list += [transforms.RandomResizedCrop(96)]
        tfm_list += [transforms.RandomHorizontalFlip(),
                  transforms.RandomApply([color_jitter], p=0.8),
                  transforms.RandomGrayscale(p=0.2),
                  transforms.ToTensor()
                  ]  # hard to do foveation without having a tensor
        if magnif:
            if gridfunc_form == "radial_quad":
                tfm_list.append(get_RandomMagnifTfm(grid_generator="radial_quad_isotrop",
                                                    bdr=bdr, fov=fov, K=K,
                                                    cover_ratio=cover_ratio))
            elif gridfunc_form == "radial_exp":
                tfm_list.append(get_RandomMagnifTfm(grid_generator="radial_exp_isotrop",
                                                    bdr=bdr, slope_C=slope_C,
                                                    cover_ratio=cover_ratio))
            else:
                raise NotImplemented

        if blur:
                tfm_list.append(GaussianBlur(kernel_size=int(0.1 * size), return_PIL=False))

        data_transforms = transforms.Compose(tfm_list)
        # transforms.Compose([transforms.ToTensor(),
        #                     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
        #                                          std=(0.2023, 0.1994, 0.2010))])
        return data_transforms
    


def visualize_samples(saldataset):
    figh, axs = plt.subplots(2, 10, figsize=(14, 3.5))
    for i in range(10):
        idx = np.random.randint(1E5)
        (img, salmap) , _ = saldataset[idx]
        axs[0, i].imshow(img.permute([1,2,0]))
        axs[0, i].axis("off")
        axs[1, i].imshow(salmap[0])
        axs[1, i].axis("off")
    figh.savefig("/scratch1/fs1/crponce/Datasets/example%03d.png" % np.random.randint(1E3))
# face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
#                                     root_dir='data/faces/')
#
# fig = plt.figure()
#
# for i in range(len(face_dataset)):
#     sample = face_dataset[i]
#
#     print(i, sample['image'].shape, sample['landmarks'].shape)
#
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     show_landmarks(**sample)
#
#     if i == 3:
#         plt.show()
#         break
#%%
# import matplotlib.pylab as plt
# img, salmap = STL10_sal[89930]
# fig, axs = plt.subplots(1, 2, figsize=[8, 4.5])
# axs[0].imshow(img[0])
# axs[1].imshow(salmap[0, 0, :, :])
# plt.show()


if __name__ == "__main__":
    #%%
    from data_aug.aug_utils import send_to_clipboard
    from data_aug.dataset_w_salmap import Contrastive_STL10_w_CortMagnif
    from data_aug.visualize_aug_dataset import visualize_augmented_dataset
    dataset = Contrastive_STL10_w_CortMagnif(r"E:\Datasets")
    #%%
    dataset.transform = dataset.get_simclr_magnif_transform(96, blur=True, magnif=True,
                        bdr=16, gridfunc_form="radial_quad", fov=20, K=20, cover_ratio=(0.05, 0.7))
    img_pil = visualize_augmented_dataset(dataset, n_views=10,)
    send_to_clipboard(img_pil)
    img_pil.show()
    # %%
    dataset.transform = dataset.get_simclr_magnif_transform(96, blur=True, magnif=True,
                        bdr=16, gridfunc_form="radial_exp", slope_C=(0.75, 3.0), cover_ratio=(0.05, 0.7))
    img_pil = visualize_augmented_dataset(dataset, n_views=10, )
    send_to_clipboard(img_pil)
    img_pil.show()

