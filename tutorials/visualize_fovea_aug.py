
from data_aug.foveation import randomFoveated, FoveateAt
from scipy.misc import face
import numpy as np
import torch
from torchvision import datasets
from PIL import Image
import matplotlib.pylab as plt
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage, ToTensor
from torch.nn.functional import interpolate
from .aug_utils import send_to_clipboard

#%%
facetsr = torch.tensor(face()/255.0).float().permute([2,0,1]).unsqueeze(0)
facetsr_rsz = interpolate(facetsr, [192, 256])
#%% test different kerW parameters
views = randomFoveated(facetsr_rsz, 9, bdr=16, kerW_coef=0.01)
ToPILImage()(make_grid(views.squeeze(1),nrow=3)).show()
#%%

stldata = datasets.STL10(r"E:\Datasets", transform=ToTensor(), split="train")#unlabeled
#%%
imgtsr, _ = stldata[np.random.randint(5000)]
views = randomFoveated(imgtsr.unsqueeze(0), 9, bdr=12, kerW_coef=0.06, spacing=0.2, fov_area_rng=(0.01, 0.5))
mtg = ToPILImage()(make_grid(views, nrow=3))
send_to_clipboard(mtg)
mtg.show()

#%%

