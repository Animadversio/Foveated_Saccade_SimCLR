import os
from os.path import join
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch import Tensor
from torchvision import datasets, transforms
from torchvision.models import resnet50, resnet18
from tqdm import tqdm


def showimg(image,figsize=[8,8],cmap=None):
  if type(image) in [tuple,list]:
    for i in range(len(image)):
      figh,ax = showimg(image[i], figsize=figsize,cmap=cmap)
  elif len(image.shape)==4:
    for i in range(image.shape[0]):
      figh,ax = showimg(image[i], figsize=figsize,cmap=cmap)
  else:
    if len(image.shape)==3 and image.shape[2]==1:
      image = image[:,:,0]
    figh,ax = plt.subplots(figsize=figsize)
    plt.imshow(image,cmap=cmap)
    plt.axis("off")
    plt.show()
  return figh,ax


def image_standardize(image):
  if len(image.shape)==2:
    image = np.repeat(image[:,:,np.newaxis],3,axis=2)
  elif image.shape[2]==1:
    image = np.repeat(image[:,:,:],3,axis=2)
  elif image.shape[2]==4:
    image = image[:,:,:3]
  elif (image.shape[2]==3) and len(image.shape)==3:
    pass
  else:
    raise ValueError("Shape of image is %s error",image.shape)
  return image


def resnet_forward(model, x: Tensor) -> Tensor:
  # See note [TorchScript super()]
  x = model.conv1(x)
  x = model.bn1(x)
  x = model.relu(x)
  x = model.maxpool(x)

  x = model.layer1(x)
  x = model.layer2(x)
  x = model.layer3(x)
  x = model.layer4(x)

  x = model.avgpool(x)
  x = torch.flatten(x, 1)
  x = model.fc(x)
  return x

## To get resnet functions 
def resnet_feature(model, x: Tensor, layers=(2, 3, 4)) -> Tensor:
  feature_dict = {}
  # See note [TorchScript super()]
  x = model.conv1(x)
  x = model.bn1(x)
  x = model.relu(x)
  x = model.maxpool(x)

  x = model.layer1(x)
  if 1 in layers: feature_dict["layer1"] = x.detach().cpu().clone()
  x = model.layer2(x)
  if 2 in layers: feature_dict["layer2"] = x.detach().cpu().clone()
  x = model.layer3(x)
  if 3 in layers: feature_dict["layer3"] = x.detach().cpu().clone()
  x = model.layer4(x)
  if 4 in layers: feature_dict["layer4"] = x.detach().cpu().clone()
  return feature_dict


def saliency_map(feature_dict, weight=(1,2,1)):
  layer2map = torch.linalg.norm(feature_dict["layer2"], dim=1, ord=2, keepdims=True)
  layer3map = torch.linalg.norm(feature_dict["layer3"], dim=1, ord=2, keepdims=True)
  layer4map = torch.linalg.norm(feature_dict["layer4"], dim=1, ord=2, keepdims=True)
  W2, W3, W4 = weight
  salmap = F.interpolate(layer2map, size=(224, 224), mode="bilinear", align_corners=True)*W2+\
    F.interpolate(layer2map, size=(224, 224), mode="bilinear", align_corners=True)*W3+\
    F.interpolate(layer2map, size=(224, 224), mode="bilinear", align_corners=True)*W4
  return salmap


def resnet_saliency(model, x: Tensor, layersW=(None, 1, 2, 1), return_maps=False):
  """Combine the 2 functions above"""
  map_dict = {}
  B, H, W = x.shape[0],x.shape[2],x.shape[3]
  salmap = torch.zeros([B, 1, H, W]).to("cuda")
  x = model.conv1(x)
  x = model.bn1(x)
  x = model.relu(x)
  x = model.maxpool(x)

  x = model.layer1(x)
  if layersW[1 - 1]: 
    map_dict["layer1"] = torch.linalg.norm(x.detach(), dim=1, ord=2, keepdims=True)
    salmap += F.interpolate(map_dict["layer1"], size=(H, W), mode="bilinear", align_corners=True) * layersW[1 - 1]
  x = model.layer2(x)
  if layersW[2 - 1]: 
    map_dict["layer2"] = torch.linalg.norm(x.detach(), dim=1, ord=2, keepdims=True)
    salmap += F.interpolate(map_dict["layer2"], size=(H, W), mode="bilinear", align_corners=True) * layersW[2 - 1]
  x = model.layer3(x)
  if layersW[3 - 1]: 
    map_dict["layer3"] = torch.linalg.norm(x.detach(), dim=1, ord=2, keepdims=True)
    salmap += F.interpolate(map_dict["layer3"], size=(H, W), mode="bilinear", align_corners=True) * layersW[3 - 1]
  x = model.layer4(x)
  if layersW[4 - 1]: 
    map_dict["layer4"] = torch.linalg.norm(x.detach(), dim=1, ord=2, keepdims=True)
    salmap += F.interpolate(map_dict["layer4"], size=(H, W), mode="bilinear", align_corners=True) * layersW[4 - 1]
  if return_maps:
    return salmap, map_dict
  else:
    del map_dict
    return salmap

#%% nonlinear functions to map saliency map to a alpha map on the image
def threshfunc(salmap):
  return salmap>torch.mean(salmap, dim=[1,2,3])

def softthreshfunc(salmap):
  floor,_ = salmap.view([salmap.shape[0],-1]).min(dim=1, keepdim=True)
  floor = floor.unsqueeze(1).unsqueeze(2)
  ceil = torch.mean(salmap, dim=[1,2,3], keepdim=True)
  return torch.clamp((salmap-ceil)/(ceil-floor),-1,0) + 1

def linearfunc(salmap):
  floor,_ = salmap.view([salmap.shape[0],-1]).min(dim=1, keepdim=True)
  floor = floor.unsqueeze(1).unsqueeze(2)
  ceil,_ = salmap.view([salmap.shape[0],-1]).max(dim=1, keepdim=True)
  ceil = ceil.unsqueeze(1).unsqueeze(2)
  # ceil = torch.mean(salmap, dim=[1,2,3], keepdim=True)
  return torch.clamp((salmap-floor)/(ceil-floor),0,1)

def linearqtlfunc(salmap, qtl=(0.20,0.90)):
  floor = torch.quantile(salmap.view([salmap.shape[0],-1]), qtl[0], dim=1, keepdim=True)
  floor = floor.unsqueeze(1).unsqueeze(2)
  ceil = torch.quantile(salmap.view([salmap.shape[0],-1]), qtl[1], dim=1, keepdim=True)
  ceil = ceil.unsqueeze(1).unsqueeze(2)
  # ceil = torch.mean(salmap, dim=[1,2,3], keepdim=True)
  return torch.clamp((salmap-floor)/(ceil-floor),0,1)

def tsr2image(img):
  inv_norm = transforms.Normalize(
      mean=(-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010),
      std=(1/0.2023, 1/0.1994, 1/0.2010))
  img = inv_norm(img)
  img = img.permute(1, 2, 0)
  return img.numpy()

# preprocess data set 
def saliency_process_dataset(model, dataset, saveprefix="sal_", 
        layersW=(None, 1, 2, 1), alphamapfunc=linearqtlfunc, procall=False):
  for subfdr in os.listdir(imgroot/"train"):
    os.makedirs(imgroot/(saveprefix+"train")/subfdr, exist_ok=True)
  
  for i, (img_pp, label) in tqdm(enumerate(dataset)):
    imgpath = dataset.img_labels.path[i]
    imgname, ext = os.path.splitext(imgpath)
    with torch.no_grad():
      salmap = resnet_saliency(model, img_pp.unsqueeze(0).cuda(), layersW=layersW, return_maps=False).cpu()
    np.save(imgroot/(saveprefix+imgname+".npy"), salmap.numpy())
    # image = img_pp * alphamapfunc(salmap)[0]
    # imsave(imgroot/(saveprefix+imgname+".png"), (255.0*tsr2image(image)).astype('uint8'))
    if not procall: 
      if i==5: break 

  # saliency_process_dataset(model_sup, train_dataset_nosal, saveprefix="sal_", procall=True) # supervised model L2 saliency>

def process_stl10(dataset_dir="/scratch1/fs1/crponce/Datasets", layersW=(None,1,2,1)):
  model = resnet18(pretrained=True).cuda()
  dataset = datasets.STL10(dataset_dir, split='unlabeled', download=True, 
                            transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))]), 
                          )
  salmap_col = []
  for i, (img_pp, _) in tqdm(enumerate(dataset)): 
    with torch.no_grad():
      salmap = resnet_saliency(model, img_pp.unsqueeze(0).cuda(), layersW=layersW, return_maps=False).cpu()
    salmap_col.append(salmap.numpy())

  salmap_arr = np.array(salmap_col)
  np.save(join(dataset_dir, "stl10_unlabeled_saliency.npy"), salmap_arr)

def process_stl10_fastsal(ckpt="weights/salicon_A.pth"):
  import model.fastSal as fastsal
  from utils import load_weight
  model = fastsal.fastsal(pretrain_mode=False, model_type='A')
  state_dict, opt_state = load_weight(ckpt, remove_decoder=False)
  model.load_state_dict(state_dict)
  model.cuda().eval()

  import torch
  import numpy as np
  import pandas as pd
  from tqdm import tqdm
  from os.path import join
  import matplotlib.pylab as plt
  import torch.nn.functional as F
  from torch.utils.data import Dataset, DataLoader
  from torchvision import datasets, transforms, utils

  dataset = datasets.STL10("/scratch1/fs1/crponce/Datasets", split="unlabeled", download=True, transform=transforms.ToTensor(),)
  dataloader = DataLoader(dataset, batch_size=75, shuffle=False, drop_last=False)
  salmap_col = []
  for images, _ in tqdm(dataloader):
    img_tsr = F.interpolate(images.to('cuda'), [512, 512]) 

    with torch.no_grad():
      salmap = model(img_tsr)

    salmap_small = F.interpolate(salmap, [96, 96]).cpu().numpy()
    salmap_col.append(salmap_small)

  salmap_arr = np.concatenate(salmap_col, axis=0)
  np.save("/scratch1/fs1/crponce/Datasets/stl10_unlabeled_salmaps_salicon.npy",salmap_arr)


def visualize_salmaps():
  figh, axs = plt.subplots(2, 10, figsize=(14, 3.5))
  for i in range(10):
    idx = np.random.randint(1E5)
    img, _ = dataset[idx]
    salmap = salmap_arr[idx,0,:,:]
    axs[0, i].imshow(img.permute([1,2,0]))
    axs[0, i].axis("off")
    axs[1, i].imshow(salmap)
    axs[1, i].axis("off")
  figh.savefig("/scratch1/fs1/crponce/Datasets/example%03d.png"%np.random.randint(1E3))