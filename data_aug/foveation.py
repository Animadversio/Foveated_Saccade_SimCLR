"""PyTorch translation of the TF version of foveation code"""
import torch
from kornia.filters import gaussian_blur2d
import numpy as np
import math
import matplotlib.pylab as plt
pi = math.pi


def cosfunc(x):
  """The cosine square smoothing function"""
  Lower = torch.cos(pi*(x + 1/4))**2;
  Upper = 1 - torch.cos(pi*(x - 3/4))**2;
  # print(tf.logical_and((x <= -1/4), (x > -3/4)).dtype)
  fval = torch.where(((x <= -1/4) & (x >-3/4)), Lower, torch.zeros(1)) + \
      torch.where(((x >= 1/4) & (x <= 3/4)), Upper, torch.zeros(1)) + \
      torch.where(((x < 1/4) & (x > -1/4)), torch.ones(1), torch.zeros(1))
  return fval


def rbf(ecc, N, spacing, e_o=1.0):
  """ Number N radial basis function
    ecc: eccentricities, torch tensor.  
    N: numbering of basis function, starting from 0. 
    spacing: log scale spacing of ring radius (deg), scalar.
    e_o: radius of 0 string, scalar. 
  """
  spacing = torch.tensor(spacing).float()
  e_o = torch.tensor(e_o).float()
  preinput = (torch.log(ecc) - (torch.log(e_o) + (N + 1) * spacing)) / spacing
  ecc_basis = cosfunc(preinput);
  return ecc_basis


def fov_rbf(ecc, spacing, e_o=1.0):
  """Initial radial basis function
    ecc: eccentricities, torch tensor.  
  """
  spacing = torch.tensor(spacing).float()
  e_o = torch.tensor(e_o).float()
  preinput = (torch.log(ecc) - torch.log(e_o)) / spacing
  preinput = torch.clamp(preinput, 0.0, 1.0) # only clip 0 is enough.
  ecc_basis = cosfunc(preinput);
  return ecc_basis


def FoveateAt(img_tsr, pnt:tuple, kerW_coef=0.04, e_o=1, \
    N_e=None, spacing=0.5, demo=False):
  """Apply foveation transform at (x,y) coordinate `pnt` to `img`

  Parameters: 
    kerW_coef: how gaussian filtering kernel std scale as a function of eccentricity 
    e_o: eccentricity of the initial ring belt
    spacing: log scale spacing between eccentricity of ring belts. 
    N_e: Number of ring belts in total. if None, it will calculate the N_e s.t. the whole image is covered by ring belts.
    bdr: width (in pixel) of border region that forbid sampling (bias foveation point to be in the center of img)
  """
  if img_tsr.ndim == 3:
    img_tsr = img_tsr.unsqueeze(0)
  H, W = img_tsr.shape[2], img_tsr.shape[3]  # if this is fixed then these two steps could be saved
  XX, YY = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32))
  deg_per_pix = 20 / math.sqrt(H**2 + W**2)
  # pixel coordinate of fixation point.
  xid, yid = pnt
  D2fov = torch.sqrt((XX - xid)**2 + (YY - yid)**2)
  D2fov_deg = D2fov * deg_per_pix
  maxecc = math.sqrt(max(xid, W-xid)**2 + max(yid, H-yid)**2) * deg_per_pix  # max([D2fov_deg[0,0], D2fov_deg[-1,0], D2fov_deg[0,-1], D2fov_deg[-1,-1]])
  # maxecc = max(D2fov_deg[0,0], D2fov_deg[-1,0], D2fov_deg[0,-1], D2fov_deg[-1,-1]) # maximal deviation at 4 corner
  # maxecc = tf.reduce_max([D2fov_deg[0,0], D2fov_deg[-1,0], D2fov_deg[0,-1], D2fov_deg[-1,-1]])
  # maxecc = max([D2fov_deg[0,0], D2fov_deg[-1,0], D2fov_deg[0,-1], D2fov_deg[-1,-1]])
  # e_r = maxecc; # 15
  if N_e is None:
    N_e = np.ceil((np.log(maxecc)-np.log(e_o))/spacing).astype("int32") 
    # N_e will be a numpy.int32 number
  rbf_basis = fov_rbf(D2fov_deg, spacing, e_o)
  finalimg = rbf_basis[None, None, :, :] * img_tsr # tf.expand_dims(rbf_basis,-1)
  for N in range(N_e):
    rbf_basis = rbf(D2fov_deg, N, spacing, e_o=e_o)
    mean_dev = math.exp(math.log(e_o) + (N + 1) * spacing)
    kerW = kerW_coef * mean_dev / deg_per_pix
    kerSz = int(kerW * 3)
    if kerSz % 2 == 0:  # gaussian_blur2d needs a odd number sized kernel.
      kerSz += 1
    img_gsft = gaussian_blur2d(img_tsr, kernel_size=(kerSz, kerSz), sigma=(kerW, kerW), border_type='reflect')
    finalimg = finalimg + rbf_basis[None,None,:,:] * img_gsft # tf.expand_dims(rbf_basis,-1)
  
  if demo: # Comment out this part when really run. 
    figh,ax = plt.subplots(figsize=[10,10])
    plt.imshow(finalimg.squeeze(0).permute(1,2,0))
    plt.axis("off")
    plt.show()
    figh,ax = plt.subplots(figsize=[10,10])
    plt.imshow(finalimg.squeeze(0).permute(1,2,0))
    plt.axis("off")
    vis_belts(ax, img_tsr.squeeze(0).permute(1,2,0), pnt, kerW_coef, e_o, N_e, spacing)
    figh.show()
  return finalimg 


def randomFoveated(img_tsr, pntN:int, kerW_coef=0.04, fov_area_rng=1, N_e=None, spacing=0.5, bdr=32, tfm_ver=False):
  """Randomly apply `pntN` foveation transform to `img`

  Parameters: 
    kerW_coef: how gaussian filtering kernel std scale as a function of eccentricity 
    e_o: eccentricity of the initial ring belt, or
    spacing: log scale spacing between eccentricity of ring belts. 
    N_e: Number of ring belts in total. if None, it will calculate the N_e s.t. the whole image is covered by ring belts.
    bdr: width (in pixel) of border region that forbid sampling (bias foveation point to be in the center of img)
  """
  if img_tsr.ndim == 3:
    img_tsr = img_tsr.unsqueeze(0)
  H, W = img_tsr.shape[2], img_tsr.shape[3] # if this is fixed then these two steps could be saved
  XX, YY = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32))
  deg_per_pix = 20/math.sqrt(H**2+W**2);
  if type(fov_area_rng) in [list, tuple, np.ndarray]:
    Rand_Fov_Size = True
  else:
    fov_rad = math.sqrt(H * W * fov_area_rng / pi)
    e_o = fov_rad * deg_per_pix
    Rand_Fov_Size = False
  xids = torch.randint(bdr, W-bdr, (pntN,), dtype=torch.int32)
  yids = torch.randint(bdr, H-bdr, (pntN,), dtype=torch.int32)
  finimg_list = []
  for it in range(pntN):
    xid, yid = xids[it], yids[it]  # pixel coordinate of fixation point.
    if Rand_Fov_Size:
      area_ratio = np.random.uniform(fov_area_rng[0], fov_area_rng[1])
      fov_rad = math.sqrt(H * W * area_ratio / pi)
      e_o = fov_rad * deg_per_pix

    D2fov = torch.sqrt((XX - xid)**2 + (YY - yid)**2)
    D2fov_deg = D2fov * deg_per_pix
    maxecc = max(D2fov_deg[0,0], D2fov_deg[-1,0], D2fov_deg[0,-1], D2fov_deg[-1,-1]) # maximal deviation at 4 corner
    e_r = maxecc; # 15
    # if N_e is None:
    N_e = np.ceil((np.log(maxecc)-np.log(e_o))/spacing+1).to(torch.int32)
    # print(N_e)
    # spacing = tf.convert_to_tensor((math.log(e_r) - math.log(e_o)) / N_e);
    # spacing = tf.convert_to_tensor(spacing, dtype="float32");
    rbf_basis = fov_rbf(D2fov_deg, spacing, e_o)
    finalimg = rbf_basis[None,None,:,:] * img_tsr # tf.expand_dims(rbf_basis,-1)
    for N in range(N_e):
      rbf_basis = rbf(D2fov_deg, N, spacing, e_o=e_o)
      mean_dev = math.exp(math.log(e_o) + (N + 1) * spacing)
      kerW = kerW_coef * mean_dev / deg_per_pix
      kerSz = int(kerW * 3)
      if kerSz % 2 == 0: # gaussian_blur2d needs a odd number sized kernel.
        kerSz += 1
      img_gsft = gaussian_blur2d(img_tsr, kernel_size=(kerSz, kerSz), sigma=(kerW, kerW), border_type='reflect')
      finalimg = finalimg + rbf_basis[None,None,:,:] * img_gsft # tf.expand_dims(rbf_basis,-1)
    finimg_list.append(finalimg)
  finimgs = torch.cat(finimg_list, dim=0)
  finimgs.clamp_(0.0, 1.0)
  return finimgs.squeeze(0) if tfm_ver else finimgs

def get_RandomFoveationTfm(kerW_coef=0.06, fov_area_rng=(0.01, 0.5), bdr=12):
  tfm = lambda imgtsr: randomFoveated(imgtsr, 1, kerW_coef=kerW_coef, fov_area_rng=fov_area_rng, spacing=0.2, bdr=bdr, tfm_ver=True)
  return tfm


def vis_belts(ax, img, pnt, kerW_coef=0.04, e_o=1, N_e=None, spacing=0.5):
  """A visualization helper for parameter tuning and diagnostics purpose.
    It plot out the masking belts for the computation, with the flat region and the smoothing region.

  """
  if ax is None: ax = plt.gca()
  H, W = img.shape[0], img.shape[1]
  deg_per_pix = 20/math.sqrt(H**2+W**2);
  # pixel coordinate of fixation point.
  xid, yid = pnt
  if N_e is None:
    maxecc = math.sqrt(max(xid, H-xid)**2 + max(yid,W-yid)**2) * deg_per_pix
    N_e = np.ceil((np.log(maxecc)-np.log(e_o))/spacing).astype("int32")
    
  print("radius of belt center:",)
  for N in range(N_e):
    radius = math.exp(math.log(e_o) + (N+1) * spacing) / deg_per_pix
    inner_smooth_rad = math.exp(math.log(e_o) + (N+1-1/4) * spacing) / deg_per_pix
    inner_smooth_rad2 = math.exp(math.log(e_o) + (N+1-3/4) * spacing) / deg_per_pix
    outer_smooth_rad = math.exp(math.log(e_o) + (N+1+1/4) * spacing) / deg_per_pix
    outer_smooth_rad2 = math.exp(math.log(e_o) + (N+1+3/4) * spacing) / deg_per_pix
    circle1 = plt.Circle((xid, yid), inner_smooth_rad, color='r', linestyle=":", fill=False, clip_on=False)
    circle12 = plt.Circle((xid, yid), inner_smooth_rad2, color='r', linestyle=":", fill=False, clip_on=False)
    circle3 = plt.Circle((xid, yid), outer_smooth_rad, color='r', linestyle=":", fill=False, clip_on=False)
    circle32 = plt.Circle((xid, yid), outer_smooth_rad2, color='r', linestyle=":", fill=False, clip_on=False)
    circle2 = plt.Circle((xid, yid), radius, color='k', linestyle="-.", fill=False, clip_on=False)
    ax.plot(xid,yid,'ro')
    ax.add_patch(circle1)
    ax.add_patch(circle12)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    ax.add_patch(circle32)

#%%
def FoveateAt_demo(img_tsr, pnt: tuple, kerW_coef=0.04, e_o=1,
              N_e=None, spacing=0.5, ):
  """Apply foveation transform at (x,y) coordinate `pnt` to `img`, demo version

  Parameters:
    kerW_coef: how gaussian filtering kernel std scale as a function of eccentricity
    e_o: eccentricity of the initial ring belt
    spacing: log scale spacing between eccentricity of ring belts.
    N_e: Number of ring belts in total. if None, it will calculate the N_e s.t. the whole image is covered by ring belts.
    bdr: width (in pixel) of border region that forbid sampling (bias foveation point to be in the center of img)
  """
  if img_tsr.ndim == 3:
    img_tsr = img_tsr.unsqueeze(0)
  H, W = img_tsr.shape[2], img_tsr.shape[3]  # if this is fixed then these two steps could be saved
  XX, YY = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32))
  deg_per_pix = 20 / math.sqrt(H ** 2 + W ** 2)
  # pixel coordinate of fixation point.
  xid, yid = pnt
  D2fov = torch.sqrt((XX - xid) ** 2 + (YY - yid) ** 2)
  D2fov_deg = D2fov * deg_per_pix
  maxecc = math.sqrt(max(xid, W - xid) ** 2 + max(yid,
                        H - yid) ** 2) * deg_per_pix
  if N_e is None:
    N_e = np.ceil((np.log(maxecc) - np.log(e_o)) / spacing).astype("int32")
    # N_e will be a numpy.int32 number
  rbf_basis = fov_rbf(D2fov_deg, spacing, e_o)
  finalimg = rbf_basis[None, None, :, :] * img_tsr  # tf.expand_dims(rbf_basis,-1)
  mask_col = [rbf_basis]
  blurimg_col = [img_tsr[0]]
  multiply_col = [(rbf_basis[None, None, :, :] * img_tsr).squeeze(0)]
  for N in range(N_e):
    rbf_basis = rbf(D2fov_deg, N, spacing, e_o=e_o)
    mean_dev = math.exp(math.log(e_o) + (N + 1) * spacing)
    kerW = kerW_coef * mean_dev / deg_per_pix
    kerSz = int(kerW * 3)
    if kerSz % 2 == 0:  # gaussian_blur2d needs a odd number sized kernel.
      kerSz += 1
    img_gsft = gaussian_blur2d(img_tsr, kernel_size=(kerSz, kerSz),
                               sigma=(kerW, kerW), border_type='reflect')
    finalimg = finalimg + rbf_basis[None, None, :, :] * img_gsft  # tf.expand_dims(rbf_basis,-1)
    mask_col.append(rbf_basis[:, :])
    blurimg_col.append(img_gsft[0])
    multiply_col.append((rbf_basis[None, None, :, :] * img_gsft).squeeze(0))

  # figh, ax = plt.subplots(figsize=[10, 10])
  # plt.imshow(finalimg.squeeze(0).permute(1, 2, 0))
  # plt.axis("off")
  # plt.show()
  # figh, ax = plt.subplots(figsize=[10, 10])
  # plt.imshow(finalimg.squeeze(0).permute(1, 2, 0))
  # plt.axis("off")
  # vis_belts(ax, img_tsr.squeeze(0).permute(1, 2, 0), pnt, kerW_coef, e_o, N_e, spacing)
  # figh.show()
  return finalimg, mask_col, blurimg_col, multiply_col