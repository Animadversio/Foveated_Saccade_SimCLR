import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as TF
if torchvision.__version__ in ['0.8.2']:
  OLD_TV_VER = True
  import PIL
else:
  OLD_TV_VER = False
  from torchvision.transforms.transforms import InterpolationMode

import math 
import numpy as np
import numbers
from torch import Tensor
from collections.abc import Sequence
from typing import Tuple, List, Optional
import warnings
from torchvision.transforms import RandomResizedCrop

def _setup_size(size, error_msg):
  if isinstance(size, numbers.Number):
    return int(size), int(size)

  if isinstance(size, Sequence) and len(size) == 1:
    return size[0], size[0]

  if len(size) != 2:
    raise ValueError(error_msg)

  return size

def unravel_indices(
  indices: torch.LongTensor,
  shape: Tuple[int, ...],
) -> torch.LongTensor:
  r"""Converts flat indices into unraveled coordinates in a target shape.
  Args:
    indices: A tensor of (flat) indices, (*, N).
    shape: The targeted shape, (D,).
  Returns:
    The unraveled coordinates, (*, N, D).
  """
  coord = []
  for dim in reversed(shape):
    coord.append(indices % dim)
    indices = indices // dim
  coord = torch.stack(coord[::-1], dim=-1)
  return coord

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3, pad=0):
  # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
  x_coord = torch.arange(kernel_size)
  x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
  y_grid = x_grid.t()
  xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

  mean = (kernel_size - 1)/2.
  variance = sigma**2.

  # Calculate the 2-dimensional gaussian kernel which is
  # the product of two gaussian distributions for two different
  # variables (in this case called x and y)
  gaussian_kernel = (1./(2.*math.pi*variance)) *\
                    torch.exp(
                        -torch.sum((xy_grid - mean)**2., dim=-1) /\
                        (2*variance)
                    )

  # Make sure sum of values in gaussian kernel equals 1.
  gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

  # Reshape to 2d depthwise convolutional weight
  gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
  gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

  gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                kernel_size=kernel_size, groups=channels, bias=False, padding=pad)

  gaussian_filter.weight.data = gaussian_kernel
  gaussian_filter.weight.requires_grad = False
  
  return gaussian_filter

class RandomCrop_with_Density(torch.nn.Module):
  """Crop the given image at a random location determined by a density map. 
  If the image is torch Tensor, it is expected
  to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions,
  but if non-constant padding is used, the input is expected to have at most 2 leading dimensions

  Args:
    size (sequence or int): Desired output size of the crop. If size is an
      int instead of sequence like (h, w), a square crop (size, size) is
      made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    padding (int or sequence, optional): Optional padding on each border
      of the image. Default is None. If a single int is provided this
      is used to pad all borders. If sequence of length 2 is provided this is the padding
      on left/right and top/bottom respectively. If a sequence of length 4 is provided
      this is the padding for the left, top, right and bottom borders respectively.

      .. note::
        In torchscript mode padding as single int is not supported, use a sequence of
        length 1: ``[padding, ]``.
    pad_if_needed (boolean): It will pad the image if smaller than the
      desired size to avoid raising an exception. Since cropping is done
      after padding, the padding seems to be done at a random offset.
    fill (number or str or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
      length 3, it is used to fill R, G, B channels respectively.
      This value is only used when the padding_mode is constant.
      Only number is supported for torch Tensor.
      Only int or str or tuple value is supported for PIL Image.
    padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
      Default is constant.

      - constant: pads with a constant value, this value is specified with fill

      - edge: pads with the last value at the edge of the image.
        If input a 5D torch Tensor, the last 3 dimensions will be padded instead of the last 2

      - reflect: pads with reflection of image without repeating the last value on the edge.
        For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
        will result in [3, 2, 1, 2, 3, 4, 3, 2]

      - symmetric: pads with reflection of image repeating the last value on the edge.
        For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
        will result in [2, 1, 1, 2, 3, 4, 4, 3]
  """
  def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant", 
               device="cuda", avgpool=True, temperature=4, density_sigma=31,):
    super().__init__()

    self.size = tuple(_setup_size(
      size, error_msg="Please provide only two dimensions (h, w) for size."
    ))
    self.padding = padding
    self.pad_if_needed = pad_if_needed
    self.fill = fill
    self.padding_mode = padding_mode
    self.temperature = temperature
    if avgpool:
      self.salmapPooling = nn.AvgPool2d(self.size, stride=1, padding=0) # note the pooling of salmap can use a smaller window than outputsize
    else:
      self.salmapPooling = get_gaussian_kernel(self.size[0], sigma=density_sigma, channels=1).to(device)
    self.device = device

  @staticmethod
  def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Get parameters for ``crop`` for a random crop.

    Args:
      img (PIL Image or Tensor): Image to be cropped.
      output_size (tuple): Expected output size of the crop.

    Returns:
      tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    w, h = TF._get_image_size(img)
    th, tw = output_size

    if h + 1 < th or w + 1 < tw:
      raise ValueError(
        "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
      )

    if w == tw and h == th:
      return 0, 0, h, w

    i = torch.randint(0, h - th + 1, size=(1, )).item()
    j = torch.randint(0, w - tw + 1, size=(1, )).item()
    return i, j, th, tw

  def sample_crops(self, img: Tensor, output_size: Tuple[int, int], salmap):
    # density: 4d tensor with [1,1,H,W]
    th, tw = output_size
    w, h = TF._get_image_size(img)

    densitymap = torch.exp((salmap.to(self.device) - torch.logsumexp(salmap.to(self.device), (0,1,2,3), keepdim=True)) / self.temperature)
    densitymap_pad = TF.pad(densitymap, self.padding, padding_mode='constant', fill=0)
    centermap = self.salmapPooling(densitymap_pad)
    flat_idx = torch.multinomial(centermap.flatten(), 1, replacement=True).cpu()
    coord = unravel_indices(flat_idx, centermap[0, 0, :, :].shape)
    i, j = coord[0,0], coord[0,1]
    return i, j, th, tw

  def forward(self, img, density=None):
    """
    Args:
      img (PIL Image or Tensor): Image to be cropped.
      density (Tensor): same size as unpadded image. density to sample the fixation signal.  

    Returns:
      PIL Image or Tensor: Cropped image.
    """
    if self.padding is not None:
      img = TF.pad(img, self.padding, self.fill, self.padding_mode)

    width, height = TF._get_image_size(img)
    # pad the width if needed
    if self.pad_if_needed and width < self.size[1]:
      padding = [self.size[1] - width, 0]
      img = TF.pad(img, padding, self.fill, self.padding_mode)
    # pad the height if needed
    if self.pad_if_needed and height < self.size[0]:
      padding = [0, self.size[0] - height]
      img = TF.pad(img, padding, self.fill, self.padding_mode)

    if density is None:
      i, j, h, w = self.get_params(img, self.size)
    else:
      i, j, h, w = self.sample_crops(img, self.size, density)

    return TF.crop(img, i, j, h, w)


  def __repr__(self):
    return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)



def _overlap_area(i, j, h, w, height, width):
  new_i = max(0, i)
  new_j = max(0, j)
  new_i_max = min(i + h, height)
  new_j_max = min(j + w, width)
  area = (new_i_max - new_i) * (new_j_max - new_j)
  return area


class RandomResizedCrop_with_Density(torch.nn.Module):
  """Crop a random portion of image and resize it to a given size.

  If the image is torch Tensor, it is expected
  to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

  A crop of the original image is made: the crop has a random area (H * W)
  and a random aspect ratio. This crop is finally resized to the given
  size. This is popularly used to train the Inception networks.

  Args:
    size (int or sequence): expected output size of the crop, for each edge. If size is an
      int instead of sequence like (h, w), a square output size ``(size, size)`` is
      made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

      .. note::
        In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
    scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
      before resizing. The scale is defined with respect to the area of the original image.
    ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
      resizing.
    interpolation (InterpolationMode): Desired interpolation enum defined by
      :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
      If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
      ``InterpolationMode.BICUBIC`` are supported.
      For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.

  """

  def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
      interpolation=PIL.Image.BILINEAR if OLD_TV_VER else TF.InterpolationMode.BILINEAR,
      temperature=1.5, pad_if_needed=False, bdr=0):
    super().__init__()
    self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

    if not isinstance(scale, Sequence):
      raise TypeError("Scale should be a sequence")
    if not isinstance(ratio, Sequence):
      raise TypeError("Ratio should be a sequence")
    if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
      warnings.warn("Scale and ratio should be of kind (min, max)")

    # Backward compatibility with integer value
    if not OLD_TV_VER and isinstance(interpolation, int):
      warnings.warn(
        "Argument interpolation should be of type InterpolationMode instead of int. "
        "Please, use InterpolationMode enum."
      )
      interpolation = _interpolation_modes_from_int(interpolation)

    self.interpolation = interpolation
    self.scale = scale
    self.ratio = ratio
    self.temperature = temperature
    self.pad_if_needed = pad_if_needed
    self.bdr = bdr  # number of border pixels that are prohibited from being sampled.


  @staticmethod
  def get_params(
      img: Tensor, scale: List[float], ratio: List[float], density=None, bdr=0
  ) -> Tuple[int, int, int, int]:
    """Get parameters for ``crop`` for a random sized crop.

    Args:
      img (PIL Image or Tensor): Input image.
      scale (list): range of scale of the origin size cropped
      ratio (list): range of aspect ratio of the origin aspect ratio cropped

    Returns:
      tuple: params (i, j, h, w) to be passed to ``crop`` for a random
      sized crop.
    """
    width, height = TF._get_image_size(img)
    area = height * width

    log_ratio = torch.log(torch.tensor(ratio))
    for _ in range(30):
      target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
      aspect_ratio = torch.exp(
        torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
      ).item()

      w = int(round(math.sqrt(target_area * aspect_ratio)))
      h = int(round(math.sqrt(target_area / aspect_ratio)))
      if density is not None:
        if bdr > 0:
          density_mat = torch.zeros_like(density)
          density_mat[:, :, bdr:-bdr, bdr:-bdr] = density[:, :, bdr:-bdr, bdr:-bdr]
          density = density_mat
        flat_idx = torch.multinomial(density.flatten(), 1, replacement=True).cpu()
        cnt_coord = unravel_indices(flat_idx, density[0, 0, :, :].shape)
        ci, cj = cnt_coord[0, 0].item(), cnt_coord[0, 1].item()
        i, j = ci - h // 2, cj - w // 2
        over_area = _overlap_area(i, j, h, w, height, width)
        if (area * scale[1]) > over_area > \
            (area * scale[0]):
          return i, j, h, w

      else: #uniform distribution. 
        if 0 < w <= width and 0 < h <= height:
          # the only sampling part
          i = torch.randint(0, height - h + 1, size=(1,)).item()
          j = torch.randint(0, width - w + 1, size=(1,)).item()
          return i, j, h, w

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
      w = width
      h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
      h = height
      w = int(round(h * max(ratio)))
    else:  # whole image
      w = width
      h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


  def forward(self, img, logdensity=None):
    """
    Args:
      img (PIL Image or Tensor): Image to be cropped and resized.

    Returns:
      PIL Image or Tensor: Randomly cropped and resized image.
    """
    if logdensity is not None:
      # densitymap = torch.exp((logdensity - torch.logsumexp(logdensity, (1,2,3), keepdim=True)) / self.temperature)
      # 0, # old obsolete version
      densitymap = torch.exp((logdensity - logdensity.max()) / self.temperature)  # debugged version, normalized max
      # to 0
    else: 
      densitymap = None
    i, j, h, w = self.get_params(img, self.scale, self.ratio, densitymap,
                                 bdr=self.bdr)
    width, height = TF._get_image_size(img)
    if self.pad_if_needed:
      new_i, new_j = i, j
      p_left, p_top, p_right, p_bottom = 0, 0, 0, 0
      if i < 0: 
        p_top = abs(i)
        new_i = 0
      elif i + h > height:
        p_bottom = i + h - height
      if j < 0:
        p_left = abs(j)
        new_j = 0
      elif j + w > width:
        p_right = j + w - width

      img_pad = TF.pad(img, [p_left, p_top, p_right, p_bottom], fill=0, padding_mode="constant")
      return TF.resized_crop(img_pad, new_i, new_j, h, w, self.size, self.interpolation)
    else:  # no padding cut off at the border.
      i, j = max(0, i), max(0, j)
      h, w = min(h, height - i), min(w, width - j)
      # print(i, j, h, w, height, width)
      return TF.resized_crop(img, i, j, h, w, self.size, self.interpolation)


  def __repr__(self):
    interpolate_str = self.interpolation.value
    format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
    format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
    format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
    format_string += ', interpolation={0})'.format(interpolate_str)
    return format_string