#%%
!git clone https://github.com/Animadversio/FastSal
#%%
from data_aug.calc_saliency import process_stl10_fastsal
import sys
sys.path.append("/home/binxu.w/FastSal")
process_stl10_fastsal("/home/binxu.w/FastSal/weights/salicon_A.pth")