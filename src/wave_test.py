import torch
from pytorch_wavelets import DWTForward, DWTInverse
import pywt
import numpy as np
device = 'cuda:0'

img_dim = 28
X = torch.randn(2, 4, img_dim, img_dim)
max_level = int(np.log2(img_dim))
print(max_level)

ffm = DWTForward(J=max_level, wave='haar')
ifm = DWTInverse(wave='haar')

Yl, Yh = ffm(X)

print(Yl.shape)
res = Yl.shape[-1] ** 2

for i in range(max_level):
    print(Yh[i].shape)
    res += 3 * (Yh[i].shape[-1] ** 2)

print(res)