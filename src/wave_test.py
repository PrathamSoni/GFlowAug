import torch
from pytorch_wavelets import DWTForward, DWTInverse
import pywt
import numpy as np
device = 'cuda:0'

def stack_wavedec(yl, yh):
    n, c = yl.shape[0], yl.shape[1]
    lis = [yl.view(n, c, -1)] + [e.view(n, c, -1) for e in yh]
    y_sizes = [e.shape[-1] for e in lis]
    stack = torch.cat(lis, dim=-1)
    return stack, y_sizes

def unstack_wavedec(y, y_sizes, wave_sizes):
    ys = torch.split(y, y_sizes, dim=-1)
    ys = [e.view(wave_sizes[i]) for i, e in enumerate(ys)]
    return ys[0], ys[1:]

img_dim = 28
X = torch.randn(2, 4, img_dim, img_dim)
max_level = int(np.log2(img_dim))
print(max_level)
print(X)

ffm = DWTForward(J=max_level, wave='haar')
ifm = DWTInverse(wave='haar')

yl, yh = ffm(X)
wave_sizes = [yl.shape] + [e.shape for e in yh]
y, y_sizes = stack_wavedec(yl, yh)
ylp, yhp = unstack_wavedec(y, y_sizes, wave_sizes)
xp = ifm((ylp, yhp))
diff = X - xp
print(torch.mean(torch.abs(diff)))

# print(Yl.shape)
# res = Yl.shape[-1] ** 2

# for i in range(max_level):
#     print(Yh[i].shape)
#     res += 3 * (Yh[i].shape[-1] ** 2)

# print(res)