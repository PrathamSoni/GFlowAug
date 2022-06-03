from torchvision.datasets import MNIST
from torchvision.transforms import *
import matplotlib.pyplot as plt
import torch
import pywt
from models import ImageGFN
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb

# CHANGE API KEY
wandb.login(key='e9d0f0abd4a0b92aa26694bdecd67aa7d57b76d6')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = MNIST(root='./datasets/MNIST', train=True, download=True, transform=ToTensor())
train_loader = DataLoader(data, batch_size=2, num_workers=4)

model = ImageGFN(n_channels=1, img_dim=28, output_dim=8, num_gaussians=8, lr=8*1e-3)
wandb_logger = WandbLogger(project="gflow_images", log_model=True)
trainer = pl.Trainer(
    overfit_batches=10,
    max_epochs=100,
    logger=wandb_logger,
    accelerator="gpu",
    auto_lr_find='lr',
    gradient_clip_val=0.9,
    gradient_clip_algorithm="norm"
)
trainer.fit(model=model, train_dataloaders=train_loader)
x = model().cpu().numpy()
plt.imshow(x[0])
plt.savefig('./gen.png')
#
# w = pywt.Wavelet('haar')
# x = [9, 7, 3, 5, 6, 10, 2, 6, 4, 1]
# y = pywt.wavedec(x, w)
# print(y)




