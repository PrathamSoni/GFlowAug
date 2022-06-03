from models import MixtureModel
import torch

n = 1
k = 8
t = 8
mu = torch.rand((n, k, t), device='cuda:0') * 5 - 10
sigma = torch.rand((n, k, t, t), device='cuda:0') * 0.5#torch.eye(t) # * torch.randint(1, 3, (n, k, t, t))
sigma = torch.matmul(sigma, torch.transpose(sigma, -2, -1)) / 2
pi = torch.softmax(torch.rand(n, k, device='cuda:0'), dim=-1)

model = MixtureModel(mu, sigma, pi)
x = 0.9 * torch.ones((n, t), device='cuda:0')
print(x.shape)
print(model.density(x, per_sample=True))