import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import pywt

torch.manual_seed(241)

class FeatureExtractor(nn.Module):
    def __init__(self, img_dim, input_channels, output_channels, t, k, act=nn.LeakyReLU(), pool=nn.MaxPool2d(kernel_size=2, stride=2)):
        super().__init__()
        final_hw = img_dim // 16
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=3, padding=1),
            act,
            pool,
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2),
            act,
            pool,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            act,
            pool,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            act,
            pool,
            nn.Flatten(),
            nn.Linear(64 * final_hw * final_hw, output_channels * k * (t * t + t + 1))
        )
        self.k = k
        self.t = t
        self.n_channels = output_channels

    def forward(self, x):
        z = self.model(x)
        mu = torch.reshape(z[..., :self.k * self.t], (-1, self.k, self.t))
        sigma = torch.reshape(z[..., self.k * self.t:-self.k], (-1, self.k, self.t, self.t))
        sigma = torch.matmul(torch.transpose(sigma, -2, -1), sigma) / 2
        pi = torch.reshape(z[..., -self.k:], (-1, self.k))
        pi = torch.softmax(pi, dim=-1)
        return mu, sigma, pi

class MixtureModel(nn.Module):
    def __init__(self, mu, sigma, pi):
        super().__init__()
        self.register_buffer('mu', mu) # n * k * c * t
        self.register_buffer('sigma', sigma) # n * k * c * t * t
        self.register_buffer('pi', pi) # n * k * c
        self.dist = MultivariateNormal(loc=mu, covariance_matrix=sigma)

    def density(self, x, per_sample=False):
        """
        :param x: n * t
        :return:
        """
        x = torch.atanh(0.99 * x)
        weighted_log_prob = torch.log(self.get_buffer('pi')) + self.dist.log_prob(x) # n * k
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=-1) # n
        return per_sample_score if per_sample else torch.mean(per_sample_score)

    def forward(self, batch_size):
        selected_gaussians = torch.multinomial(self.get_buffer('pi'), batch_size, replacement=True)
        selected_mu = self.get_buffer('mu')[:, selected_gaussians]
        selected_sigma = self.get_buffer('sigma')[:, selected_gaussians]
        selected_dist = MultivariateNormal(loc=selected_mu, covariance_matrix=selected_sigma)
        samples = selected_dist.sample((batch_size,))
        return samples

class ImageGFN(pl.LightningModule):
    def __init__(self, n_channels, img_dim, output_dim, num_gaussians, lr=1e-3, uniform_PB=True):
        super().__init__()
        self.uniform_PB = uniform_PB
        self.step_size = output_dim
        self.img_dim = img_dim
        self.k = num_gaussians
        self.n_channels = n_channels
        self.feature_model = FeatureExtractor(img_dim, n_channels + 2, n_channels, self.step_size, self.k)
        self.lr = lr
        self.save_hyperparameters()

    def forward(self):
        """
        Sampling
        :return:
        """
        out = torch.zeros((1, self.n_channels, self.img_dim, self.img_dim), device=self.device)
        vis = torch.zeros((1, 1, self.img_dim, self.img_dim), device=self.device)
        indices = torch.nonzero(vis == 0, as_tuple=True)
        order = torch.randperm(self.img_dim * self.img_dim, device=self.device)

        if len(order) % self.step_size != 0:
            pad_length = self.step_size - (len(order) % self.step_size) # in case we can't evenly divide, add from the beginning again
            order = torch.cat([order, order[:pad_length]])

        order = order.view((-1, self.step_size))

        for i in range(order.shape[0]):
            selected_indices = [dim[order[i]] for dim in indices]
            take = torch.zeros(vis.shape, device=self.device)
            take[selected_indices] = 1
            img = torch.cat([out, vis, take], dim=1)
            mu, sigma, pi = self.feature_model(img)
            sigma *= torch.eye(self.step_size, device=self.device)
            gmm = MixtureModel(mu, sigma, pi)
            out[selected_indices] = gmm(1).squeeze(0)
            vis[selected_indices] = 1

        return (out.squeeze(0) + 1) * 0.5

    def likelihood(self, x, batch_idx):
        x = x * 2 - 1

        batch_size = x.shape[0]
        eps = self.step_size / (self.img_dim * self.img_dim)
        p = torch.rand((batch_size, 1, 1, 1), device=self.device) * (1 - eps) + eps
        vis = (torch.rand((batch_size, 1, self.img_dim, self.img_dim), device=self.device) > p).float()

        x_hat = torch.masked_fill(x, vis == 0, -1)
        max_steps = (2 * self.img_dim * self.img_dim) // self.step_size
        ll = 0

        for i in range(max_steps):
            num_left = int(torch.sum(vis == 0))

            if num_left == 0:
                break

            selection = torch.randperm(num_left, device=self.device)[:self.step_size]
            indices_left = torch.nonzero(vis == 0, as_tuple=True)
            selected_indices = [row[selection] for row in indices_left]

            if num_left < self.step_size:
                rem = self.step_size - num_left
                indices_fill = torch.nonzero(vis, as_tuple=True)
                fill_selection = torch.randperm(len(indices_fill[0]), device=self.device)[:rem]
                indices_fill = [dim[fill_selection] for dim in indices_fill]
                selected_indices = [torch.cat([orig, fill]) for orig, fill in zip(selected_indices, indices_fill)]

            take = torch.zeros(vis.shape, device=self.device)
            take[selected_indices] = 1

            inp = torch.cat([x_hat, vis, take], dim=1)
            mu, sigma, pi = self.feature_model(inp)  # k * t, k * t * t, k
            sigma *= torch.eye(self.step_size, device=self.device)  # Diagonal for now - think of workaround later
            gmm = MixtureModel(mu, sigma, pi)

            x_true = x[selected_indices]  # ground truth
            ll += gmm.density(x_true)

            x_hat[selected_indices] = x_true
            vis[selected_indices] = 1

        return -ll

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.likelihood(x, batch_idx)

        if loss is None:
            return None

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.likelihood(x)
        self.log('val_loss', loss)

    def on_epoch_end(self) -> None:
        self.logger.log_image("sample_image", [self.forward().cpu().numpy()])

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=2)
        return {
            "optimizer": optim,
            "lr_scheduler": scheduler,
            "monitor": "train_loss"
        }

        # loss_TB = torch.zeros(batch_size)
        # dones = torch.full(batch_size, False)
        # states = torch.zeros((batch_size, self.h_dim))
        # visited = torch.full((batch_size, self.h_dim), False)
        # actions = None
        #
        # max_steps = 1e8
        #
        # for i in range(max_steps):
        #     if torch.all(dones):
        #         break
        #
        #     non_terminal_states = states[~dones]
        #     current_batch_size = non_terminal_states.shape[0]
        #     logits = self.model(non_terminal_states)
        #     PB_logits, PF_logits = torch.chunk(logits, 2)
        #
        #     # Backward Policy
        #     if self.uniform_PB:
        #         PB_logits *= 0
        #
        #     log_PB = -torch.log_softmax(torch.masked_fill(PB_logits, non_terminal_states == 0, -torch.inf), dim=1)
        #
        #     if actions is not None:
        #         loss_TB[~dones] += torch.gather(log_PB, dim=1, index=actions[actions != self.h_dim].unsqueeze(1)).squeeze(1)
        #
        #     # Forward Policy
        #     log_PF = -torch.log_softmax(torch.masked_fill(PF_logits, visited, -torch.inf), dim=1)
        #     tau = 1
        #     sample_probs = torch.softmax(log_PF / tau, dim=-1)
        #     actions = torch.multinomial(sample_probs, 1)
        #     loss_TB[~dones] += torch.gather(log_PF, dim=1, index=actions).squeeze(1)
        #
        #     # Terminate states
        #     terminates = None
        #
        #     dones[~dones] |= terminates
