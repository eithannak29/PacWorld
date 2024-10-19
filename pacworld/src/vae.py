import torch
import torch.nn as nn
import torch.nn.functional as F

# This VAE (Variational Autoencoder) class is used to encode and decode observations 
# for the Pacman environment, compressing high-dimensional inputs into a latent space.

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2) # 2 for mean and variance respresentations 
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, log_var = x.chunk(2, dim=-1)
        return mean, log_var
      
    def decode(self, z):
      return self.decoder(z)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decoder(z), mean, log_var
      
    def loss_function(self, recon_x, x, mean, log_var):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return recon_loss + kl_div
      


# Test the VAE model
# input_dim = 28 * 28  
# latent_dim = 32

# vae = VAE(input_dim=input_dim, latent_dim=latent_dim)

# x = torch.randn(1, input_dim)

# output, mean, log_var = vae(x)
