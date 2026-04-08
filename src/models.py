from torch import nn
import torch.nn.functional as F

class SimpleAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(SimpleAE, self).__init__()
        # Encoder: Compresses 36,601 -> 128 -> 32
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Decoder: Reconstructs 32 -> 128 -> 36,601
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid() # Use Sigmoid if data is scaled [0,1]
        )
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
    

class SimCLRGenomics(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=64):
        super().__init__()
        # The Encoder: This is what we actually want to keep
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )
        # The Projection Head: Only used during training
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z