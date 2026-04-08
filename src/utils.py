import torch
import torch.nn.functional as F

def augment_genomics(x, dropout_rate=0.2, noise_std=0.05):
    # 1. Random Masking
    mask = torch.bernoulli(torch.full_like(x, 1 - dropout_rate))
    x_aug = x * mask

    # 2. Additive Gaussian Noise
    noise = torch.randn_like(x_aug) * noise_std
    return x_aug + noise

def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]
    # Combine the views: [z_i_1, z_i_2, ... z_j_1, z_j_2, ...]
    z = torch.cat([z_i, z_j], dim=0)
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

    # Scale by temperature
    sim_matrix = sim_matrix / temperature

    # Create labels for the positive pairs
    labels = torch.cat([torch.arange(batch_size) + batch_size - 1 , torch.arange(batch_size)], dim=0)
    labels = labels.to(z.device)

    # We want to ignore the diagonal (self-similarity)
    mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    logits = sim_matrix[mask].reshape(2 * batch_size, -1)

    return F.cross_entropy(logits, labels)