import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsistencyLoss(nn.Module):
    """
    Computes consistency loss between embeddings of two augmented views.
    Ensures that the model learns domain-invariant representations.
    """
    def __init__(self):
        super().__init__()

    def forward(self, z1, z2):
        # Normalize embeddings to unit sphere
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Mean Squared Error between normalized embeddings 
        # (Equivalent to 2 - 2 * Cosine Similarity)
        loss = 2 - 2 * (z1 * z2).sum(dim=-1)
        
        return loss.mean()
