import torch
import torch.nn as nn

class LatentClassifier(nn.Module):
    def __init__(self, latent_dim=32*7*7, num_classes=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, latent):
        latent = latent.view(latent.size(0), -1)
        return self.fc(latent)
