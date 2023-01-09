import torch.functional as F
from torch import nn


class mnist_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3), #[N, 64, 26]
            nn.ReLU(), #Leaky
            nn.Conv2d(64, 32, 3), #[N, 32, 24]
            nn.ReLU(),
            nn.Conv2d(32, 16, 3), #[N, 16, 22]
            nn.ReLU(),
            nn.Conv2d(16, 8, 3), #[N, 8, 20]
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*20*20, 128),
            nn.Dropout(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return(self.classifier(self.backbone(x)))