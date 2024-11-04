import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dorsalnet import DorsalNet_DoG  # Import existing model
from test_util import create_drifting_gratings

class DiscriminationDorsalNet(DorsalNet_DoG):
    def __init__(self, symmetric=True, nfeats=32):
        super().__init__(symmetric, nfeats)
        
        # Add discrimination head
        self.discrimination_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(nfeats, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # Process first stimulus
        feat1 = super().forward(x1)
        
        # Process second stimulus
        feat2 = super().forward(x2)
        
        # Compare features and make decision
        diff = torch.abs(feat1 - feat2)
        out = self.discrimination_head(diff)
        return out

def create_discrimination_pairs(device='cpu'):
    # Create pairs of drifting gratings with different orientations
    n_directions = 16
    reference_angle = np.random.randint(0, n_directions)
    
    # Reference stimulus
    stim1 = torch.tensor(create_drifting_gratings(ndirections=n_directions)).to(device=device, dtype=torch.float)
    
    # Test stimulus with different orientation
    angle_diff = np.random.choice([-2, -1, 1, 2])
    test_angle = (reference_angle + angle_diff) % n_directions
    stim2 = torch.tensor(create_drifting_gratings(ndirections=n_directions)).to(device=device, dtype=torch.float)
    
    # Label: 1 if different orientation, 0 if same
    label = torch.tensor([1.0 if angle_diff != 0 else 0.0], device=device)
    
    return stim1, stim2, label