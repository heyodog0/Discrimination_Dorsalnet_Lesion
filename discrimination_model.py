import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dorsalnet import DorsalNet_DoG
from test_util import create_drifting_gratings

class DiscriminationDorsalNet(DorsalNet_DoG):
    def __init__(self, symmetric=True, nfeats=32):
        super().__init__(symmetric, nfeats)
        
        # Add discrimination head for comparison tasks
        self.discrimination_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(nfeats, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Add analysis mode flag
        self._analysis_mode = False
        self._lesion_mask = None

    def apply_lesion(self, lesion_type='complete'):
        """Apply permanent V1 lesion by directly modifying weights"""
        if lesion_type == 'none':
            return
                
        with torch.no_grad():
            if lesion_type == 'complete':
                # Zero out all V1 weights
                self.s1.conv1.weight.data.zero_()  # Set weights to zero
                # Freeze the weights
                self.s1.conv1.weight.requires_grad = False
                # Also zero out and freeze dog params if they exist
                if hasattr(self.s1, 'dog'):
                    self.s1.dog.weight1.data.zero_()
                    self.s1.dog.weight2.data.zero_()
                    self.s1.dog.weight1.requires_grad = False
                    self.s1.dog.weight2.requires_grad = False
                    
            print(f"Applied {lesion_type} lesion to V1")
            print(f"V1 weight mean: {self.s1.conv1.weight.data.abs().mean().item():.4f}")

    def forward(self, x1, x2=None):
        """Forward pass with proper batch size handling"""
        if self._analysis_mode or x2 is None:
            # Analysis mode: just process single input through base network
            return super().forward(x1)
        
        # Discrimination mode: compare two inputs
        with torch.amp.autocast(device_type='cuda'):
            # Process each input
            feat1 = super().forward(x1)
            feat2 = super().forward(x2)
            
            # Compare features
            diff = torch.abs(feat1 - feat2)
            out = self.discrimination_head(diff)
            return out.squeeze()

    def set_analysis_mode(self, mode=True):
        """Switch between analysis and discrimination modes"""
        self._analysis_mode = mode
        return self

    def is_analysis_mode(self):
        """Check if model is in analysis mode"""
        return self._analysis_mode

def create_discrimination_pairs(device='cuda', performance=0.5, batch_size=None):
    """Create discrimination pairs with consistent batch handling"""
    n_directions = 16
    
    if batch_size is None:
        # Single pair
        reference_angle = np.random.randint(0, n_directions)
        difficulty = max(0, min(1, (performance - 0.5) * 2))
        
        if np.random.random() < 0.5:
            test_angle = reference_angle
            label = 0.0
            noise_level = 0.005 + (difficulty * 0.015)
        else:
            min_diff = max(1, int(4 * (1 - difficulty)))
            angle_diff = np.random.choice([-min_diff, min_diff])
            test_angle = (reference_angle + angle_diff) % n_directions
            label = 1.0
            noise_level = 0.003 + (difficulty * 0.007)
        
        # Generate single pair
        stim1 = torch.tensor(create_drifting_gratings(ndirections=n_directions)[reference_angle],
                           device=device, dtype=torch.float32)
        stim2 = torch.tensor(create_drifting_gratings(ndirections=n_directions)[test_angle],
                           device=device, dtype=torch.float32)
        
        # Add noise
        stim1 += torch.randn_like(stim1) * noise_level
        stim2 += torch.randn_like(stim2) * noise_level
        
        return stim1, stim2, torch.tensor(label, device=device, dtype=torch.float32)
    
    else:
        # Generate batch
        stim1_batch = []
        stim2_batch = []
        label_batch = []
        
        for _ in range(batch_size):
            s1, s2, l = create_discrimination_pairs(device, performance)
            stim1_batch.append(s1)
            stim2_batch.append(s2)
            label_batch.append(l)
        
        return (torch.stack(stim1_batch),
                torch.stack(stim2_batch),
                torch.stack(label_batch))