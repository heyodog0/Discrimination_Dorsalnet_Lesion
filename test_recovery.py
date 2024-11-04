import torch
import torch.nn as nn
import torch.optim as optim
from discrimination_model import DiscriminationDorsalNet
from test_util import create_drifting_gratings

def test_recovery():
    print("Testing recovery condition...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = DiscriminationDorsalNet().to(device)
    
    # Debug: Print model structure
    print("\nModel structure:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            print(f"Found Conv3d in {name}")
    
    # Print initial state
    print("\nInitial state:")
    print(f"V1 weight mean: {model.s1.conv1.weight.data.abs().mean().item():.6f}")
    
    # Apply lesion
    print("\nApplying lesion...")
    model.apply_lesion(lesion_type='complete')
    
    # Verify lesion
    print("\nPost-lesion state:")
    print(f"V1 weight mean: {model.s1.conv1.weight.data.abs().mean().item():.6f}")
    print(f"V1 requires_grad: {model.s1.conv1.weight.requires_grad}")
    
    # Try a few training steps
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler()
    
    print("\nTrying a few training steps...")
    for i in range(3):
        # Generate a small batch
        stim1 = torch.tensor(create_drifting_gratings(ndirections=16)[0], device=device, dtype=torch.float)
        stim2 = torch.tensor(create_drifting_gratings(ndirections=16)[1], device=device, dtype=torch.float)
        stim1 = stim1.unsqueeze(0).repeat(8, 1, 1, 1, 1)  # Make batch of 8
        stim2 = stim2.unsqueeze(0).repeat(8, 1, 1, 1, 1)
        label = torch.ones(8, device=device)
        
        # Forward pass
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(stim1, stim2)
            loss = criterion(outputs, label)
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Check weights after training step
        print(f"\nAfter training step {i+1}:")
        print(f"V1 weight mean: {model.s1.conv1.weight.data.abs().mean().item():.6f}")

if __name__ == "__main__":
    test_recovery()