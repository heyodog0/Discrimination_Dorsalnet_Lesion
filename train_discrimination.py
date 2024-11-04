import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from discrimination_model import DiscriminationDorsalNet, create_discrimination_pairs
from sklearn.decomposition import PCA
from tqdm import tqdm

def train_discrimination_model(device='cuda', save_path='discrimination_results'):
    model = DiscriminationDorsalNet()
    model = model.to(device)
    
    scaler = torch.amp.GradScaler()
    
    batch_size = 8
    n_epochs = 20
    
    # Use BCEWithLogitsLoss instead of BCELoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    pre_lesion_acc = []
    post_lesion_acc = []
    
    print("Starting pre-lesion training...")
    for epoch in range(n_epochs):
        model.train()
        epoch_acc = []
        epoch_loss = []
        
        for batch in range(100):
            # Generate batch
            stim1_batch = []
            stim2_batch = []
            label_batch = []
            
            for _ in range(batch_size):
                s1, s2, l = create_discrimination_pairs(device)
                stim1_batch.append(s1)
                stim2_batch.append(s2)
                label_batch.append(l)
            
            stim1_batch = torch.stack(stim1_batch)
            stim2_batch = torch.stack(stim2_batch)
            label_batch = torch.stack(label_batch)
            
            # Forward pass
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(stim1_batch, stim2_batch)
                loss = criterion(outputs, label_batch)
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Calculate accuracy (apply sigmoid here since we removed it from model)
            with torch.no_grad():
                predictions = torch.sigmoid(outputs)
                acc = ((predictions > 0.5) == label_batch).float().mean()
                epoch_acc.append(acc.item())
                epoch_loss.append(loss.item())
            
            if batch % 10 == 0:
                print(f"Batch {batch}: Loss = {loss.item():.4f}, Accuracy = {acc.item():.3f}")
        
        pre_lesion_acc.append(np.mean(epoch_acc))
        print(f"Pre-lesion Epoch {epoch}: Accuracy = {pre_lesion_acc[-1]:.3f}, Loss = {np.mean(epoch_loss):.4f}")
        
    return model, pre_lesion_acc, post_lesion_acc
# def train_discrimination_model(device='cuda', save_path='discrimination_results'):
#     # Initialize model
#     model = DiscriminationDorsalNet()
#     model = model.to(device)
    
#     # Reduce memory usage
#     batch_size = 8  # Reduced from 32
#     n_epochs = 20
    
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
#     # Storage for accuracies
#     pre_lesion_acc = []
#     post_lesion_acc = []
    
#     print("Starting pre-lesion training...")
#     for epoch in range(n_epochs):
#         pbar = tqdm(range(100), desc=f'Epoch {epoch+1}/{n_epochs}')
#         model.train()
#         epoch_acc = []
        
#         for batch in pbar:
        
#         # for batch in range(100):
#             # Clear GPU cache periodically
#             if batch % 10 == 0:
#                 torch.cuda.empty_cache()
            
#             # Generate batch
#             stim1_batch = []
#             stim2_batch = []
#             label_batch = []
            
#             for _ in range(batch_size):
#                 s1, s2, l = create_discrimination_pairs(device)
#                 stim1_batch.append(s1)
#                 stim2_batch.append(s2)
#                 label_batch.append(l)
            
#             # Move to GPU efficiently
#             stim1_batch = torch.stack(stim1_batch).to(device, non_blocking=True)
#             stim2_batch = torch.stack(stim2_batch).to(device, non_blocking=True)
#             label_batch = torch.stack(label_batch).squeeze().to(device, non_blocking=True)
            
#             # Forward pass
#             outputs = model(stim1_batch, stim2_batch)
#             loss = criterion(outputs, label_batch)
            
#             # Backward pass
#             optimizer.zero_grad(set_to_none=True)  # More memory efficient
#             loss.backward()
#             optimizer.step()
            
#             # Calculate accuracy
#             with torch.no_grad():  # Save memory during evaluation
#                 acc = ((outputs > 0.5) == label_batch).float().mean()
#                 epoch_acc.append(acc.item())
            
#             # Clear unnecessary tensors
#             del outputs, loss
#             torch.cuda.empty_cache()
        
#         pre_lesion_acc.append(np.mean(epoch_acc))
#         print(f"Pre-lesion Epoch {epoch}: Accuracy = {pre_lesion_acc[-1]:.3f}")
        
#         # Save checkpoint
#         if epoch % 5 == 0:
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'accuracy': pre_lesion_acc,
#             }, f'{save_path}/checkpoint_epoch_{epoch}.pt')
    
#     # Apply V1 lesion
#     print("Applying V1 lesion...")
#     def lesion_hook(module, input, output):
#         output[:,:,7:14,7:14,:] = 0
#         return output
    
#     layer = model.layers[1][-1]  # V1 layer
#     layer.register_forward_hook(lesion_hook)
    
#     # Post-lesion training
#     print("Starting post-lesion training...")
#     for epoch in range(n_epochs):
#         model.train()
#         epoch_acc = []
        
#         for batch in range(100):
#             # Same training loop as pre-lesion
#             stim1_batch = []
#             stim2_batch = []
#             label_batch = []
            
#             for _ in range(batch_size):
#                 s1, s2, l = create_discrimination_pairs(device)
#                 stim1_batch.append(s1)
#                 stim2_batch.append(s2)
#                 label_batch.append(l)
            
#             stim1_batch = torch.stack(stim1_batch)
#             stim2_batch = torch.stack(stim2_batch)
#             label_batch = torch.stack(label_batch)
            
#             outputs = model(stim1_batch, stim2_batch)
#             loss = criterion(outputs, label_batch)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             acc = ((outputs > 0.5) == label_batch).float().mean()
#             epoch_acc.append(acc.item())

#             pbar.set_postfix({'loss': f'{loss.item():.4f}', 
#                             'acc': f'{acc.item():.3f}',
#                             'mem': f'{torch.cuda.memory_allocated()/1e9:.1f}GB'})
        
#         post_lesion_acc.append(np.mean(epoch_acc))
#         print(f"Post-lesion Epoch {epoch}: Accuracy = {post_lesion_acc[-1]:.3f}")
    
#     # Save results
#     np.save(f'{save_path}/pre_lesion_acc.npy', pre_lesion_acc)
#     np.save(f'{save_path}/post_lesion_acc.npy', post_lesion_acc)
#     torch.save(model.state_dict(), f'{save_path}/post_lesion_model.pt')
    
#     return model, pre_lesion_acc, post_lesion_acc

def plot_results(pre_acc, post_acc, save_path='discrimination_results'):
    plt.figure(figsize=(10,5))
    plt.plot(pre_acc, label='Pre-lesion')
    plt.plot(post_acc, label='Post-lesion')
    plt.xlabel('Epoch')
    plt.ylabel('Discrimination Accuracy')
    plt.title('Recovery of Discrimination Performance')
    plt.legend()
    plt.savefig(f'{save_path}/accuracy_plot.png')
    plt.show()