# complete_adaptive_experiment.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Tuple

# Ensure all imports are available
try:
    from discrimination_model import DiscriminationDorsalNet
    from test_util import create_drifting_gratings
except ImportError as e:
    print(f"Error importing required modules: {e}")
    raise

class AdaptiveDifficulty:
    """Manages adaptive difficulty during training with smoother transitions"""
    def __init__(self,
                 initial_difficulty: float = 0.2,
                 min_difficulty: float = 0.1,
                 max_difficulty: float = 0.9,  # Reduced max difficulty
                 window_size: int = 200,       # Increased window size
                 target_accuracy: float = 0.75,
                 adjustment_rate: float = 0.01):  # Reduced adjustment rate
        self.difficulty = initial_difficulty
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.window_size = window_size
        self.target_accuracy = target_accuracy
        self.adjustment_rate = adjustment_rate
        
        self.recent_accuracy = []
        self.difficulty_history = []
        self.accuracy_threshold = 0.1  # Tolerance band around target accuracy
        
    def update(self, batch_accuracy: float) -> float:
        """Update difficulty based on recent performance with smoother transitions"""
        self.recent_accuracy.append(batch_accuracy)
        
        if len(self.recent_accuracy) >= self.window_size:
            current_acc = np.mean(self.recent_accuracy[-self.window_size:])
            acc_diff = current_acc - self.target_accuracy
            
        
            if abs(acc_diff) > self.accuracy_threshold:
                # Adjustment proportional to distance from target
                adjustment = self.adjustment_rate * acc_diff
                
                self.difficulty = np.clip(
                    self.difficulty + adjustment,
                    self.min_difficulty,
                    self.max_difficulty
                )
            
            self.recent_accuracy = self.recent_accuracy[-self.window_size:]
            
        self.difficulty_history.append(self.difficulty)
        return self.difficulty

def create_adaptive_discrimination_pairs(device='cuda', 
                                      difficulty: float = 1.0,
                                      min_angle_diff: float = 5.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create stimulus pairs with smoother difficulty scaling, optimized for GPU"""
    n_directions = 16
    reference_angle = np.random.randint(0, n_directions)
    
    if np.random.random() < 0.5:
        # Same orientation trials
        test_angle = reference_angle
        label = 0.0
        noise_level = 0.01 + 0.03 * difficulty**2
    else:
        # Different orientation trials
        max_angle_diff = 45  # degrees
        min_diff = min_angle_diff / (360/n_directions)
        max_diff = max_angle_diff / (360/n_directions)
        
        angle_diff = int(max(min_diff, max_diff * (1 - difficulty**1.5)))
        angle_diff *= np.random.choice([-1, 1])
        test_angle = (reference_angle + angle_diff) % n_directions
        label = 1.0
        noise_level = 0.01 + 0.01 * difficulty

    # Create gratings and move to GPU immediately
    stim1 = torch.tensor(
        create_drifting_gratings(ndirections=n_directions)[reference_angle],
        device=device,
        dtype=torch.float
    )
    stim2 = torch.tensor(
        create_drifting_gratings(ndirections=n_directions)[test_angle],
        device=device,
        dtype=torch.float
    )
    
    # Generate noise directly on GPU
    noise1 = torch.randn_like(stim1, device=device) * noise_level
    noise2 = torch.randn_like(stim2, device=device) * noise_level
    
    # Add noise on GPU
    stim1 = stim1 + noise1
    stim2 = stim2 + noise2
    
    return (stim1, stim2, torch.tensor(label, device=device))


class LesionExperimentModel(DiscriminationDorsalNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only for condition tracking and analysis
        self.lesion_type = None
        self.hooks = {}
        self.register_hooks()
        
    def register_hooks(self):
        """Register hooks for layer analysis only"""
        def get_activation(name):
            def hook(model, input, output):
                self.hooks[name] = output.detach()
            return hook
        
        # These hooks are only for analysis, not for lesioning
        self.s1.conv1.register_forward_hook(get_activation('layer01'))
        self.res1.register_forward_hook(get_activation('layer03'))
        self.res3.register_forward_hook(get_activation('layer04'))

        
    def apply_lesion(self, lesion_type='complete', lesion_size=0.5):
        """Apply permanent V1 lesion by directly modifying weights"""
        self.lesion_type = lesion_type
        
        if lesion_type == 'none':
            return
            
        with torch.no_grad():
            if lesion_type == 'complete':
                # Zero out all V1 weights
                self.s1.conv1.weight.data.zero_()
                # Freeze the weights
                self.s1.conv1.weight.requires_grad = False
                
            elif lesion_type == 'partial':
                # Zero out portion of V1 weights
                n_channels = int(self.s1.conv1.weight.size(0) * lesion_size)
                self.s1.conv1.weight.data[:n_channels].zero_()
                # Freeze the lesioned weights
                self.s1.conv1.weight.requires_grad = False
                
        print(f"Applied {lesion_type} lesion to V1")
        print(f"V1 weight mean: {self.s1.conv1.weight.data.mean().item():.4f}")
        print(f"V1 weight std: {self.s1.conv1.weight.data.std().item():.4f}")
            
    def forward(self, x1, x2=None):
        """Forward pass - no special handling needed as weights are permanently modified"""
        return super().forward(x1, x2)

def train_adaptive_model_with_verification(model, device, condition="", n_epochs=20, batch_size=8, save_dir='training_results'):
    """Train model with adaptive difficulty and continuous V1 weight verification"""
    os.makedirs(save_dir, exist_ok=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Only use GradScaler if CUDA is available
    use_amp = device == 'cuda'
    if use_amp:
        scaler = torch.amp.GradScaler()
    
    difficulty_manager = AdaptiveDifficulty(
        initial_difficulty=0.2,
        target_accuracy=0.75
    )
    
    metrics = {
        'train_acc': [],
        'train_loss': [],
        'difficulty': [],
        'epoch_times': [],
        'v1_stats': []  # Track V1 weight statistics
    }
    
    for epoch in range(n_epochs):
        epoch_start = datetime.now()
        model.train()
        epoch_acc = []
        epoch_loss = []
        epoch_diff = []
        
        # Verify V1 state at start of epoch
        print(f"\nEpoch {epoch} V1 verification:")
        v1_stats = verify_v1_state(model, f"{condition} - Epoch {epoch} start")
        metrics['v1_stats'].append(v1_stats)
        
        pbar = tqdm(range(100), desc=f'Epoch {epoch+1}/{n_epochs}')
        for batch in pbar:
            current_difficulty = difficulty_manager.difficulty
            
            # Generate and process batch
            stim1_batch, stim2_batch, label_batch = [], [], []
            for _ in range(batch_size):
                s1, s2, l = create_adaptive_discrimination_pairs(device, difficulty=current_difficulty)
                stim1_batch.append(s1)
                stim2_batch.append(s2)
                label_batch.append(l)
            
            stim1_batch = torch.stack(stim1_batch)
            stim2_batch = torch.stack(stim2_batch)
            label_batch = torch.stack(label_batch)
            
            # Training step with appropriate device handling
            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(stim1_batch, stim2_batch)
                    loss = criterion(outputs, label_batch)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(stim1_batch, stim2_batch)
                loss = criterion(outputs, label_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                predictions = torch.sigmoid(outputs)
                acc = ((predictions > 0.5) == label_batch).float().mean()
                epoch_acc.append(acc.item())
                epoch_loss.append(loss.item())
                
                new_difficulty = difficulty_manager.update(acc.item())
                epoch_diff.append(new_difficulty)
            
            # Verify V1 weights periodically during training
            if batch % 25 == 0:  # Check every 25 batches
                v1_stats = verify_v1_state(model, f"{condition} - Epoch {epoch} Batch {batch}")
                metrics['v1_stats'].append(v1_stats)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc.item():.3f}',
                'diff': f'{new_difficulty:.2f}'
            })
        
        # Update and save metrics
        metrics['train_acc'].append(np.mean(epoch_acc))
        metrics['train_loss'].append(np.mean(epoch_loss))
        metrics['difficulty'].append(np.mean(epoch_diff))
        metrics['epoch_times'].append((datetime.now() - epoch_start).total_seconds())
        
        print(f"Epoch {epoch}: Acc={metrics['train_acc'][-1]:.3f}, "
              f"Loss={metrics['train_loss'][-1]:.4f}, "
              f"Diff={metrics['difficulty'][-1]:.2f}")
        
        # Save checkpoints with V1 verification data
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'difficulty_history': difficulty_manager.difficulty_history,
                'v1_stats': metrics['v1_stats']
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    return metrics

# def train_adaptive_model(model, device, n_epochs=20, batch_size=8, save_dir='training_results'):
#     """Train model with adaptive difficulty"""
#     os.makedirs(save_dir, exist_ok=True)
    
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
#     # Only use GradScaler if CUDA is available
#     use_amp = device == 'cuda'
#     if use_amp:
#         scaler = torch.amp.GradScaler()
    
#     difficulty_manager = AdaptiveDifficulty(
#         initial_difficulty=0.2,
#         target_accuracy=0.75
#     )
    
#     metrics = {
#         'train_acc': [],
#         'train_loss': [],
#         'difficulty': [],
#         'epoch_times': []
#     }
    
#     for epoch in range(n_epochs):
#         epoch_start = datetime.now()
#         model.train()
#         epoch_acc = []
#         epoch_loss = []
#         epoch_diff = []
        
#         pbar = tqdm(range(100), desc=f'Epoch {epoch+1}/{n_epochs}')
#         for batch in pbar:
#             current_difficulty = difficulty_manager.difficulty
            
#             # Generate batch
#             stim1_batch = []
#             stim2_batch = []
#             label_batch = []
            
#             for _ in range(batch_size):
#                 s1, s2, l = create_adaptive_discrimination_pairs(
#                     device, 
#                     difficulty=current_difficulty
#                 )
#                 stim1_batch.append(s1)
#                 stim2_batch.append(s2)
#                 label_batch.append(l)
            
#             stim1_batch = torch.stack(stim1_batch)
#             stim2_batch = torch.stack(stim2_batch)
#             label_batch = torch.stack(label_batch)
            
#             # Different handling for CPU and GPU
#             if use_amp:
#                 with torch.amp.autocast(device_type='cuda'):
#                     outputs = model(stim1_batch, stim2_batch)
#                     loss = criterion(outputs, label_batch)
                
#                 optimizer.zero_grad()
#                 scaler.scale(loss).backward()
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 # Regular CPU training
#                 outputs = model(stim1_batch, stim2_batch)
#                 loss = criterion(outputs, label_batch)
                
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
            
#             with torch.no_grad():
#                 predictions = torch.sigmoid(outputs)
#                 acc = ((predictions > 0.5) == label_batch).float().mean()
#                 epoch_acc.append(acc.item())
#                 epoch_loss.append(loss.item())
                
#                 new_difficulty = difficulty_manager.update(acc.item())
#                 epoch_diff.append(new_difficulty)
            
#             pbar.set_postfix({
#                 'loss': f'{loss.item():.4f}',
#                 'acc': f'{acc.item():.3f}',
#                 'diff': f'{new_difficulty:.2f}'
#             })
        
#         metrics['train_acc'].append(np.mean(epoch_acc))
#         metrics['train_loss'].append(np.mean(epoch_loss))
#         metrics['difficulty'].append(np.mean(epoch_diff))
#         metrics['epoch_times'].append((datetime.now() - epoch_start).total_seconds())
        
#         print(f"Epoch {epoch}: Acc={metrics['train_acc'][-1]:.3f}, "
#               f"Loss={metrics['train_loss'][-1]:.4f}, "
#               f"Diff={metrics['difficulty'][-1]:.2f}")
        
#         # Save checkpoints
#         if (epoch + 1) % 5 == 0:
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'metrics': metrics,
#                 'difficulty_history': difficulty_manager.difficulty_history
#             }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
#     return metrics

# def train_adaptive_model(model, device, n_epochs=20, batch_size=8, save_dir='training_results'):
#     """Train model with adaptive difficulty"""
#     os.makedirs(save_dir, exist_ok=True)
    
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
#     scaler = torch.amp.GradScaler()
    
#     difficulty_manager = AdaptiveDifficulty(
#         initial_difficulty=0.2,
#         target_accuracy=0.75
#     )
    
#     metrics = {
#         'train_acc': [],
#         'train_loss': [],
#         'difficulty': [],
#         'epoch_times': []
#     }
    
#     for epoch in range(n_epochs):
#         epoch_start = datetime.now()
#         model.train()
#         epoch_acc = []
#         epoch_loss = []
#         epoch_diff = []
        
#         pbar = tqdm(range(100), desc=f'Epoch {epoch+1}/{n_epochs}')
#         for batch in pbar:
#             current_difficulty = difficulty_manager.difficulty
            
#             # Generate batch
#             stim1_batch = []
#             stim2_batch = []
#             label_batch = []
            
#             for _ in range(batch_size):
#                 s1, s2, l = create_adaptive_discrimination_pairs(
#                     device, 
#                     difficulty=current_difficulty
#                 )
#                 stim1_batch.append(s1)
#                 stim2_batch.append(s2)
#                 label_batch.append(l)
            
#             stim1_batch = torch.stack(stim1_batch)
#             stim2_batch = torch.stack(stim2_batch)
#             label_batch = torch.stack(label_batch)
            
#             with torch.amp.autocast(device_type='cuda'):
#                 outputs = model(stim1_batch, stim2_batch)
#                 loss = criterion(outputs, label_batch)
            
#             optimizer.zero_grad()
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
            
#             with torch.no_grad():
#                 predictions = torch.sigmoid(outputs)
#                 acc = ((predictions > 0.5) == label_batch).float().mean()
#                 epoch_acc.append(acc.item())
#                 epoch_loss.append(loss.item())
                
#                 new_difficulty = difficulty_manager.update(acc.item())
#                 epoch_diff.append(new_difficulty)
            
#             pbar.set_postfix({
#                 'loss': f'{loss.item():.4f}',
#                 'acc': f'{acc.item():.3f}',
#                 'diff': f'{new_difficulty:.2f}'
#             })
        
#         metrics['train_acc'].append(np.mean(epoch_acc))
#         metrics['train_loss'].append(np.mean(epoch_loss))
#         metrics['difficulty'].append(np.mean(epoch_diff))
#         metrics['epoch_times'].append((datetime.now() - epoch_start).total_seconds())
        
#         print(f"Epoch {epoch}: Acc={metrics['train_acc'][-1]:.3f}, "
#               f"Loss={metrics['train_loss'][-1]:.4f}, "
#               f"Diff={metrics['difficulty'][-1]:.2f}")
        
#         # Save checkpoints
#         if (epoch + 1) % 5 == 0:
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'metrics': metrics,
#                 'difficulty_history': difficulty_manager.difficulty_history
#             }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
#     return metrics

def verify_v1_state(model, stage_name=""):
    """Verify V1 weights and state"""
    v1_weights = model.s1.conv1.weight.data
    weight_stats = {
        'mean': v1_weights.abs().mean().item(),
        'max': v1_weights.abs().max().item(),
        'min': v1_weights.abs().min().item(),
        'requires_grad': model.s1.conv1.weight.requires_grad,
        'nonzero_count': torch.count_nonzero(v1_weights).item()
    }
    
    print(f"\nV1 State Check - {stage_name}")
    print("-" * 50)
    print(f"Mean absolute weight: {weight_stats['mean']:.8f}")
    print(f"Max absolute weight: {weight_stats['max']:.8f}")
    print(f"Min absolute weight: {weight_stats['min']:.8f}")
    print(f"Requires gradient: {weight_stats['requires_grad']}")
    print(f"Number of non-zero weights: {weight_stats['nonzero_count']}")
    
    if hasattr(model.s1, 'dog'):
        dog_weights = model.s1.dog.weight1.data
        print("\nDoG weights:")
        print(f"Mean: {dog_weights.abs().mean().item():.8f}")
        print(f"Requires gradient: {model.s1.dog.weight1.requires_grad}")
    
    return weight_stats

def verify_during_training(model, epoch):
    """Verify V1 weights during training"""
    stats = verify_v1_state(model, f"Training Epoch {epoch}")
    if stats['nonzero_count'] > 0:
        print("\nWARNING: Non-zero weights detected in V1 during training!")
        return False
    return True

def analyze_model_layers(model, hooks, stimuli: torch.Tensor) -> Dict[str, Dict[str, np.ndarray]]:
    """Analyze layer properties"""
    layers = {'layer01': 'V1', 'layer03': 'MT', 'layer04': 'MST'}
    results = {}
    
    for layer_key, layer_name in layers.items():
        output = model(stimuli)
        resp = hooks[layer_key]
        resp_shape = resp.shape
        num_filters = resp_shape[1]
        
        OSI = np.zeros(num_filters)
        DSI = np.zeros(num_filters)
        preferred = np.zeros(num_filters)
        response = np.zeros((num_filters, 16))
        entropy = np.zeros(num_filters)
        
        for filt in range(num_filters):
            mean_resp = np.mean(resp[:, filt, :, :, :].detach().cpu().numpy(), (1,2,3))
            mean_resp = mean_resp - np.min(mean_resp)
            
            OSI[filt] = np.abs(np.sum(mean_resp * np.exp(2j * np.arange(16)/8 * np.pi)) / 
                              np.abs(np.sum(mean_resp)))
            
            preferred[filt] = np.argmax(mean_resp)
            
            cosine = np.cos(np.arange(16)/8 * np.pi - np.pi/8 * preferred[filt])
            DSI[filt] = np.abs(np.sum(mean_resp * np.exp(2j * np.arange(16)/8 * np.pi) * cosine) / 
                              np.abs(np.sum(mean_resp)))
            
            response[filt,:] = mean_resp
            
            mean_resp_norm = mean_resp / np.sum(mean_resp)
            entropy[filt] = -np.sum(mean_resp_norm * np.log(mean_resp_norm + 1e-6))
        
        results[layer_name] = {
            'response': response,
            'OSI': OSI,
            'DSI': DSI,
            'preferred': preferred,
            'entropy': entropy
        }
    
    return results

def plot_layer_analysis(results, condition, save_dir):
    """Plot detailed layer analysis"""
    # Tuning curves
    ori = (np.arange(16) - 8) * 180/8
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, layer in enumerate(['V1', 'MT', 'MST']):
        if layer in results:
            response = results[layer]['response']
            preferred = results[layer]['preferred']
            
            mean_resp = np.zeros(16)
            for i in range(response.shape[0]):
                mean_resp += np.roll(response[i,:], 8-int(preferred[i]))
            mean_resp /= response.shape[0]
            mean_resp_norm = mean_resp / np.max(mean_resp)
            
            axes[idx].plot(ori, mean_resp_norm)
            axes[idx].set_title(f'{condition} {layer}')
            axes[idx].set_xlabel('Direction (deg)')
            axes[idx].set_ylabel('Normalized response')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tuning_curves.png'))
    plt.close()
    
    # Selectivity distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, layer in enumerate(['V1', 'MT', 'MST']):
        if layer in results:
            sns.histplot(data=results[layer]['OSI'], ax=axes[idx], label='OSI')
            sns.histplot(data=results[layer]['DSI'], ax=axes[idx], label='DSI')
            axes[idx].set_title(f'{layer} Selectivity')
            axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'selectivity_dist.png'))
    plt.close()

def plot_adaptive_metrics(metrics, save_dir):
    """Plot adaptive training metrics"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot accuracy and loss
    ax1.plot(metrics['train_acc'], label='Accuracy')
    ax1.plot(metrics['train_loss'], label='Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True)
    
    # Plot difficulty progression
    ax2.plot(metrics['difficulty'], label='Task Difficulty')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Difficulty')
    ax2.set_title('Task Difficulty Progression')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'adaptive_metrics.png'))
    plt.close()

def run_adaptive_experiment(experiment_dir='adaptive_experiment_results'):
    """Run complete experimental pipeline with adaptive difficulty"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(experiment_dir, f'experiment_{timestamp}')
    os.makedirs(base_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test stimuli
    print("\nGenerating test stimuli...")
    test_stimuli = torch.tensor(create_drifting_gratings(radius=100, lx=16, lt=25))
    test_stimuli = test_stimuli.to(device=device, dtype=torch.float)
    test_stimuli = test_stimuli + 0.1 * torch.randn_like(test_stimuli)
    
    # Define experimental conditions
    conditions = {
        'healthy': {'lesion_type': 'none'},
        'acute_lesion': {'lesion_type': 'complete'},
        'recovery': {'lesion_type': 'complete'}  # Will be trained post-lesion
    }
    
    results = {}
    condition_metrics = {}
    
    for condition, lesion_config in conditions.items():
        print(f"\n{'='*50}")
        print(f"Processing {condition} condition...")
        condition_dir = os.path.join(base_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)
        
        # Initialize model
        model = LesionExperimentModel().to(device)
        verify_v1_state(model, "Initial state")
        
        # For recovery, load the acute lesion model first
        if condition == 'recovery':
            acute_model_path = os.path.join(base_dir, 'acute_lesion', 'acute_lesion_model.pt')
            model.load_state_dict(torch.load(acute_model_path))
            print("\nLoaded acute lesion model for recovery")
            verify_v1_state(model, "After loading acute model")
        
        # Apply lesion if needed
        model.apply_lesion(**lesion_config)
        verify_v1_state(model, f"After applying {condition} lesion")
        
        # Train if needed
        if condition in ['healthy', 'recovery']:
            print(f"\nTraining {condition} model with adaptive difficulty...")
            train_adaptive_model_with_verification(
                model, 
                device,
                condition=condition,
                save_dir=os.path.join(condition_dir, 'training')
            )
            
            # Verify final state
            verify_v1_state(model, f"After {condition} training")
        
        # Analyze layers
        print(f"Analyzing {condition} model layers...")
        model.eval()
        with torch.no_grad():
            layer_results = analyze_model_layers(model, model.hooks, test_stimuli)
        results[condition] = layer_results
        
        # Save model
        torch.save(model.state_dict(), 
                  os.path.join(condition_dir, f'{condition}_model.pt'))
        
        # Plot layer-specific results
        plot_layer_analysis(layer_results, condition, condition_dir)
    
    # Plot condition comparisons
    print("\nGenerating comparison plots...")
    plot_condition_comparison(results, base_dir)
    
    # Plot training comparison for conditions that were trained
    if len(condition_metrics) > 0:
        plot_training_comparison(condition_metrics, base_dir)
    
    # Save complete results
    torch.save({
        'layer_results': results,
        'training_metrics': condition_metrics
    }, os.path.join(base_dir, 'complete_results.pt'))
    
    print(f"\nExperiment complete! Results saved to {base_dir}")
    return results, condition_metrics

def plot_condition_comparison(results, save_dir):
    """Plot comparisons across conditions"""
    conditions = list(results.keys())
    layers = ['V1', 'MT', 'MST']
    metrics = ['OSI', 'DSI', 'entropy']
    
    # Prepare data for plotting
    plot_data = []
    for condition in conditions:
        for layer in layers:
            if layer in results[condition]:
                for metric in metrics:
                    values = results[condition][layer][metric]
                    plot_data.extend([{
                        'Condition': condition,
                        'Layer': layer,
                        'Metric': metric,
                        'Value': val
                    } for val in values])
    
    df = pd.DataFrame(plot_data)
    
    # Plot comparisons for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        df_metric = df[df['Metric'] == metric]
        sns.boxplot(data=df_metric, x='Layer', y='Value', hue='Condition')
        plt.title(f'{metric} Comparison Across Conditions')
        plt.savefig(os.path.join(save_dir, f'{metric}_comparison.png'))
        plt.close()

def plot_training_comparison(condition_metrics, save_dir):
    """Plot training comparisons for different conditions"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot accuracy comparison
    for condition, metrics in condition_metrics.items():
        ax1.plot(metrics['train_acc'], label=f'{condition} Accuracy')
        ax1.plot(metrics['train_loss'], '--', label=f'{condition} Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Training Progress Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Plot difficulty progression comparison
    for condition, metrics in condition_metrics.items():
        ax2.plot(metrics['difficulty'], label=f'{condition} Difficulty')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Task Difficulty')
    ax2.set_title('Difficulty Progression Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_comparison.png'))
    plt.close()

if __name__ == "__main__":
    print("Starting adaptive V1 lesion experiment...")
    try:
        results, metrics = run_adaptive_experiment()
        print("Experiment completed successfully!")
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()

# def test_recovery_only(experiment_dir='recovery_test_results'):
#     """Test only the recovery phase"""
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     base_dir = os.path.join(experiment_dir, f'recovery_test_{timestamp}')
#     os.makedirs(base_dir, exist_ok=True)
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     # Initialize model and apply lesion
#     print("\nInitializing model and applying lesion...")
#     model = LesionExperimentModel().to(device)
#     model.apply_lesion(lesion_type='complete')
    
#     # Create test stimuli
#     print("\nGenerating test stimuli...")
#     test_stimuli = torch.tensor(create_drifting_gratings(radius=100, lx=16, lt=25))
#     test_stimuli = test_stimuli.to(device=device, dtype=torch.float)
#     test_stimuli = test_stimuli + 0.1 * torch.randn_like(test_stimuli)
    
#     # Train recovery
#     print("\nStarting recovery training...")
#     metrics = train_adaptive_model(
#         model, 
#         device,
#         n_epochs=5,  # Shorter for testing
#         save_dir=base_dir
#     )
    
#     # Analyze final state
#     print("\nAnalyzing final state...")
#     model.eval()
#     with torch.no_grad():
#         layer_results = analyze_model_layers(model, model.hooks, test_stimuli)
    
#     # Plot results
#     plot_layer_analysis(layer_results, 'recovery', base_dir)
#     plot_adaptive_metrics(metrics, base_dir)
    
#     # Save results
#     torch.save({
#         'layer_results': layer_results,
#         'training_metrics': metrics
#     }, os.path.join(base_dir, 'recovery_results.pt'))
    
#     print(f"\nRecovery test complete! Results saved to {base_dir}")
#     return metrics, layer_results

# if __name__ == "__main__":
#     print("Starting recovery test...")
#     try:
#         metrics, results = test_recovery_only()
#         print("Recovery test completed successfully!")
#     except Exception as e:
#         print(f"Error during recovery test: {e}")
#         import traceback
#         traceback.print_exc()