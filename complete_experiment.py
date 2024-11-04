# complete_experiment.py
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
    from discrimination_model import DiscriminationDorsalNet, create_discrimination_pairs
    from test_util import create_drifting_gratings
except ImportError as e:
    print(f"Error importing required modules: {e}")
    raise

class LesionExperimentModel(DiscriminationDorsalNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lesion_mask = None
        self.lesion_type = None
        self.hooks = {}
        self.register_hooks()
        
    def register_hooks(self):
        """Register hooks for layer analysis"""
        def get_activation(name):
            def hook(model, input, output):
                self.hooks[name] = output.detach()
            return hook
        
        self.s1.conv1.register_forward_hook(get_activation('layer01'))  # V1
        self.res1.register_forward_hook(get_activation('layer03'))      # MT
        self.res3.register_forward_hook(get_activation('layer04'))      # MST
        
    def apply_lesion(self, lesion_type='complete', lesion_size=0.5):
        """Apply V1 lesion"""
        self.lesion_type = lesion_type
        
        if lesion_type == 'none':
            self.lesion_mask = None
            return
            
        v1_shape = self.s1.conv1.weight.shape
        
        if lesion_type == 'complete':
            self.lesion_mask = torch.zeros(v1_shape)
        elif lesion_type == 'partial':
            self.lesion_mask = torch.ones(v1_shape)
            n_channels = int(v1_shape[0] * lesion_size)
            self.lesion_mask[:n_channels] = 0
            
    def forward(self, x1, x2=None):
        """Forward pass with lesion handling"""
        if self.lesion_mask is not None and self.lesion_type != 'none':
            original_weights = self.s1.conv1.weight.data.clone()
            self.s1.conv1.weight.data *= self.lesion_mask.to(original_weights.device)
            
        output = super().forward(x1, x2)
        
        if self.lesion_mask is not None and self.lesion_type != 'none':
            self.s1.conv1.weight.data = original_weights
            
        return output

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
            
            # Orientation selectivity
            OSI[filt] = np.abs(np.sum(mean_resp * np.exp(2j * np.arange(16)/8 * np.pi)) / 
                              np.abs(np.sum(mean_resp)))
            
            # Preferred direction
            preferred[filt] = np.argmax(mean_resp)
            
            # Direction selectivity
            cosine = np.cos(np.arange(16)/8 * np.pi - np.pi/8 * preferred[filt])
            DSI[filt] = np.abs(np.sum(mean_resp * np.exp(2j * np.arange(16)/8 * np.pi) * cosine) / 
                              np.abs(np.sum(mean_resp)))
            
            response[filt,:] = mean_resp
            
            # Response entropy
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

def train_model(model, device, n_epochs=20, batch_size=8, save_dir='training_results'):
    """Train model with detailed logging"""
    os.makedirs(save_dir, exist_ok=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler()
    
    metrics = {
        'train_acc': [],
        'train_loss': [],
        'epoch_times': []
    }
    
    for epoch in range(n_epochs):
        epoch_start = datetime.now()
        model.train()
        epoch_acc = []
        epoch_loss = []
        
        pbar = tqdm(range(100), desc=f'Epoch {epoch+1}/{n_epochs}')
        for batch in pbar:
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
            
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(stim1_batch, stim2_batch)
                loss = criterion(outputs, label_batch)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            with torch.no_grad():
                predictions = torch.sigmoid(outputs)
                acc = ((predictions > 0.5) == label_batch).float().mean()
                epoch_acc.append(acc.item())
                epoch_loss.append(loss.item())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{acc.item():.3f}'})
        
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        metrics['train_acc'].append(np.mean(epoch_acc))
        metrics['train_loss'].append(np.mean(epoch_loss))
        metrics['epoch_times'].append(epoch_time)
        
        print(f"Epoch {epoch}: Acc={metrics['train_acc'][-1]:.3f}, "
              f"Loss={metrics['train_loss'][-1]:.4f}, Time={epoch_time:.1f}s")
        
        # Save checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    return metrics

def run_complete_experiment(experiment_dir='experiment_results'):
    """Run complete experimental pipeline"""
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
    
    for condition, lesion_config in conditions.items():
        print(f"\n{'='*50}")
        print(f"Processing {condition} condition...")
        condition_dir = os.path.join(base_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)
        
        # Initialize model
        model = LesionExperimentModel().to(device)
        
        # Apply lesion if needed
        model.apply_lesion(**lesion_config)
        
        # Train if needed
        if condition in ['healthy', 'recovery']:
            print(f"Training {condition} model...")
            metrics = train_model(
                model, 
                device,
                save_dir=os.path.join(condition_dir, 'training')
            )
            
            # Plot training curves
            plt.figure(figsize=(10, 5))
            plt.plot(metrics['train_acc'], label='Accuracy')
            plt.plot(metrics['train_loss'], label='Loss')
            plt.title(f'{condition.capitalize()} Model Training')
            plt.xlabel('Epoch')
            plt.legend()
            plt.savefig(os.path.join(condition_dir, 'training_curves.png'))
            plt.close()
        
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
    
    # Save complete results
    torch.save(results, os.path.join(base_dir, 'complete_results.pt'))
    
    print(f"\nExperiment complete! Results saved to {base_dir}")
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

def plot_condition_comparison(results, save_dir):
    """Plot comparisons across conditions"""
    conditions = list(results.keys())
    layers = ['V1', 'MT', 'MST']
    metrics = ['OSI', 'DSI', 'entropy']
    
    # Prepare data for plotting
    plot_data = []
    for condition in conditions:
        for layer in layers:
            for metric in metrics:
                values = results[condition][layer][metric]
                plot_data.extend([{
                    'Condition': condition,
                    'Layer': layer,
                    'Metric': metric,
                    'Value': val
                } for val in values])
    
    df = pd.DataFrame(plot_data)
    
    # Plot comparisons
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[df['Metric'] == metric],
                   x='Layer', y='Value', hue='Condition')
        plt.title(f'{metric} Comparison Across Conditions')
        plt.savefig(os.path.join(save_dir, f'{metric}_comparison.png'))
        plt.close()

def test_recovery_only(experiment_dir='recovery_test_results'):
    """Run only the recovery phase to test functionality"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(experiment_dir, f'experiment_{timestamp}')
    os.makedirs(base_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nInitializing model for recovery test...")
    model = LesionExperimentModel().to(device)
    
    print("\nApplying lesion...")
    model.apply_lesion(lesion_type='complete')
    print(f"V1 weight mean after lesion: {model.s1.conv1.weight.data.abs().mean().item():.6f}")
    print(f"V1 requires_grad: {model.s1.conv1.weight.requires_grad}")
    
    print("\nStarting recovery training...")
    metrics = train_model(
        model,
        device,
        save_dir=os.path.join(base_dir, 'training'),
        n_epochs=5  # Shorter training for test
    )
    
    print("\nAnalyzing final state...")
    test_stimuli = torch.tensor(create_drifting_gratings(radius=100, lx=16, lt=25))
    test_stimuli = test_stimuli.to(device=device, dtype=torch.float)
    test_stimuli = test_stimuli + 0.1 * torch.randn_like(test_stimuli)
    
    model.eval()
    with torch.no_grad():
        layer_results = analyze_model_layers(model, model.hooks, test_stimuli)
        
    print("\nChecking final weights:")
    print(f"V1 final weight mean: {model.s1.conv1.weight.data.abs().mean().item():.6f}")
    
    return metrics, layer_results

if __name__ == "__main__":
    print("Starting recovery test...")
    try:
        metrics, results = test_recovery_only()
        print("Recovery test completed successfully!")
    except Exception as e:
        print(f"Error during recovery test: {e}")
        import traceback
        traceback.print_exc()

# if __name__ == "__main__":
#     print("Starting complete V1 lesion experiment...")
#     results = run_complete_experiment()