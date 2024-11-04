# run_experiment.py
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path if needed
sys.path.append('../')  # Adjust this path as needed

try:
    print("Importing required modules...")
    from train_discrimination import train_discrimination_model, plot_results
    from layer_analysis import analyze_model_layers, plot_tuning_curves, plot_selectivity_comparison
    from discrimination_model import DiscriminationDorsalNet
    from test_util import create_drifting_gratings
    print("All modules imported successfully")
except Exception as e:
    print(f"Error importing modules: {e}")
    raise

def main():
    try:
        print("\n1. Setting up directories and device...")
        save_path = 'discrimination_results'
        os.makedirs(save_path, exist_ok=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        print("\n2. Initializing model...")
        model = DiscriminationDorsalNet()
        model = model.to(device)
        print("Model initialized successfully")
        
        print("\n3. Creating test stimuli...")
        stimuli = torch.tensor(create_drifting_gratings(radius=100, lx=16, lt=25))
        stimuli = stimuli.to(device=device, dtype=torch.float)
        noise = 0.1
        stimuli = stimuli + noise * torch.randn_like(stimuli)
        print(f"Stimuli created with shape: {stimuli.shape}")
        
        print("\n4. Setting up pre-training analysis...")
        hooks_pre = {}
        def get_activation(name):
            def hook(model, input, output):
                hooks_pre[name] = output.detach()
            return hook
        
        print("Registering hooks...")
        model.s1.conv1.register_forward_hook(get_activation('layer01'))
        model.res1.register_forward_hook(get_activation('layer03'))
        model.res3.register_forward_hook(get_activation('layer04'))
        
        print("\n5. Running pre-training analysis...")
        # Switch to analysis mode before running analysis
        model.set_analysis_mode(True)
        with torch.no_grad():
            results_pre = analyze_model_layers(model, hooks_pre, stimuli)
        print("Pre-training analysis complete")
        
        print("\n6. Starting model training...")
        # Switch back to discrimination mode for training
        model.set_analysis_mode(False)
        model, pre_acc, post_acc = train_discrimination_model(device, save_path)
        print("Training complete")
        
        print("\n7. Plotting training results...")
        # Plot training results
        plot_results(pre_acc, post_acc, save_path)
        
        print("\n8. Running post-training analysis...")
        # Analyze post-training state
        hooks_post = {}
        def get_activation(name):
            def hook(model, input, output):
                hooks_post[name] = output.detach()
            return hook
        
        # Re-register hooks for post-training
        model.s1.conv1.register_forward_hook(get_activation('layer01'))
        model.res1.register_forward_hook(get_activation('layer03'))
        model.res3.register_forward_hook(get_activation('layer04'))
        
        # Get post-training analysis
        with torch.no_grad():
            results_post = analyze_model_layers(model, hooks_post, stimuli)
        
        print("\n9. Generating comparison plots...")
        # Compare pre and post training
        results_dict = {
            'Pre-training': results_pre,
            'Post-training': results_post
        }
        
        # Save all plots
        print("Saving plots...")
        figs = {
            'pre_tuning': plot_tuning_curves(results_pre, 'Pre-training'),
            'post_tuning': plot_tuning_curves(results_post, 'Post-training'),
            'selectivity': plot_selectivity_comparison(results_dict)
        }
        
        for name, fig in figs.items():
            fig.savefig(os.path.join(save_path, f'{name}.png'))
            plt.close(fig)
        
        # Save numerical results
        print("Saving numerical results...")
        np.save(os.path.join(save_path, 'pre_training_results.npy'), results_pre)
        np.save(os.path.join(save_path, 'post_training_results.npy'), results_post)
        
        print("\n10. Generating summary...")
        # Print summary
        print("\nSummary of changes:")
        layers = ['V1', 'MT', 'MST']
        for layer in layers:
            if layer in results_pre and layer in results_post:
                print(f"\n{layer} Layer:")
                osi_pre = np.mean(results_pre[layer]['OSI'][~np.isnan(results_pre[layer]['OSI'])])
                osi_post = np.mean(results_post[layer]['OSI'][~np.isnan(results_post[layer]['OSI'])])
                dsi_pre = np.mean(results_pre[layer]['DSI'][~np.isnan(results_pre[layer]['DSI'])])
                dsi_post = np.mean(results_post[layer]['DSI'][~np.isnan(results_post[layer]['DSI'])])
                
                print(f"OSI: {osi_pre:.3f} -> {osi_post:.3f} ({(osi_post-osi_pre)/osi_pre*100:+.1f}%)")
                print(f"DSI: {dsi_pre:.3f} -> {dsi_post:.3f} ({(dsi_post-dsi_pre)/dsi_pre*100:+.1f}%)")
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    print("Starting analysis pipeline...")
    main()