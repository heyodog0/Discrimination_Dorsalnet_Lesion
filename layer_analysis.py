# layer_analysis.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple

def analyze_model_layers(model, hooks, stimuli: torch.Tensor) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Analyze orientation and direction selectivity of different layers
    """
    layers = {'layer01': 'V1', 'layer03': 'MT', 'layer04': 'MST'}
    results = {}
    
    for layer_key, layer_name in layers.items():
        # Get layer responses
        output = model(stimuli)
        resp = hooks[layer_key]
        resp_shape = resp.shape
        num_filters = resp_shape[1]
        
        # Initialize metrics
        OSI = np.zeros(num_filters)
        DSI = np.zeros(num_filters)
        preferred = np.zeros(num_filters)
        response = np.zeros((num_filters, 16))
        entropy = np.zeros(num_filters)
        
        # Calculate metrics for each filter
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
            
            # Store full response
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

def plot_tuning_curves(results: Dict[str, Dict[str, np.ndarray]], condition: str):
    """Plot orientation/direction tuning curves for each layer"""
    ori = (np.arange(16) - 8) * 180/8
    layers = ['V1', 'MT', 'MST']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, layer in enumerate(layers):
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
    return fig

def plot_selectivity_comparison(results_dict: Dict[str, Dict[str, Dict[str, np.ndarray]]]):
    """Compare selectivity indices across conditions and layers"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    conditions = list(results_dict.keys())
    layers = ['V1', 'MT', 'MST']
    
    osi_data = []
    dsi_data = []
    
    for cond in conditions:
        for layer in layers:
            if layer in results_dict[cond]:
                osi = results_dict[cond][layer]['OSI']
                dsi = results_dict[cond][layer]['DSI']
                
                osi = osi[~np.isnan(osi)]
                dsi = dsi[~np.isnan(dsi)]
                
                osi_data.append({
                    'Condition': cond,
                    'Layer': layer,
                    'OSI': np.mean(osi)
                })
                dsi_data.append({
                    'Condition': cond,
                    'Layer': layer,
                    'DSI': np.mean(dsi)
                })
    
    if osi_data:
        sns.barplot(data=pd.DataFrame(osi_data), x='Layer', y='OSI', hue='Condition', ax=ax1)
        ax1.set_title('Orientation Selectivity Index')
    
    if dsi_data:
        sns.barplot(data=pd.DataFrame(dsi_data), x='Layer', y='DSI', hue='Condition', ax=ax2)
        ax2.set_title('Direction Selectivity Index')
    
    plt.tight_layout()
    return fig