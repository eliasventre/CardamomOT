"""
PyTorch version of network inference using Adam optimizer
Simplified version focusing on core inference without refinement
"""

from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
from joblib import Parallel, delayed
from typing import Any
from numpy import ndarray, floating

# Hyperparameters
eps_CE = 1e-6
sc = 1e-3
r_elasticnet = 0.5


def base_kon_torch(theta_basal, theta_inter, y_prot) -> torch.Tensor:
    """
    PyTorch version of base_kon with numerical stability
    
    Args:
        theta_basal: (n_networks,) basal parameters
        theta_inter: (G, n_networks) interaction parameters
        y_prot: (n_cells, G) protein expression values
    
    Returns:
        (n_cells, n_networks+1) probabilities for each network state
    """
    
    # Compute logits: Z[i,k] = theta_basal[k] + sum_g(y_prot[i,g] * theta_inter[g,k])
    Z = theta_basal.unsqueeze(0) + torch.matmul(y_prot, theta_inter)  # (n_cells, n_networks)
    
    # Stable softmax: include class 0 (no network)
    Z_max: torch.Tensor = torch.max(Z, dim=1, keepdim=True)[0]
    Z_max: torch.Tensor = torch.clamp(Z_max, max=50.0)  # Prevent overflow
    
    # exp(Z - Z_max) for networks 1...n_networks
    exp_Z: torch.Tensor = torch.exp(Z - Z_max)  # (n_cells, n_networks)
    
    # exp(-Z_max) for network 0
    exp_0: torch.Tensor = torch.exp(-Z_max)  # (n_cells, 1)
    
    # Concatenate [class_0, class_1, ..., class_n]
    exp_all: torch.Tensor = torch.cat([exp_0, exp_Z], dim=1)  # (n_cells, n_networks+1)
    
    # Normalize
    denom: torch.Tensor = exp_all.sum(dim=1, keepdim=True)
    sigma: torch.Tensor = exp_all / denom
    
    return sigma


def smoothed_l1_loss(tensor, l1_weight) -> torch.Tensor:
    """Smoothed L1 penalization"""
    abs_tensor: torch.Tensor = torch.abs(tensor)
    smoothed: torch.Tensor = torch.where(
        abs_tensor < sc,
        l1_weight * 0.5 * (tensor ** 2) / sc,
        l1_weight * (abs_tensor - 0.5 * sc)
    )
    return smoothed.sum()


def l2_loss(tensor, l2_weight):
    """L2 penalization"""
    return l2_weight * (tensor ** 2).sum()


def elasticnet_loss(tensor, weight):
    """Elastic net: combination of L1 and L2"""
    return r_elasticnet * l2_loss(tensor, weight) + (1 - r_elasticnet) * smoothed_l1_loss(tensor, weight)


def compute_main_loss(y_pred, y_true, loss_type='CE'):
    """
    Compute main loss (L1, L2, or Cross-Entropy)
    
    Args:
        y_pred: predicted values
        y_true: true values
        loss_type: 'l1', 'l2', or 'CE'
    """
    if loss_type == 'l1':
        diff = y_pred - y_true
        abs_diff: torch.Tensor = torch.abs(diff)
        smoothed: torch.Tensor = torch.where(
            abs_diff < sc,
            0.5 * (diff ** 2) / sc,
            abs_diff - 0.5 * sc
        )
        return smoothed.sum()
    
    elif loss_type == 'l2':
        return ((y_pred - y_true) ** 2).sum()
    
    else:  # Cross-Entropy
        y_pred_clipped: torch.Tensor = torch.clamp(y_pred, eps_CE, 1 - eps_CE)
        ce_loss = -(y_true * torch.log(y_pred_clipped) + 
                    (1 - y_true) * torch.log(1 - y_pred_clipped))
        return ce_loss.sum()


class GRNInference(nn.Module):
    """
    PyTorch module for Gene Regulatory Network inference
    """
    def __init__(self, G, n_networks, inter_init, basal_init, ref_network) -> None:
        """
        Args:
            G: number of genes
            n_networks: number of networks to infer
            inter_init: (G, n_networks) initial interaction parameters
            basal_init: (n_networks,) initial basal parameters
            ref_network: (G, n_networks) reference network mask
        """
        super(GRNInference, self).__init__()
        
        self.G: Any = G
        self.n_networks: Any = n_networks
        
        # Learnable parameters
        self.theta_inter = nn.Parameter(torch.FloatTensor(inter_init))
        self.theta_basal = nn.Parameter(torch.FloatTensor(basal_init))
        
        # Register reference network and reference parameters as buffers (not learned)
        self.register_buffer('ref_network', torch.FloatTensor(ref_network))
        self.register_buffer('theta_inter_ref', torch.FloatTensor(inter_init))
        self.register_buffer('theta_basal_ref', torch.FloatTensor(basal_init))
        
    def forward(self, y_prot) -> torch.Tensor:
        """
        Forward pass: compute network probabilities
        
        Args:
            y_prot: (n_cells, G) protein expression
            
        Returns:
            sigma: (n_cells, n_networks+1) network probabilities
        """
        # Apply reference network mask
        theta_inter_masked = self.theta_inter * self.ref_network
        
        # Compute probabilities
        sigma: torch.Tensor = base_kon_torch(self.theta_basal, theta_inter_masked, y_prot)
        
        return sigma
    
    def compute_loss(self, y_samples, y_proba, y_prot, y_prot_prev, y_kon, ks, 
                     proba, l_pen, weight_prev=0.5, loss_type='CE'):
        """
        Compute total loss including data fitting and regularization
        
        Args:
            y_samples: (n_cells,) sample/batch indices
            y_proba: (n_cells, n_networks+1) target probabilities (if proba=True)
            y_prot: (n_cells, G) protein expression current timepoint
            y_prot_prev: (n_cells, G) protein expression previous timepoint
            y_kon: (n_cells,) target kon values (if proba=False)
            ks: (n_networks+1,) network activation rates
            proba: bool, whether to predict probabilities or kon
            l_pen: penalization weight
            weight_prev: weight for previous timepoint loss
            loss_type: 'l1', 'l2', or 'CE'
            final: bool, use elastic net if True
        """
        Q = 0.0
        
        # Get unique samples and their weights
        unique_samples = torch.unique(y_samples)
        ns: int = len(unique_samples)
        
        # Current timepoint loss
        sigma: torch.Tensor = self.forward(y_prot)
        
        for s in unique_samples:
            mask = (y_samples == s)
            n_samples_s = mask.sum().item()
            
            # Balance weights across samples
            weight_s = len(y_samples) / n_samples_s
            
            if proba:
                # Predict network probabilities
                loss_s = compute_main_loss(sigma[mask], y_proba[mask], loss_type)
            else:
                # Predict kon = sum(ks * sigma)
                y_pred = (ks.unsqueeze(0) * sigma[mask]).sum(dim=1)
                loss_s = compute_main_loss(y_pred, y_kon[mask], loss_type)
            
            Q += (1 - weight_prev) * loss_s * weight_s / ns
        
        # Previous timepoint loss (if weight_prev > 0)
        if weight_prev > 0:
            sigma_prev: torch.Tensor = self.forward(y_prot_prev)
            
            for s in unique_samples:
                mask = (y_samples == s)
                n_samples_s = mask.sum().item()
                weight_s = len(y_samples) / n_samples_s
                
                if proba:
                    loss_s = compute_main_loss(sigma_prev[mask], y_proba[mask], loss_type)
                else:
                    y_pred = (ks.unsqueeze(0) * sigma_prev[mask]).sum(dim=1)
                    loss_s = compute_main_loss(y_pred, y_kon[mask], loss_type)
                
                Q += weight_prev * loss_s * weight_s / ns
        
        # Regularization: penalize deviation from reference
        theta_diff_inter = (self.theta_inter - self.theta_inter_ref) * (self.ref_network > 0)
        theta_diff_basal = self.theta_basal - self.theta_basal_ref
        
        Q += elasticnet_loss(theta_diff_inter, l_pen)
        Q += elasticnet_loss(theta_diff_basal, l_pen)
        
        return Q


def inference_pytorch(y_samples, y_kon, y_proba, y_prot, y_prot_prev, ks, 
                      inter_init, basal_init, ref_network,
                      inter_ref=None, basal_ref=None,
                      proba=True, scale=100, 
                      weight_prev=0.5, loss_type='CE', 
                      lr=1e-3, n_epochs=1000, 
                      verbose=True,
                      device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    PyTorch-based network inference using Adam optimizer
    
    Args:
        y_samples: (n_cells,) sample/batch indices
        y_kon: (n_cells,) target kon values
        y_proba: (n_cells, n_networks+1) target network probabilities
        y_prot: (n_cells, G) protein expression (current)
        y_prot_prev: (n_cells, G) protein expression (previous)
        ks: (n_networks+1,) network activation rates
        inter_init: (G, n_networks) initial interaction parameters
        basal_init: (n_networks,) initial basal parameters
        ref_network: (G, n_networks) reference network structure
        inter_ref: (G, n_networks) reference interaction parameters (defaults to inter_init)
        basal_ref: (n_networks,) reference basal parameters (defaults to basal_init)
        proba: bool, predict probabilities (True) or kon values (False)
        scale: scaling factor for penalization
        weight_prev: weight for previous timepoint in loss
        loss_type: 'l1', 'l2', or 'CE'
        lr: learning rate for Adam
        n_epochs: number of training epochs
        final: use elastic net regularization
        verbose: print progress
        device: 'cuda' or 'cpu'
    
    Returns:
        inter: (G, n_networks) inferred interaction parameters
        basal: (n_networks,) inferred basal parameters
        losses: list of loss values during training
    """
    
    # Convert to torch tensors
    y_samples: torch.Tensor = torch.LongTensor(y_samples).to(device)
    y_kon: torch.Tensor = torch.FloatTensor(y_kon).to(device)
    y_proba: torch.Tensor = torch.FloatTensor(y_proba).to(device)
    y_prot: torch.Tensor = torch.FloatTensor(y_prot).to(device)
    y_prot_prev: torch.Tensor = torch.FloatTensor(y_prot_prev).to(device)
    ks: torch.Tensor = torch.FloatTensor(ks).to(device)
    
    # Get dimensions
    n_cells, G = y_prot.shape
    n_networks = inter_init.shape[1]
    
    # Use init as ref if not provided
    if inter_ref is None:
        inter_ref = inter_init.copy()
    if basal_ref is None:
        basal_ref = basal_init.copy()
    
    # Initialize model
    model: GRNInference = GRNInference(G, n_networks, inter_init, basal_init, ref_network).to(device)
    
    # Update reference parameters
    model.theta_inter_ref = torch.FloatTensor(inter_ref).to(device)
    model.theta_basal_ref = torch.FloatTensor(basal_ref).to(device)

    # Compute penalization coefficient
    l_pen= np.size(y_prot, 0) / (scale*(1+np.sqrt(G)))
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    losses = []
    
    iterator: tqdm[int] | range = tqdm(range(n_epochs), desc="Training") if verbose else range(n_epochs)
    
    for epoch in iterator:
        optimizer.zero_grad()
        
        # Compute loss
        loss = model.compute_loss(
            y_samples, y_proba, y_prot, y_prot_prev, y_kon, ks,
            proba, l_pen, weight_prev, loss_type
        )
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Record loss
        loss_val = loss.item()
        losses.append(loss_val)
        
        # Update progress bar
        if verbose and epoch % 10 == 0:
            iterator.set_postfix({'loss': f'{loss_val:.6f}'})
    
    # Extract final parameters
    with torch.no_grad():
        inter = (model.theta_inter * model.ref_network).cpu().numpy()
        basal = model.theta_basal.cpu().numpy()
    
    return inter, basal, losses


def inference_network_pytorch(vect_t, times, y_samples, y_kon, y_proba, y_prot, y_prot_prev, ks,
                               inter_init=np.zeros(2), basal_init=np.zeros(2), 
                               ref_network=np.zeros(2),
                               inter_ref=None, basal_ref=None,
                               proba=True, scale=100,
                               weight_prev=0.5, loss='CE',
                               lr=1e-2, n_epochs=1000,
                               verbose=True, final=1,
                               device='cuda' if torch.cuda.is_available() else 'cpu') -> Any:
    """
    Main entry point for PyTorch network inference
    Simplified version without temporal dynamics and refinement
    
    Args:
        y_samples: (n_cells,) sample indices
        y_kon: (n_cells, G) target kon values for all genes
        y_proba: (n_cells, G, n_networks+1) target probabilities for all genes
        y_prot: (n_cells, G) protein expression (current)
        y_prot_prev: (n_cells, G) protein expression (previous) 
        ks: (G, n_networks+1) activation rates per gene
        inter_init: (G, G, n_networks) initial interaction parameters
        basal_init: (G, n_networks) initial basal parameters
        ref_network: (G, G, n_networks) reference network structure
        inter_ref: (G, G, n_networks) reference interaction parameters
        basal_ref: (G, n_networks) reference basal parameters
        proba: predict probabilities or kon
        scale: penalization scaling
        weight_prev: weight for previous timepoint
        loss_type: 'l1', 'l2', or 'CE'
        lr: learning rate
        n_epochs: training epochs
        final: use elastic net
        verbose: print progress
        device: computation device
    
    Returns:
        inter: (G, G, n_networks) inferred interaction matrix
        basal: (G, n_networks) inferred basal parameters
        all_losses: dict with losses per gene
    """
    
    G = y_prot.shape[1]
    n_networks: int = inter_init.shape[2]
    
    # Initialize outputs
    inter_final = np.zeros((G, G, n_networks))
    basal_final = np.zeros((G, n_networks))
    
    # Set default ref_network if not provided
    if ref_network is None:
        ref_network = np.ones_like(inter_init)
    
    if inter_ref is None:
        inter_ref = inter_init.copy()
    
    if basal_ref is None:
        basal_ref = basal_init.copy()
    
    # Infer network for each gene

    def run_main_loop_for_gene(g):
        # Get active networks for this gene
        n_networks_g = int(1 + np.argmax(ks[g, 1:]))
        # Run inference for this gene
        inter_g, basal_g, losses_g = inference_pytorch(
        y_samples=y_samples,
        y_kon=y_kon[:, g].copy(),
        y_proba=y_proba[:, g, :n_networks_g+1].copy(),
        y_prot=y_prot.copy(),
        y_prot_prev=y_prot_prev[g, :, :].copy(),
        ks=ks[g, :n_networks_g+1],
        inter_init=inter_init[:, g, :n_networks_g].copy(),
        basal_init=basal_init[g, :n_networks_g].copy(),
        ref_network=ref_network[:, g, :n_networks_g].copy(),
        inter_ref=inter_ref[:, g, :n_networks_g].copy(),
        basal_ref=basal_ref[g, :n_networks_g].copy(),
        proba=proba,
        scale=scale,
        weight_prev=weight_prev,
        loss_type=loss,
        lr=lr,
        n_epochs=n_epochs,
        verbose=verbose,
        device=device)
        return inter_g, basal_g, n_networks_g

    results = Parallel(n_jobs=-1)(
    delayed(run_main_loop_for_gene)(g) for g in range(1, G)
    )

    for idx, g in enumerate(range(1, G)):
        inter_g, basal_g, n_networks_g = results[idx]
    
        # Store results
        inter_final[:, g, :n_networks_g] = inter_g
        basal_final[g, :n_networks_g] = basal_g
    
        # Set inactive networks to very low values
        if n_networks_g < n_networks:
            basal_final[g, n_networks_g:] = -100

    return basal_final, inter_final, basal_final, inter_final