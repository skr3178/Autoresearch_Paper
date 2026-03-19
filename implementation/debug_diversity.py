#!/usr/bin/env python3
"""Debug script to understand why diversity test fails."""

import sys
sys.path.insert(0, '/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit')
sys.path.insert(0, '/media/skr/storage/autoresearch/autoresearch-paper/implementation')

import torch
import torch.nn.functional as F
from transition_model import TransitionModel

# Create model
model = TransitionModel(
    bev_channels=7,
    bev_height=224,
    bev_width=224,
    action_dim=2,
    hidden_dim=64
)
model.eval()

# Same BEV, different actions
torch.manual_seed(456)
bev = torch.randn(1, 7, 224, 224)

# Two very different actions
action1 = torch.tensor([[1.0, 0.0]])  # Strong steer right, no acceleration
action2 = torch.tensor([[-1.0, 1.0]])  # Strong steer left, full acceleration

print("Actions:")
print(f"  action1: {action1}")
print(f"  action2: {action2}")
print(f"  difference: {(action1 - action2).abs().sum().item()}")
print()

# Trace through the network
with torch.no_grad():
    # Encode BEV
    bev_features = model.encoder(bev)
    print(f"BEV features shape: {bev_features.shape}")
    print(f"BEV features mean: {bev_features.mean().item():.6f}")
    print()
    
    # Apply FiLM conditioning
    gamma1 = model.film.gamma_net(action1)
    beta1 = model.film.beta_net(action1)
    gamma2 = model.film.gamma_net(action2)
    beta2 = model.film.beta_net(action2)
    
    print(f"FiLM gamma1: mean={gamma1.mean().item():.6f}, std={gamma1.std().item():.6f}")
    print(f"FiLM gamma2: mean={gamma2.mean().item():.6f}, std={gamma2.std().item():.6f}")
    print(f"FiLM gamma difference: {(gamma1 - gamma2).abs().mean().item():.6f}")
    print()
    print(f"FiLM beta1: mean={beta1.mean().item():.6f}, std={beta1.std().item():.6f}")
    print(f"FiLM beta2: mean={beta2.mean().item():.6f}, std={beta2.std().item():.6f}")
    print(f"FiLM beta difference: {(beta1 - beta2).abs().mean().item():.6f}")
    print()
    
    # Conditioned features
    cond1 = model.film(bev_features, action1)
    cond2 = model.film(bev_features, action2)
    
    print(f"Conditioned features1 mean: {cond1.mean().item():.6f}")
    print(f"Conditioned features2 mean: {cond2.mean().item():.6f}")
    print(f"Conditioned features difference: {(cond1 - cond2).abs().mean().item():.6f}")
    print()
    
    # Full forward pass
    next_bev1, done1 = model(bev, action1)
    next_bev2, done2 = model(bev, action2)
    
    print(f"next_bev1: shape={next_bev1.shape}, mean={next_bev1.mean().item():.6f}, std={next_bev1.std().item():.6f}")
    print(f"next_bev2: shape={next_bev2.shape}, mean={next_bev2.mean().item():.6f}, std={next_bev2.std().item():.6f}")
    print()
    
    # Check if they're actually identical
    diff = (next_bev1 - next_bev2).abs()
    print(f"Absolute difference:")
    print(f"  mean: {diff.mean().item():.8f}")
    print(f"  max: {diff.max().item():.8f}")
    print(f"  sum: {diff.sum().item():.8f}")
    print()
    
    # Check first few values
    print(f"First 10 values of next_bev1[0, 0, 0, :]: {next_bev1[0, 0, 0, :10]}")
    print(f"First 10 values of next_bev2[0, 0, 0, :]: {next_bev2[0, 0, 0, :10]}")
    print()
    
    # Cosine similarity
    flat1 = next_bev1.flatten()
    flat2 = next_bev2.flatten()
    cosine_sim = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0)).item()
    print(f"Cosine similarity: {cosine_sim:.6f}")
