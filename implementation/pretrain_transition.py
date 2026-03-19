#!/usr/bin/env python3
"""
Pre-training script for Transition Model
Implements Stage 1 of Algorithm 1 from paper.
"""

import sys
import os
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit')

from data_loader import NuPlanDataset, build_dataloader
from transition_model import TransitionModel


def pretrain_transition(
    data_root: str,
    map_root: str,
    save_dir: str,
    num_epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-4,
    num_workers: int = 0,
    device: str = 'cuda'
):
    """
    Pretrain the transition model.
    
    Args:
        data_root: Path to nuPlan mini split data
        map_root: Path to nuPlan maps
        save_dir: Directory to save models
        num_epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
    """
    print("=" * 80)
    print("Pretraining Transition Model")
    print("=" * 80)
    
    # Create dataloader
    dataloader = build_dataloader(
        data_root=data_root,
        map_root=map_root,
        batch_size=batch_size,
        shuffle=True,
        num_scenarios=100
    )
    
    # Create model
    model = TransitionModel(
        bev_channels=7,
        bev_height=224,
        bev_width=224,
        num_actions=2
    ).to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Loss function
    criterion = MSELoss()
    
    # Training loop
    model.train()
    
    num_batches = len(dataloader)
    print(f"Training on {num_batches} batches...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches_in_epoch = 0
        
        for batch_idx, tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            bev = batch['bev'].to(device)
            ego_history = batch['ego_history'].to(device)
            gt_trajectory = batch['gt_trajectory'].to(device)
            
            # Get ego actions (extract from GT trajectory)
            # For simplicity, we'll use the as a proxy for actions
            # In practice, you model predicts actions at time t
            # For now, use ground truth trajectory as action
            action = gt_trajectory[:, :2]  # Just take x, y, heading from GT
            
            # Forward pass
            next_bev, done, collision = model(bev, action)
            
            # Compute loss
            loss = criterion(next_bev, bev)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches_in_epoch += 1
        
        avg_loss = epoch_loss / num_batches_in_epoch
        print(f"Epoch {epoch+1} Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f"transition_model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    print("=" * 80)
    print("Pretraining Complete!")
    print("=" * 80)


if __name__ == "__main__":
    pretrain_transition(
        data_root='/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-extracted/data/cache/mini',
        map_root='/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/maps/nuplan-maps-v1.0',
        save_dir='./checkpoints',
        num_epochs=2,
        batch_size=8,
        lr=1e-4
    )
