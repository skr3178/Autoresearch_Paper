#!/usr/bin/env python3
"""
NuPlan Dataset Loader for CarPlanner
Implements submodules.md specification for data_loader submodule.

Input: nuPlan SQLite database files
Output: batched dict with keys:
  - bev: (B, C, H, W) float32 BEV raster (agents, map layers, ego history)
  - ego_history: (B, T_hist, 3) float32
  - gt_trajectory: (B, T_future, 3) float32 ground-truth future trajectory
  - scenario_id: str
"""

import sys
import os
import time
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Add nuplan-devkit to path
sys.path.insert(0, '/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit')

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class NuPlanDataset(Dataset):
    """
    nuPlan Dataset for CarPlanner
    
    Loads scenarios from nuPlan database and constructs BEV rasters, agent boxes, and ground-truth trajectories.
    """
    
    def __init__(
        self,
        data_root: str,
        map_root: str,
        map_version: str = '1.0',
        include_cameras: bool = False,
        max_workers: int = 1,
        verbose: bool = False,
        num_scenarios: Optional[int] = None,
        vehicle_parameters=None
    ):
        """
        Initialize dataset.
        
        Args:
            data_root: Path to nuPlan mini split data
            map_root: Path to nuPlan maps
            map_version: Map version to use
            include_cameras: Whether to include camera data
            max_workers: Number of workers for loading
            verbose: Whether to print debug info
            num_scenarios: Limit number of scenarios (None = all)
            vehicle_parameters: Vehicle parameters
        """
        super().__init__()
        self.data_root = data_root
        self.map_root = map_root
        self.map_version = map_version
        self.include_cameras = include_cameras
        self.max_workers = max_workers
        self.verbose = verbose
        self.vehicle_parameters = vehicle_parameters or get_pacifica_parameters()
        
        # BEV parameters
        self.bev_h = 224  # BEV height (pixels)
        self.bev_w = 224  # BEV width (pixels)
        self.bev_c = 7    # BEV channels (agents, map, ego history)
        self.resolution = 0.5  # meters per pixel
        self.bev_radius = 56.0  # meters (224 * 0.5 / 2)
        
        # Trajectory parameters
        self.t_hist = 5    # History steps
        self.t_future = 8  # Future steps
        
        # Create scenario builder
        if self.verbose:
            print("Loading scenarios...")
        
        self.scenario_builder = NuPlanScenarioBuilder(
            data_root=data_root,
            map_root=map_root,
            sensor_root='',
            db_files=None,
            map_version=map_version,
            include_cameras=include_cameras,
            max_workers=max_workers,
            verbose=verbose,
            vehicle_parameters=self.vehicle_parameters
        )
        
        # Load scenarios
        self.scenarios = []
        self._load_scenarios(num_scenarios)
        
        if self.verbose:
            print(f"Dataset initialized with {len(self.scenarios)} scenarios")
    
    def _load_scenarios(self, num_scenarios: Optional[int] = None):
        """Load scenarios from nuPlan dataset."""
        scenario_filter = ScenarioFilter(
            scenario_tokens=None,
            log_names=None,
            scenario_types=None,
            num_scenarios_per_type=None,
            limit_total_scenarios=num_scenarios,
            expand_scenarios=False,
            remove_invalid_goals=True,
            shuffle=False,
            ego_displacement_minimum_m=None,
            ego_start_speed_threshold=None,
            ego_stop_speed_threshold=None,
            speed_noise_tolerance=None,
            ego_route_radius=None,
            map_names=None,
            timestamp_threshold_s=None,
            token_set_path=None,
            fraction_in_token_set_threshold=None
        )
        
        worker = Sequential()
        self.scenarios = self.scenario_builder.get_scenarios(scenario_filter, worker)
    
    def __len__(self) -> int:
        return len(self.scenarios)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict containing all sample data
        """
        if idx >= len(self.scenarios):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.scenarios)} samples")
        
        scenario = self.scenarios[idx]
        iteration = 0  # Use first iteration for each scenario
        
        # Get ego state
        ego_state = scenario.get_ego_state_at_iteration(iteration)
        ego_pos = ego_state.rear_axle
        
        # Get ego future trajectory (ground truth)
        ego_future_gen = scenario.get_ego_future_trajectory(
            iteration=iteration, 
            time_horizon=8.0, 
            num_samples=8
        )
        ego_future = list(ego_future_gen)
        
        # Convert to numpy array (global coordinates)
        gt_trajectory = np.array([
            [wp.rear_axle.x, wp.rear_axle.y, wp.rear_axle.heading] 
            for wp in ego_future
        ], dtype=np.float32)
        
        # Get ego history (last 5 timesteps including current)
        # For now, just repeat current position (TODO: get actual history)
        ego_history = np.zeros((self.t_hist, 3), dtype=np.float32)
        for i in range(self.t_hist):
            # Repeat current state for history (placeholder)
            ego_history[i] = [ego_pos.x, ego_pos.y, ego_pos.heading]
        
        # Get tracked objects
        gt_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=None)
        tracked_objects = scenario.get_tracked_objects_at_iteration(iteration, gt_sampling)
        objects = tracked_objects.tracked_objects.tracked_objects
        
        # Build BEV raster (placeholder - just zeros for now)
        bev = torch.zeros((self.bev_c, self.bev_h, self.bev_w), dtype=torch.float32)
        
        # TODO: Implement proper BEV rasterization with:
        # - Agent boxes
        # - Map features (lanes, drivable area, etc.)
        # - Ego history trajectory
        
        # Convert to tensors
        ego_history_tensor = torch.from_numpy(ego_history)
        gt_trajectory_tensor = torch.from_numpy(gt_trajectory)
        
        return {
            'bev': bev,
            'ego_history': ego_history_tensor,
            'gt_trajectory': gt_trajectory_tensor,
            'agent_boxes': [],  # Placeholder
            'scenario_id': scenario.scenario_name
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of sample dicts
        
    Returns:
        Batched dict
    """
    bev = torch.stack([item['bev'] for item in batch])
    ego_history = torch.stack([item['ego_history'] for item in batch])
    gt_trajectory = torch.stack([item['gt_trajectory'] for item in batch])
    agent_boxes = [item['agent_boxes'] for item in batch]
    scenario_ids = [item['scenario_id'] for item in batch]
    
    return {
        'bev': bev,
        'ego_history': ego_history,
        'gt_trajectory': gt_trajectory,
        'agent_boxes': agent_boxes,
        'scenario_id': scenario_ids
    }


def build_dataloader(
    data_root: str,
    map_root: str,
    map_version: str = '1.0',
    batch_size: int = 2,
    shuffle: bool = True,
    num_scenarios: Optional[int] = None,
    num_workers: int = 0
) -> DataLoader:
    """
    Build DataLoader for nuPlan dataset.
    
    Args:
        data_root: Path to nuPlan mini split data
        map_root: Path to nuPlan maps
        map_version: Map version to use
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_scenarios: Limit number of scenarios
        num_workers: Number of workers
        
    Returns:
        DataLoader instance
    """
    dataset = NuPlanDataset(
        data_root=data_root,
        map_root=map_root,
        map_version=map_version,
        num_scenarios=num_scenarios,
        verbose=True
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Testing NuPlanDataset")
    print("=" * 80)
    
    # Dataset paths
    data_root = '/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-extracted/data/cache/mini'
    map_root = '/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/maps/nuplan-maps-v1.0'
    map_version = '1.0'
    
    # Test single sample loading
    print("\n1. Creating dataset...")
    dataset = NuPlanDataset(
        data_root=data_root,
        map_root=map_root,
        map_version=map_version,
        num_scenarios=5,
        verbose=True
    )
    
    print(f"\n2. Dataset length: {len(dataset)}")
    
    print(f"\n3. Loading single sample...")
    sample = dataset[0]
    
    print(f"   Sample keys: {sample.keys()}")
    print(f"   BEV shape: {sample['bev'].shape}")
    print(f"   BEV dtype: {sample['bev'].dtype}")
    print(f"   Ego history shape: {sample['ego_history'].shape}")
    print(f"   Ego history dtype: {sample['ego_history'].dtype}")
    print(f"   GT trajectory shape: {sample['gt_trajectory'].shape}")
    print(f"   GT trajectory dtype: {sample['gt_trajectory'].dtype}")
    print(f"   Scenario ID: {sample['scenario_id']}")
    
    # Test batch loading
    print(f"\n4. Testing batch loading...")
    dataloader = build_dataloader(
        data_root=data_root,
        map_root=map_root,
        map_version=map_version,
        batch_size=2,
        shuffle=False,
        num_scenarios=5
    )
    
    print(f"\n5. Loading 3 batches...")
    start_time = time.time()
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        print(f"   Batch {batch_count}:")
        print(f"     BEV shape: {batch['bev'].shape}")
        print(f"     Ego history shape: {batch['ego_history'].shape}")
        print(f"     GT trajectory shape: {batch['gt_trajectory'].shape}")
        
        if batch_count >= 3:
            break
    
    elapsed = time.time() - start_time
    print(f"\n6. Throughput test:")
    print(f"   Loaded {batch_count} batches in {elapsed:.2f}s")
    print(f"   Throughput: {batch_count/elapsed:.2f} batches/s")
    
    print("\n" + "=" * 80)
    print("Data Loader Test Complete")
    print("=" * 80)
