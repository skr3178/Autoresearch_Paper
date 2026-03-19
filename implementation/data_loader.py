import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class NuPlanDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.data_files[idx])
        data = np.load(file_path)
        bev = data['bev']
        ego_history = data['ego_history']
        gt_trajectory = data['gt_trajectory']
        scenario_id = data['scenario_id']

        if self.transform:
            bev = self.transform(bev)

        return {
            'bev': torch.tensor(bev, dtype=torch.float32),
            'ego_history': torch.tensor(ego_history, dtype=torch.float32),
            'gt_trajectory': torch.tensor(gt_trajectory, dtype=torch.float32),
            'scenario_id': scenario_id
        }

def collate_fn(batch):
    bev = torch.stack([item['bev'] for item in batch])
    ego_history = torch.stack([item['ego_history'] for item in batch])
    gt_trajectory = torch.stack([item['gt_trajectory'] for item in batch])
    scenario_ids = [item['scenario_id'] for item in batch]
    return {'bev': bev, 'ego_history': ego_history, 'gt_trajectory': gt_trajectory, 'scenario_id': scenario_ids}

def build_dataloader(data_dir, batch_size=4, shuffle=True, num_workers=0):
    dataset = NuPlanDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
