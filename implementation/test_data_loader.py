import unittest
import torch
from data_loader import build_dataloader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Assuming the data is extracted to 'implementation/cache/'
        self.data_dir = 'implementation/cache/'
        self.batch_size = 2
        self.dataloader = build_dataloader(self.data_dir, batch_size=self.batch_size)

    def test_data_loader_output(self):
        for batch in self.dataloader:
            bev = batch['bev']
            ego_history = batch['ego_history']
            gt_trajectory = batch['gt_trajectory']
            scenario_id = batch['scenario_id']

            # Check types
            self.assertIsInstance(bev, torch.Tensor)
            self.assertIsInstance(ego_history, torch.Tensor)
            self.assertIsInstance(gt_trajectory, torch.Tensor)
            self.assertIsInstance(scenario_id, list)

            # Check shapes
            self.assertEqual(bev.shape[0], self.batch_size)
            self.assertEqual(ego_history.shape[0], self.batch_size)
            self.assertEqual(gt_trajectory.shape[0], self.batch_size)

            # Check dtypes
            self.assertEqual(bev.dtype, torch.float32)
            self.assertEqual(ego_history.dtype, torch.float32)
            self.assertEqual(gt_trajectory.dtype, torch.float32)

            # Check value ranges
            self.assertTrue(torch.all(bev >= 0) and torch.all(bev <= 1))

            break  # Only check the first batch for this test

if __name__ == '__main__':
    unittest.main()
