import numpy as np
import torch
from torch.utils.data import Dataset



class MarkerDataset(Dataset):
    '''
    Dataset class for joint transformation Y, and local reference frame F
    '''

    def __init__(self, joint_transform, local_reference):
        '''
        Parameters:
            joint_transform  -- joint's global homogeneous transformation matrices Y     (nof, noj, 4, 4)
            local_reference  -- homogeneous local reference frame F                      (nof, 4, 4)
        '''

        self.joint_transform = joint_transform
        self.local_reference = local_reference

    def __len__(self):
        nof = np.size(self.joint_transform, axis=0)
        return nof

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        joint_transform = self.joint_transform[idx]
        F = self.local_reference[idx]
        sample = {'frame_idx': idx, 'joint_transform': joint_transform, 'F': F}
        return sample
        



