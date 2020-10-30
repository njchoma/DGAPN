import os
import torch


class MyDataset(torch.util.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.files = os.listdir(root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample, score = torch.load(os.path.join(self.root, self.files[idx]))
        return sample, score

