import torch

class Normalization(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
class Standardization(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()

        self.mean = mean
        self.std = std
    
    def forward(self, x):
        return (x - self.mean) / self.std