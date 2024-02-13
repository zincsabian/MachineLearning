import torch.nn as nn
import torch.functional as F

class LR(nn.Module):
    def __init__(self, in_features, out_features): 
        super(LR, self).__init__()
        self.layer1 = nn.Linear(in_features, out_features)

    def __init__(self):
        pass

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return x
    
    # y = relu(kx + b)