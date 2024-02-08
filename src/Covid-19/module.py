import torch.nn as nn
import torch.functional as F
class LR(nn.Module):
    def __init__(self): 
        super(LR, self).__init__()
        self.layer = nn.Linear(in_features, out_features)
