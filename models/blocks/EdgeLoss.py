import torch.nn as nn
import torch.nn.functional as F
import torch 

class EdgeLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        kernel = [[-1, -1, -1],
                  [-1,  8, -1],
                  [-1, -1, -1]]
        self.kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        
    def forward(self, pred, target):
        self.kernel = self.kernel.to(pred.device)
        edge_target = F.conv2d(target, self.kernel, padding=1)
        edge_pred = F.conv2d(pred, self.kernel, padding=1)
        return torch.mean(torch.abs(edge_pred - edge_target))
