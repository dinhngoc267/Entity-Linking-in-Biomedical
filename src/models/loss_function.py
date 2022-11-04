import torch
import torch.nn as nn

class TripletLosss(nn.Module):
  def __init__(self, margin):
    super().__init__()
    self.margin = margin
  
  def forward(self, anchor_pos: torch.Tensor, anchor_neg: torch.Tensor) -> torch.Tensor:
    loss = torch.sum(F.relu(anchor_neg - anchor_pos + self.margin))
    return loss