"""
Tests on torch.eye (identity kernel)
04/10/2020 Michelle Chao (mc2244@cornell.edu)
"""
import torch

torch.manual_seed(42)

def test_torch():
  torch.eye(3,3, device=torch.device("hammerblade"))

