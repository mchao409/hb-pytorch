"""
Tests on torch.arange
04/14/2020 Michelle Chao (mc2244@cornell.edu)
"""
import torch

def _test_torch_arange(start, end, step):
  assert torch.allclose(torch.arange(start,end,step), 
  torch.arange(start,end,step,device=torch.device("hammerblade")).cpu())

def _test_torch_arange_end(end):
  assert torch.allclose(torch.arange(end), 
  torch.arange(end,device=torch.device("hammerblade")).cpu())

def _test_torch_arange_start_end(start,end):
  assert torch.allclose(torch.arange(start,end), 
  torch.arange(start,end,device=torch.device("hammerblade")).cpu())

def test_torch_arange_1():
  _test_torch_arange(1,5,1)
  
def test_torch_arange_2():
  _test_torch_arange(1,10,3)

def test_torch_arange_3():
  _test_torch_arange(100,1000,4)

def test_torch_arange_4():
  _test_torch_arange(100,1000,11)

def test_torch_arange_5():
  _test_torch_arange(1,1,1)

def test_torch_arange_6():
  _test_torch_arange(-10,10,1)

def test_torch_arange_7():
  _test_torch_arange_end(5)

def test_torch_arange_8():
  _test_torch_arange_start_end(10,100)
