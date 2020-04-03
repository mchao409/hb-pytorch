"""
Unit tests for torch.mean
04/03/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
from math import isnan

torch.manual_seed(42)

def _test_torch_mean(tensor, dim=None, keepdim=False):
    tensor_h = tensor.hammerblade()
    if dim is None:
        mean_ = torch.mean(tensor_h)
        assert mean_.device == torch.device("hammerblade")
        assert torch.allclose(mean_.cpu(), torch.mean(tensor))
    else:
        mean_ = torch.mean(tensor_h, dim, keepdim=keepdim)
        assert mean_.device == torch.device("hammerblade")
        assert torch.allclose(mean_.cpu(), torch.mean(tensor, dim, keepdim=keepdim))

def test_torch_mean_1():
    x = torch.ones(10)
    _test_torch_mean(x)

def test_torch_mean_2():
    x = torch.ones(10)
    _test_torch_mean(x, dim=0)

def test_torch_mean_3():
    x = torch.ones(10)
    _test_torch_mean(x, dim=0, keepdim=True)

def test_torch_mean_4():
    x = torch.randn(3, 4)
    _test_torch_mean(x)

def test_torch_mean_5():
    x = torch.randn(3, 4)
    _test_torch_mean(x, dim=0)

def test_torch_mean_6():
    x = torch.randn(3, 4)
    _test_torch_mean(x, dim=0, keepdim=True)

def test_torch_mean_7():
    x = torch.randn(3, 4)
    _test_torch_mean(x, dim=1)

def test_torch_mean_8():
    x = torch.randn(3, 4)
    _test_torch_mean(x, dim=1, keepdim=True)

def test_torch_mean_9():
    x = torch.randn(3, 4)
    _test_torch_mean(x, dim=(0, 1))

def test_torch_mean_10():
    x = torch.randn(3, 4)
    _test_torch_mean(x, dim=(0, 1), keepdim=True)

def test_torch_mean_11():
    x = torch.randn(3, 4, 5)
    _test_torch_mean(x)

def test_torch_mean_12():
    x = torch.randn(3, 4, 5)
    _test_torch_mean(x, dim=0)

def test_torch_mean_13():
    x = torch.randn(3, 4, 5)
    _test_torch_mean(x, dim=0, keepdim=True)

def test_torch_mean_14():
    x = torch.randn(3, 4, 5)
    _test_torch_mean(x, dim=1)

def test_torch_mean_15():
    x = torch.randn(3, 4, 5)
    _test_torch_mean(x, dim=1, keepdim=True)

def test_torch_mean_16():
    x = torch.randn(3, 4, 5)
    _test_torch_mean(x, dim=2)

def test_torch_mean_17():
    x = torch.randn(3, 4, 5)
    _test_torch_mean(x, dim=2, keepdim=True)

def test_torch_mean_18():
    x = torch.randn(3, 4, 5)
    _test_torch_mean(x, dim=(0, 1))

def test_torch_mean_19():
    x = torch.randn(3, 4, 5)
    _test_torch_mean(x, dim=(0, 1), keepdim=True)

def test_torch_mean_20():
    x = torch.randn(3, 4, 5)
    _test_torch_mean(x, dim=(1, 2))

def test_torch_mean_21():
    x = torch.randn(3, 4, 5)
    _test_torch_mean(x, dim=(1, 2), keepdim=True)

def test_torch_mean_22():
    x = torch.randn(3, 4, 5)
    _test_torch_mean(x, dim=(0, 2))

def test_torch_mean_23():
    x = torch.randn(3, 4, 5)
    _test_torch_mean(x, dim=(0, 2), keepdim=True)

def test_torch_mean_24():
    x = torch.randn(3, 4, 5)
    _test_torch_mean(x, dim=(0, 1, 2))

def test_torch_mean_25():
    x = torch.randn(3, 4, 5)
    _test_torch_mean(x, dim=(0, 1, 2), keepdim=True)

def test_torch_mean_26():
    x = torch.tensor([[[1.], [2.], [3.]]])
    h = x.hammerblade()
    x = x.expand(2, 3, 4)
    h = h.expand(2, 3, 4)
    assert h.device == torch.device("hammerblade")
    assert not h.is_contiguous()
    mean_ = torch.mean(h)
    assert mean_.device == torch.device("hammerblade")
    assert torch.allclose(mean_.cpu(), torch.mean(x))

def test_torch_mean_27():
    x = torch.tensor([[[1.], [2.], [3.]]])
    h = x.hammerblade()
    x = x.expand(2, 3, 4)
    h = h.expand(2, 3, 4)
    assert h.device == torch.device("hammerblade")
    assert not h.is_contiguous()
    mean_ = torch.mean(h, (0, 2))
    assert mean_.device == torch.device("hammerblade")
    assert torch.allclose(mean_.cpu(), torch.mean(x, (0, 2)))

def test_torch_mean_28():
    x = torch.tensor([[[1.], [2.], [3.]]])
    h = x.hammerblade()
    x = x.expand(2, 3, 4)
    h = h.expand(2, 3, 4)
    assert h.device == torch.device("hammerblade")
    assert not h.is_contiguous()
    mean_ = torch.mean(h, (0, 2), keepdim=True)
    assert mean_.device == torch.device("hammerblade")
    assert torch.allclose(mean_.cpu(), torch.mean(x, (0, 2), keepdim=True))

def test_torch_mean_29():
    x = torch.rand(2, 3, 4, 5)
    _test_torch_mean(x)

def test_torch_mean_30():
    x = torch.rand(2, 3, 4, 5)
    for dim in range(4):
        _test_torch_mean(x, dim=dim)

def test_torch_mean_nan():
    x = torch.tensor([])
    h = x.hammerblade()
    assert isnan(x.mean())
    assert isnan(h.mean())
    assert h.mean().device == torch.device("hammerblade")
    assert h.mean().dtype == torch.float32
