import math
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.distributed as dist
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from typing import Union,Tuple,List
from collections import OrderedDict
import numpy as np

def seq(items, as_sequential=True):
    if items is None: 
        return None
    if isinstance(items, nn.Module):
        return items
    if isinstance(items, (tuple, list)):
        if len(items) == 0: 
            return nn.Identity()
        elif len(items) == 1:
            return seq(items[0], as_sequential=as_sequential)
        elif as_sequential:
            return nn.Sequential(*[seq(item) for item in items if item is not None])
        else:
            return [seq(item) for item in items if item is not None]
    if isinstance(items, dict):
        if len(items) == 0:
            return nn.Identity()
        elif as_sequential:
            return nn.Sequential(OrderedDict({key: seq(item) for key,item in items.items() if item is not None}))
        else:
            return OrderedDict({key: seq(item) for key,item in items.items() if item is not None})
    assert False, "Unexpected type while unpacking sequential"

def all_combinations(*args):
    assert len(args)
    if len(args) == 1: return [[a] for a in args[0]]
    else: return [a + b for a in all_combinations(*args[:len(args)//2]) for b in all_combinations(*args[len(args)//2:])]

def all_cosine_similarities(a,b, absolute=False):
    assert a.dtype == b.dtype
    if a.dtype in { torch.cfloat, torch.cdouble }:
        if absolute: r = (a @ b.mT.conj()).abs()
        else: r = (a @ b.mT.conj()).real
        r = r / (a.abs().pow(2).sum(-1).unsqueeze(-1) * b.abs().pow(2).sum(-1).unsqueeze(-2)).sqrt()
    else:
        r = (a @ b.mT)
        r = r / (a.pow(2).sum(-1).unsqueeze(-1) * b.pow(2).sum(-1).unsqueeze(-2)).sqrt()
    return torch.nan_to_num(r)

def padstack(tensors, dim=0, pad_mode="constant", pad_value=0):
    assert len(tensors) > 0
    ndims = tensors[0].dim()
    if dim < 0: dim += ndims
    for t in tensors[1:]: assert t.dim() ==  ndims
    new_shape = [max([t.shape[d] for t in tensors]) for d in range(ndims)]
    padding = [[new_shape[d//2] - t.shape[d//2] if (d & 1) else 0 for d in range(ndims*2,0,-1)] for t in tensors]
    return torch.stack([F.pad(t,p,pad_mode,pad_value) for t,p in zip(tensors,padding)], dim=dim)

def padcat(tensors, dim=0, pad_mode="constant", pad_value=0):
    assert len(tensors) > 0
    ndims = tensors[0].dim()
    if dim < 0: dim += ndims
    for t in tensors[1:]: assert t.dim() ==  ndims
    new_shape = [max([t.shape[d] for t in tensors]) if d != dim else -1 for d in range(ndims)]
    padding = [[new_shape[d//2] - t.shape[d//2] if (d & 1 and d//2 != dim) else 0 for d in range(ndims*2,0,-1)] for t in tensors]
    return torch.cat([F.pad(t,p,pad_mode,pad_value) for t,p in zip(tensors,padding)], dim=dim)

def rgb_to_yuv(rgb: torch.Tensor, clamp=True) -> torch.Tensor:
    if rgb.dtype == torch.uint8: 
        rgb = rgb / 255
    m = torch.tensor([
        [ 0.21260,  0.71520, 0.07220],
        [-0.09991, -0.33609, 0.43600],
        [ 0.61500, -0.55861,-0.05639]],
        device=rgb.device)
    yuv = (m @ rgb.flatten(-2)).view(rgb.shape)
    if clamp:
        yuv.select(-3, 0).clamp_(0,1)
        yuv.select(-3, 1).clamp_(-1,1)
        yuv.select(-3, 2).clamp_(-1,1)
    return yuv

def yuv_to_rgb(yuv: torch.Tensor, clamp=True) -> torch.Tensor:
    m = torch.tensor([
        [1, 0.00000, 1.28033],
        [1,-0.21482,-0.38059],
        [1, 2.12798, 0.00000]], 
        device=yuv.device)
    rgb = (m @ yuv.flatten(-2)).view(yuv.shape)
    return rgb.clamp(0,1) if clamp else rgb

def nth_color(i: int, n: int) -> tuple: 
    """Pick color i out of n colors equaly divided over a color circle"""
    return (0.5+math.cos((i/n + 0/3) * 2 * math.pi)/2, 0.5+math.cos((i/n + 1/3) * 2 * math.pi)/2,0.5+math.cos((i/n + 2/3) * 2 * math.pi)/2) 

def nth_color_u8(i: int, n: int) -> tuple: 
    """Pick color i out of n colors equaly divided over a color circle"""
    return (int(127.5+math.cos((i/n + 0/3) * 2 * math.pi)*127.5), int(127.5+math.cos((i/n + 1/3) * 2 * math.pi)*127.5),int(127.5+math.cos((i/n + 2/3) * 2 * math.pi)*127.5)) 

def n_colors(n: int) -> list:
    return [nth_color(i, n) for i in range(n)]

def n_colors_u8(n: int) -> list:
    return [nth_color_u8(i, n) for i in range(n)]

def gaussian1d(kernel_size: int, sigma=None, sym:bool=True) -> torch.Tensor:
    if kernel_size == 1: return torch.ones(1)
    odd = kernel_size % 2
    if sigma is None:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    if not sym and not odd:
        kernel_size = kernel_size + 1
    n = torch.arange(0, kernel_size) - (kernel_size - 1.0) / 2.0
    sig2 = 2 * sigma * sigma
    w = torch.exp(-n ** 2 / sig2)
    if not sym and not odd:
        w = w[:-1]
    return w / w.sum()

def gaussian2d(kernel_size: int, std=None, sym:bool=True) -> torch.Tensor:
    w = gaussian1d(kernel_size, std, sym)
    w = torch.outer(w,w)
    return w

def multinomial_nd(t, eps=1e-10):
    dims = tuple(range(1,t.dim()))
    position = []
    while len(dims):
        s = t.sum(dims)
        p = torch.multinomial(s + eps, 1).item()
        position.append(p)
        dims = dims[:-1]
        t = t[p]
    p = torch.multinomial(t + eps,1).item()
    position.append(p)
    return tuple(position)

def unflatten_dict(d):
    def _insert(d, key, v):
        if "." in key:
            head,tail = key.split(".",1)
            d[head] =_insert(d[head] if head in d else {}, tail, v)
        else: d[key] = v
        return d
    r = {}
    for key in d: r = _insert(r, key, d[key])
    return r

def all_detach(data):
    if isinstance(data, torch.Tensor):
        return data.detach()
    if isinstance(data, list):
        return list([all_detach(d) for d in data])
    if isinstance(data, tuple):
        return tuple([all_detach(d) for d in data])
    if isinstance(data, dict):
        copy = {}
        for k in data:
            copy[k] = all_detach(data[k])
        return copy
    return data

def all_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, list):
        return list([all_to_device(d, device) for d in data])
    if isinstance(data, tuple):
        return tuple([all_to_device(d, device) for d in data])
    if isinstance(data, dict):
        copy = {}
        for k in data:
            copy[k] = all_to_device(data[k], device)
        return copy
    return data


def norm(t, mean=None, std=None, clamp=(None,None), eps:float=1e-8) -> torch.Tensor:
    if t.dtype == torch.uint8: 
        t = t / 255
    
    if mean is None:
        mean = t.mean((-1,-2), keepdim=True)
    if std is None:
        std = t.std((-1,-2), keepdim=True)
    
    if not isinstance(mean, (tuple, list, torch.Tensor)):
        t = t - mean 
    elif isinstance(mean, torch.Tensor):
        while mean.dim() < 3: mean = mean[...,None]
        t = t - mean
    else:
        assert len(mean) == t.shape[-3]
        t = torch.stack([t.select(-3,i) - v for i,v in enumerate(mean)]) 
    
    if not isinstance(std, (tuple, list, torch.Tensor)):
        if std < eps: std = eps
        t = t / std
    elif isinstance(mean, torch.Tensor):
        while std.dim() < 3: std = std[...,None]
        t = t / std.clamp(eps, None)
    else:
        assert len(std) == t.shape[-3]
        t = torch.stack([t.select(-3,i) / (eps if v < eps else v) for i,v in enumerate(std)]) 
    
    if t.dtype not in (torch.cfloat, torch.cdouble):
        if clamp[0] is not None or clamp[1] is not None:
            t = t.clamp(*clamp)
    
    return t

def denorm(t:torch.Tensor, mean=0.0, std=1.0, clamp=(0,1), eps:float=1e-8) -> torch.Tensor:
    if t.dtype == torch.uint8: 
        t = t / 255
        
    if not isinstance(std, (tuple, list, torch.Tensor)):
        t = t * std 
    elif isinstance(mean, torch.Tensor):
        while std.dim() < 3: std = std[...,None]
        t = t * std 
    else:
        assert len(std) == t.shape[-3]
        t = torch.stack([t.select(-3,i) * v for i,v in enumerate(std)]) 
    
    if not isinstance(mean, (tuple, list, torch.Tensor)):
        t = t + mean 
    elif isinstance(mean, torch.Tensor):
        while mean.dim() < 3: mean = mean[...,None]
        t = t + mean
    else:
        assert len(mean) == t.shape[-3]
        t = torch.stack([t.select(-3,i) + v for i,v in enumerate(mean)]) 
    
    if t.dtype not in (torch.cfloat, torch.cdouble):
        if clamp[0] is not None or clamp[1] is not None:
            t = t.clamp(*clamp)
    return t

def disp(t: torch.Tensor, permute=False, numpy=False, clamp=True, channels=None, device=None, dtype=torch.float, rotations=1, brightness=1, saturation=1, normalize=False) -> torch.Tensor:
    t = t.detach()
    
    dim = t.dim()
    while t.dim() < 4: t = t[None]
    t = t.view(-1, *t.shape[-3:])
    if channels is not None:
        if t.shape[-3] < channels:
            t = F.pad(t, [0,0,0,0,0,channels-t.shape[-3]], "constant", 0)
        else:
            t = t[:,:channels]
            
    elif channels is None and t.shape[-3] == 2:
        t = F.pad(t, [0,0,0,0,0,1], "constant", 0.5)
    
    if normalize:
        if t.dtype == torch.uint8: t = t / 255
        t = t.sub(t.mean((-1,-2), keepdim=True)).div(t.std((-1,-2),keepdim=True).add(1e-5))
        
    if channels is not None or t.shape[-3] > 4:
        channels = channels or t.shape[-3]
        s = torch.linspace(-0.5* math.pi, 0.5 * math.pi, channels).repeat(t.shape[0])
        s = torch.complex(s.cos(), s.sin()).to(t.device).view(t.shape[0],channels,1,1)
        gs = None
        for i in range(0,channels,32):
            g = t[...,i:i+32,:,:]
            if g.dtype == torch.uint8:
                g = g / 255
            g = torch.complex(g, torch.zeros_like(g)) * s[...,i:i+32,:,:]
            g = g.sum(-3, keepdim=True)
            gs = g if gs is None else g + gs
        
        t = gs
        #t = 0.5 * (t - t.mean((-1,-2),keepdim=True)) / t.std((-1,-2),keepdim=True)
        t = 0.5 * (t - t.mean()) / t.std()
        t.real = torch.nan_to_num(t.real)
        t.imag = torch.nan_to_num(t.imag)
        
    # convert to float
    if t.dtype == torch.uint8: 
        t = t / 255
    elif t.dtype in [torch.cfloat, torch.cdouble]:
        if rotations != 1:
            t = t ** rotations / t.abs() ** (rotations-1)
        y = t.abs()
        u,v = t.real, t.imag
        y = torch.cat((y,u,v), -3)
        t = yuv_to_rgb(y)
        
    if t.dtype in [torch.float, torch.double]:
        if brightness != 1:
            t = torch.nan_to_num(t ** (1/brightness))
        if saturation != 1:
            mean = t.mean(-3, keepdim=True)
            std = t.std(-3, keepdim=True)
            y = torch.nan_to_num((t - mean) / std)
            #mask = y < 0
            y = y * saturation
            t = y * std + mean
    else:
        assert False, f"input.dtype of {t.dtype} is not supported"
    
    if clamp:
        t = t.clamp(0,1)
    
    if dtype == torch.uint8:
        t = (t * 255).byte()
    
    if device is not None:
        t = t.to(device)
    
    if t.shape[1] == 1:
        t = t.repeat(1,3,*((1,) * (t.dim()-2)))
        
    if dim < 4: t = t[0]
    return t
    
class _ValueFunc(nn.Module):
    def __init__(self, func, *args, requires_grad=False, **kwargs):
        super().__init__()
        self.value = torch.nn.parameter.Parameter(func(*args, **kwargs), requires_grad=requires_grad)
        
    def forward(self, *args, **kwargs):
        return self.value
    
    def extra_repr(self):
        return f"value={self.value}"
   
class Zeros(_ValueFunc):
    def __init__(self, *args, **kwargs):
        super().__init__(torch.zeros, *args, **kwargs)

class ZerosLike(_ValueFunc):
    def forward(self, x):
        return torch.zeros_like(x)
        
class Ones(_ValueFunc):
    def __init__(self, *args, **kwargs):
        super().__init__(torch.ones, *args, **kwargs)

class OnesLike(_ValueFunc):
    def forward(self, x):
        return torch.zeros_like(x)

class Full(_ValueFunc):
    def __init__(self, *args, **kwargs):
        super().__init__(torch.full, *args, **kwargs)

class FullLike(_ValueFunc):
    def __init__(self, value):
        super().__init__()
        self.value = value
        
    def forward(self, x):
        return torch.full_like(x, self.value)

class Rand(_ValueFunc):
    def __init__(self, *args, **kwargs):
        super().__init__(torch.rand, *args, **kwargs)

class Randn(_ValueFunc):
    def __init__(self, *args, **kwargs):
        super().__init__(torch.randn, *args, **kwargs)

class RandnLike(nn.Module):
    def forward(self, x):
        return torch.randn_like(x)
        
class Tensor(_ValueFunc):
    def __init__(self, *args, **kwargs):
        super().__init__(torch.randn, *args, **kwargs)

class ZerosLike(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)

class OnesLike(nn.Module):
    def forward(self, x):
        return torch.ones_like(x)

class RandnLike(nn.Module):
    def forward(self, x):
        return torch.randn_like(x)

class RandLike(nn.Module):
    def forward(self, x):
        return torch.rand_like(x)

class FullLike(nn.Module):
    def __init__(self, fill_value):
        super().__init__()
        self.fill_value = fill_value
        
    def forward(self, x):
        return torch.full_like(x, fill_value)

class Iterate(nn.Module):
    def __init__(self, *args, loops=1):
        super().__init__()
        self.body = seq(args)
        self.loops = loops

    def forward(self, x):
        for _ in range(self.loops):
            x = self.body(x)
        return x

    def extra_repr(self):
        return f"loops={self.loops}"

class Get(nn.Module):
    def __init__(self, key=None):
        super().__init__()
        self.key = key

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            assert isinstance(self.key, int)
            return x[self.key]
        elif isinstance(x, (dict, OrderedDict)):
            if self.key is None:
                return x[next(iter(x.keys()))]
            else:
                return x[self.key]
        elif hasattr(x, self.key):
            return getattr(x, self.key)
        else:
            assert False, "invalid instance type"

class Set(nn.Module):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def forward(self, x):
        return { self.key: x }
        
def to_tensor(value, mod=None, x=None, device=None):
    if isinstance(value, str):
        value = eval(value)
    if callable(value): 
        value = value(mod,x)
    if isinstance(value, torch.Tensor): 
        return value.to(device)
    else:
        return torch.tensor(value, device=device)

class Multiply(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        
    def forward(self,x):
        x = x * to_tensor(self.value, mod=self, x=x, device=x.device)
        return x

class Add(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        
    def forward(self,x):
        x = x + to_tensor(self.value, mod=self, x=x, device=x.device)
        return x

class Sum(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs  = kwargs

    def forward(self, x):
        return x.sum(*self.args, **self.kwargs)

class Sqrt(nn.Module):
    def forward(self, x):
        return x.sqrt()
    
class Pow(nn.Module):
    def __init__(self, exponent):
        super().__init__()
        self.exponent = exponent
    
    def forward(self, x):
        return x.pow(self.exponent)
        
class MLP(nn.Sequential):
    def __init__(self, in_channels, out_channels, hidden_channels=None, layers=1, expand=4, act=nn.ReLU, dropout=0.0):
        super().__init__(
                *[nn.Sequential(nn.Linear(c, hidden_channels or in_channels*expand), act()) \
                  for c in ([in_channels] + [hidden_channels or in_channels*expand] * (layers-1))],
                nn.Linear(in_channels*expand, out_channels))

def flatten_dims(x, dims):
    sorted_dims = sorted([d if d >= 0 else len(x.shape)+d for d in dims])
    reshape = list(x.shape[:sorted_dims[0]]) + [-1] + list(x.shape[sorted_dims[-1]+1:])
    return x.reshape(*reshape)

class FlattenDims(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    
    def forward(self, x):
        return flatten_dims(x, self.dims)
        
    def extra_repr(self):
        return "dims=(" + ", ".join([str(v) for v in self.dims]) + ")"

def reshape_dim(x, dim, shape):
    reshape = x.shape[:dim] + shape + x.shape[dim+1:]
    return x.reshape(*reshape)

class ReshapeDim(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()
        self.shape = shape
        self.dim = dim

    def forward(self, x):
        return reshape_dim(x, self.dim, self.shape)
        
    def extra_repr(self):
        return "shape=(" + ", ".join([str(v) for v in self.shape]) + "), dim=" + str(self.dim)
        
class Affine(nn.Module):
    def __init__(self, multiply=1, add=0):
        super().__init__()
        self.multiply = multiply
        self.add = add
        
    def forward(self,x):
        x = x * to_tensor(self.multiply, mod=self, x=x, device=x.device) + to_tensor(self.add, mod=self, x=x, device=x.device)
        return x

class Contiguous(nn.Module):
    def forward(self, x):
        return x.contiguous()
    
    
class Ignore(nn.Module):
    def __init__(self, *args,**kwargs):
        super().__init__()
        
    def forward(self, x):
        return x
    
class Sequential(nn.Sequential):
    def __init__(self, *args):
        body = seq(args, as_sequential=False)
        if isinstance(body, OrderedDict):
            super().__init__(body)
        elif isinstance(body, nn.Module):
            super().__init__(body)
        else:
            super().__init__(*body)

class Tee(Sequential):
    def __init__(self, *args, clone=False):
        super().__init__(args)
        self.clone = clone
        
    def forward(self, x):
        if self.clone: super().forwarD(x.clone())
        else: super().forward(x)
        return x
        
    def extra_repr(self):
        return "clone=True" if self.clone else ""
        
class Skip(nn.Sequential):
    def __init__(self, *args, reduction="+", drop_path=0, dim=None):
        body = seq(args, as_sequential=False)
        
        if isinstance(body, OrderedDict):
            super().__init__(body)
        else:
            super().__init__(*body)
        self.reduction = reduction
        self.drop_path = drop_path
        if self.reduction in ("cat","stack"):
            self.dim = dim or 1
            
    def forward(self, x):
        if self.training and self.drop_path > 0 and torch.rand((1,)) < self.drop_path: return x
        if self.reduction == None: return x
        
        y = super().forward(x)
        if self.reduction == "+": return x + y
        if self.reduction == "add": return x + y
        if self.reduction == "-": return x - y
        if self.reduction == "sub": return x - y
        if self.reduction == "*": return x * y
        if self.reduction == "mul": return x * y
        if self.reduction == "/": return x / y
        if self.reduction == "div": return x / y
        if self.reduction == "cat": return torch.cat((x,y), self.dim)
        if self.reduction == "stack": return torch.stack((x,y), self.dim)
        assert False, "invalid reduction"

    def extra_repr(self):
        if self.reduction == "cat":
            return f"reduction=\"{self.reduction}\", dim={self.dim}"  
        elif self.reduction != "+":
            return  f"reduction=\"{self.reduction}\""
        else:
            return ""
            
class Parameter(nn.Module):
    def __init__(self, shape=None, init=None):
        if shape is None:
            assert isinstance(init, torch.Tensor)
        super().__init__()
        self.shape = shape or init.shape
        self.init = init or 0
        self.value = nn.parameter.Parameter(torch.empty(*self.shape))
        
        init = self.init
        if isinstance(init, str):
            init = eval(init)
        if callable(init):
            init(self.value)
        elif isinstance(init, torch.Tensor):
            with torch.no_grad():
                self.value.copy_(init)
        else:
            with torch.no_grad():
                self.value.copy_(torch.full(self.shape, init))
        
    def forward(self, x):
        return self.value
         
def nan_to_num(input, nan=0.0, posinf=None, neginf=None):
    if input.dtype in [torch.cfloat, torch.cdouble]:
        real = torch.nan_to_num(input.real, nan=nan, posinf=posinf, neginf=neginf)
        imag = torch.nan_to_num(input.imag, nan=nan, posinf=posinf, neginf=neginf)
        return torch.complex(real, imag)
    else:
        return torch.nan_to_num(input, nan=nan, posinf=posinf, neginf=neginf)
        
class NanToNum(nn.Module):
    def __init__(self, nan=0.0, posinf=None, neginf=None):
        super().__init__()
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf
        
    def forward(self, x):
        return nan_to_num(x, self.nan, self.posinf, self.neginf)
    
class Select(nn.Module):
    def __init__(self, dim, index):
        super().__init__()
        self.dim = dim
        self.index = index
        
    def forward(self, x):
        if isinstance(self.index, (list, tuple)):
            return torch.stack([x.select(self.dim, idx) for idx in self.index], dim=self.dim)
        else:
            return x.select(self.dim, self.index)
    def extra_repr(self):
        return f"dim={self.dim}, index={self.index}"
    
class CenterCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size if isinstance(size, tuple) else (size, size)
    
    def forward(self ,x):
        return TF.center_crop(x, self.size)

    def extra_repr(self):
        return "size=" + str(self.size)

class CenterCrop2d(CenterCrop): ... 
class CenterCrop3d(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        assert self.size <= x.shape[-1] # for now no padding
        assert self.size <= x.shape[-2]
        assert self.size <= x.shape[-3]
        from_x = (x.shape[-1] - self.size) // 2
        from_y = (x.shape[-2] - self.size) // 2
        from_z = (x.shape[-3] - self.size) // 2
        return x[...,
                 from_z:from_z+self.size, 
                 from_y:from_y+self.size, 
                 from_x:from_x+self.size]

class View(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args
        
    def forward(self, x):
        return x.view(*self.args)

    def extra_repr(self):
        r = ""
        s = ""
        for arg in self.args:
            r += s + str(arg)
            s = ", "
        return r
    
class Abs(nn.Module):
    def forward(self, x):
        return x.abs()

class Real(nn.Module):
    def forward(self, x):
        return x.real

class Imag(nn.Module):
    def forward(self, x):
        return x.imag

class Transpose(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args
    
    def forward(self, x):
        return x.transpose(*self.args)
    
    def extra_repr(self):
        r = ""
        s = ""
        for arg in self.args:
            r += s + str(arg)
            s = ", "
        return r

class Pad(nn.Module):
    def __init__(self, padding, mode="constant", pad_value=0):
        super().__init__()
        self.padding = padding
        self.mode = mode
        self.pad_value = pad_value

    def forward(self, x):
        return F.pad(x, self.padding, self.mode, self.pad_value)

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args
    
    def forward(self, x):
        return x.permute(*self.args)
    
    def extra_repr(self):
        r = ""
        s = ""
        for arg in self.args:
            r += s + str(arg)
            s = ", "
        return r
    
class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args
    
    def forward(self, x):
        return x.reshape(*self.args)

    def extra_repr(self):
        r = ""
        s = ""
        for arg in self.args:
            r += s + str(arg)
            s = ", "
        return r
    
class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode="nearest"):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        assert size is not None or scale_factor is not None
        
    def forward(self, x):
        return interpolate(x, self.size, self.scale_factor, self.mode)
        

    def extra_repr(self):
        a = f"size={self.size}, " if self.size is not None else ""
        b = f"scale_factor={self.scale_factor}, " if self.scale_factor is not None else ""
        return a + b + f"mode={self.mode}"
    
def interpolate(x, size=None, scale_factor=None, mode="bilinear"):
        if x.dtype in [torch.cfloat, torch.cdouble]:
            real = F.interpolate(x.real, size = size, scale_factor=scale_factor, mode=mode)
            imag = F.interpolate(x.imag, size = size, scale_factor=scale_factor, mode=mode)
            return torch.complex(real, imag)
        else:
            return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)

class Repeat(nn.Module):
    '''Repeats the input'''
    def __init__(self, *args):
        super().__init__()
        self.args = args
        
    def forward(self, x):
        return x.repeat(*self.args)

    def extra_repr(self):
        return ",".join([str(a) for a in self.args])

class Autocontrast(nn.Module):
    def forward(self, x):
        return TF.autocontrast(x)

class RandomRotation(nn.Module):
    def __init__(self, angle=180, fill=0):
        super().__init__()
        self.angle = angle
        self.fill = fill
    
    def forward(self,x):
        angle = torch.rand(1).item() * self.angle * 2 - self.angle
        return TF.rotate(x, self.angle, fill=self.fill)

    def extra_repr(self):
        return f"angle={self.angle}, fill={self.fill}"
    
class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=3, sigma=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def forward(self, x):
        y = []
        for i,p in enumerate(x):
            y.append(TF.gaussian_blur(p, self.kernel_size, sigma=self.sigma))
        return torch.stack(y)

    def extra_repr(self):
        return f"kernel_size={self.kernel_size}, sigma={self.sigma}"
    
class PoissonNoise(nn.Module):
    def __init__(self, well_capacity=10500):
        super().__init__()
        self.well_capacity = well_capacity
        
    def forward(self, x):
        discrete = x * self.well_capacity
        return (torch.normal(discrete, discrete.sqrt()) / self.well_capacity).clamp(0,1)

    def extra_repr(self):
        return f"well_capacity={self.well_capacity}"
    
class Clamp(nn.Module):
    def __init__(self, min=None, max=None):
        super().__init__()
        self.min = min
        self.max = max
    
    def forward(self, x):
        return x.clamp(self.min, self.max)
        
    def extra_repr(self):
        return f"min={self.min}, max={self.max}"
    
class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim    
        
    def forward(self, x):
        return x.squeeze(self.dim)

    def extra_repr(self):
        return f"dim={self.dim}"
    
class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim    
    
    def forward(self, x):
        return x.unsqueeze(self.dim)
    
    def extra_repr(self):
        return f"dim={self.dim}"

def right_flatten(x, dim=0):
    if not dim: return x.flatten()
    return x.view(-1, *x.shape[dim:])

class RightFlatten(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return right_flatten(x, self.dim)
    
class Every(nn.Module):
    '''Will be called every <n>th pass through the model'''
    def __init__(self, n, *args):
        super().__init__()
        self.n = n
        self.i = 0
        self.body = seq(*args)
        
    def forward(self, x):
        self.i += 1
        if self.i == self.n:
            self.i = 0
            return self.body(x)
        else:
            return x

    def extra_repr(self):
        return f"n={self.n}"

class Lambda(nn.Module):
    def __init__(self, callback, *args, **kwargs):
        super().__init__()
        self.callback = callback
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        callback = self.callback
        if isinstance(callback, str) and callback.startswith("lambda"):
            callback = eval(callback)
        assert callable(callback)
        return callback(self, x, *self.args, **self.kwargs)
    
    def extra_repr(self):
        return f"{str(self.callback)}"
    
class Print(nn.Module):
    def __init__(self, value=None, end="\n"):
        super().__init__()
        self.value = value
        self.end = end
        
    def forward(self, x):
        value = self.value
        if isinstance(value, str) and value.startswith("lambda "):
            value = eval(value)
        if callable(value):
            value = value(self, x)
        value = str(value)
        print(value, end=self.end)
        return x

    def extra_repr(self):
        return f"{str(self.value)}, end={self.end}"

class Hyper(nn.Module):
    def __init__(self, params={}, strict=False, **kwargs):
        super().__init__()
        self.params = params
        for name, value in kwargs.items():
            if strict: assert name in params
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            self.register_buffer(name, value)
        for name, value in params:
            if name not in kwargs:
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value)
                self.register_buffer(name, value)
                
    def extra_repr(self):
        s = []
        for name, b in self.named_buffers():
            if b.numel() == 1:
                s.append(f"{name}: {b.item()}")
            else:
                s.append(f"{name}: {b}")
        return "\n".join(s)


class Sharpen(nn.Module):
    def __init__(self, strength=1):
        super().__init__()
        self.strength = strength
        g = torch.tensor([-0.5,-1,-0.5,-1,6,-1,-0.5,-1,-0.5]).view(3,3) / 6
        self.register_buffer("kernel", g)
        
    def forward(self, x):
        padding=1
        x = F.pad(x, (padding,)*4, mode="reflect")
        k = self.kernel * self.strength + torch.tensor([0]*4+[1]+[0]*4, device=self.kernel.device).view(3,3)
        return F.conv2d(x, k.repeat(x.shape[1],1,1,1), groups=x.shape[1])

class Sharpen2d(Sharpen):...
class Sharpen3d(nn.Module):
    def __init__(self, strength=1):
        super().__init__()
        self.strength = strength
        g = torch.tensor([-0.25,-0.5,-0.25,-0.5,-1,-0.5,-0.25,-0.5,-0.25,
                          -0.5, -1, -0.5, -1, 14, -1,-0.5,-1,-0.5,
                          -0.25,-0.5,-0.25,-0.5,-1,-0.5,-0.25,-0.5,-0.25]).view(3,3,3) / 14
        self.register_buffer("kernel", g)
    
    def forward(self, x):
        padding=1
        x = F.pad(x, (padding,)*6, mode="reflect")
        k = self.kernel * self.strength + torch.tensor([0]*13+[1]+[0]*13, device=self.kernel.device).view(3,3,3)
        return F.conv3d(x, k.repeat(x.shape[1],1,1,1,1), groups=x.shape[1])

class Blur(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        g = gaussian1d(kernel_size)
        g = g[None,:] * g[:,None]
        self.register_buffer("kernel", g)
        
    def forward(self, x):
        padding=self.kernel.shape[-1]//2
        x = F.pad(x, (padding,)*4, mode="reflect")
        return F.conv2d(x, self.kernel.repeat(x.shape[1],1,1,1), 
                        groups=x.shape[1])

class Blur2d(Blur):...
class Blur3d(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        g = gaussian1d(kernel_size)
        g = g[None,None,:] * g[None,:,None] * g[:,None,None]
        self.register_buffer("kernel", g)
    
    def forward(self, x):
        padding=self.kernel.shape[-1]//2
        x = F.pad(x, (padding,)*6, mode="reflect")
        return F.conv3d(x, self.kernel.repeat(x.shape[1],1,1,1,1), 
                        groups=x.shape[1])

class Downsample(nn.Module):
    def __init__(self, dim, blur=True):
        super().__init__()
        if blur:
            self.body = nn.Sequential(
                Blur2d(), 
                nn.Conv2d(dim,dim,3,stride=2,padding=1, padding_mode="replicate"))
        else:
            self.body = nn.Conv2d(dim,dim,3,stride=2,padding=1, padding_mode="replicate")

    def forward(self, x):
        return self.body(x)

class Downsample2d(Downsample):...
class Downsample3d(nn.Module):
    def __init__(self, dim, blur=True):
        super().__init__()
        if blur:
            self.body = nn.Sequential(
                Blur3d(), 
                nn.Conv3d(dim,dim,3,stride=2,padding=1, padding_mode="reflect"))
        else:
            self.body = nn.Conv3d(dim,dim,3,stride=2,padding=1, padding_mode="reflect")

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim, blur=False, sharpen=0, mode="learned"):
        super().__init__()
        
        assert mode in ["learned", "bilinear", "bicubic", "nearest"]
        self.mode = mode
        
        if mode == "learned":
            self.upsample = nn.ConvTranspose2d(dim,dim,4,stride=2,padding=1),
        elif mode in ["biilinear", "bicubic", "nearest"]:
            self.upsample = Interpolate(scale_factor=2, mode=mode)
        
        if blur:
            self.blur = Blur2d(),
        else:
            self.register_module("blur",None)

        if sharpen > 0:
            self.sharpen = Sharpen2d(strength=sharpen)
        else:
            self.register_module("sharpen", None)

    def forward(self, x):
        x = self.upsample(x)
        if self.blur is not None:
            x = self.blur(x)
        if self.sharpen is not None:
            x = self.sharpen(x)
        return x

class Upsample2d(Upsample):...
class Upsample3d(nn.Module):
    def __init__(self, dim, blur=False, sharpen=0, mode="learned"):
        super().__init__()
        
        assert mode in ["learned", "trilinear", "nearest"]
        self.mode = mode
        
        if mode == "learned":
            self.upsample = nn.ConvTranspose3d(dim,dim,4,stride=2,padding=1),
        elif mode in ["trilinear", "nearest"]:
            self.upsample = Interpolate(scale_factor=2, mode=mode)
        
        if blur:
            self.blur = Blur3d(),
        else:
            self.register_module("blur",None)

        if sharpen > 0:
            self.sharpen = Sharpen3d(strength=sharpen)
        else:
            self.register_module("sharpen", None)

    def forward(self, x):
        x = self.upsample(x)
        if self.blur is not None:
            x = self.blur(x)
        if self.sharpen is not None:
            x = self.sharpen(x)
        return x

class ReshapeDims(nn.Module):
    def __init__(self, shape, dim, dim_to=None):
        super().__init__()
        self.shape = shape
        self.dim = dim
        self.dim_to = dim if dim_to is None else dim_to

    def forward(self, x):
        dim = self.dim
        if dim < 0: dim = x.dim() + dim
        dim_to = self.dim_to
        if dim_to < 0: dim_to = x.dim() + dim_to    
        s = *x.shape[:dim], *self.shape, *x.shape[dim_to+1:]
        return x.reshape(*s)
        
class PadToMultipleOf(nn.Module):
    def __init__(self, multiple_of, pad_mode="constant", pad_value=0):
        super().__init__()
        self.multiple_of = multiple_of
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        
    def forward(self, x):
        pad = []
        for i in reversed(range(2, x.dim())):
            s = x.shape[i]
            pad.extend((0,(self.multiple_of - (s % self.multiple_of)) % self.multiple_of))
        result = F.pad(x, pad, self.pad_mode, self.pad_value)
        return result


def voxel_unshuffle(x, kernel_size):
    s = x.shape
    x = x.view(-1, kernel_size, *s[-2:])  # [N...] * Z//k, k, H, W
    x = F.pixel_unshuffle(x, kernel_size) # [N...] * Z//k, k**3, H//k, W//k
    x = x.view(*s[:-3], -1, *x.shape[-3:])        # [N...], Z//k, k**3, H//k, W//k
    x = x.transpose(-4,-3)                     # [N...], k**3, Z//k, H//k, W//k
    x = x.reshape(*x.shape[:-5], -1, *x.shape[-3:])
    return x

class VoxelUnshuffle(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        return voxel_unshuffle(x, self.kernel_size)

def voxel_shuffle(x, kernel_size):
    s = x.shape
    x = x.view(*s[:-4],-1, kernel_size**3, *s[-3:])
    x = x.transpose(-4, -3)                  # [N...], Z//k, k**3, H//k, W//k
    x = x.reshape(-1, *x.shape[-3:])         # [N...] * Z//k, k**3, H//k, W//k
    x = F.pixel_shuffle(x, kernel_size) # [N...] * Z//k, k, H, W
    x = x.reshape(*s[:-4], -1, s[-3]*kernel_size,  *x.shape[-2:])  # [N...], Z, H, W
    return x

class VoxelShuffle(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size=kernel_size
    
    def forward(self, x):
        return voxel_shuffle(x, self.kernel_size)

class PixelUnshuffle3d(VoxelUnshuffle):...
class PixelShuffe3d(VoxelShuffle):...

def vector_length(x, eps=1e-5):
    return x.mul(x).sum(1, keepdim=True).add(eps).sqrt()

class VectorLength(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        return x / vector_length(x, eps=self.eps)        

def vector_norm(x, eps=1e-5):
    return x / vector_length(x, eps=eps)

class VectorNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        return vector_norm(x, eps=self.eps)        

def sobel3d(x):
    assert x.dim() >= 4
    k = torch.tensor([
        [ 1, 2, 1,
          0, 0, 0,
         -1,-2,-1 ],
        [ 2, 4, 2,
          0, 0, 0,
         -2,-4,-2 ],
        [ 1, 2, 1,
          0, 0, 0,
         -1,-2,-1 ]], device=x.device, dtype=torch.float).view(1,3,3,3) / 16
    k = torch.stack((k,k.transpose(-1,-2),k.transpose(-1,-3)))
    return F.conv3d(x,k.repeat(x.shape[-4],1,1,1,1), padding=1, groups=x.shape[-4])

class Sobel3d(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([
            [ 1, 2, 1,
              0, 0, 0,
             -1,-2,-1 ],
            [ 2, 4, 2,
              0, 0, 0,
             -2,-4,-2 ],
            [ 1, 2, 1,
              0, 0, 0,
             -1,-2,-1 ]], dtype=torch.float).view(1,3,3,3) / 16
        k = torch.stack((k,k.transpose(-1,-2),k.transpose(-1,-3)))
        self.register_buffer("kernel", k)
        
    def forward(self, x):
        assert x.dim() == 5
        return F.conv3d(x,self.kernel.repeat(x.shape[1],1,1,1,1), padding=1, groups = x.shape[1])
        

class RandomRotationNd(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        angle = torch.rand(1).item() * 360
        keep = torch.arange(x.dim() - self.dims)
        perm = -torch.randperm(self.dims)-1
        x = x.permute(*[k.item() for k in keep], *[p.item() for p in perm])
        rad = math.pi * angle / 180
        scale = abs(math.sin(rad)) + abs(math.cos(rad))
        for i in range(0, x.shape[-3],8):
            v = x[...,i:i+8,:,:]
            w = v.view(-1, *v.shape[-3:])
            w = TF.rotate(w, angle)
            v = w.view(*v.shape)
            x[...,i:i+8,:,:] = v
        s = x.shape
        x = F.interpolate(x, scale_factor=scale, mode="nearest")
        x = TF.center_crop(x, s[-2:])
        perm = -torch.randperm(self.dims)-1
        x = x.permute(*[k.item() for k in keep], *[p.item() for p in perm])
        return x

class RandomRot90Nd(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        dims = -torch.randperm(self.dims)[:2]-1
        dims = [d.item() for d in dims]
        rot = torch.randint(4, (1,)).item()
        return x.rot90(rot, dims)

class RandomPermuteNd(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        perm = -torch.randperm(self.dims)-1
        keep = torch.arange(x.dim() - self.dims)
        return x.permute(*[k.item() for k in keep], *[p.item() for p in perm])

class RandomFlipNd(nn.Module):
    def  __init__(self, dims, p=0.5):
        super().__init__()
        self.dims = dims
        self.p = p
        
    def forward(self, x):
        for i in range(self.dims):
            if torch.rand(1) < self.p:
                x = x.flip(-i-1)
        return x

class ToDevice(nn.Module):
    def __init__(self, device, non_blocking=False):
        super().__init__()
        self.device= device
        self.non_blocking = non_blocking
        
    def forward(self, x):
        return x.to(self.device, non_blocking=self.non_blocking)
 

def plot(t, title="", width=20, height=None, cols=None, axis="off", **kwargs):
    if isinstance(t, str):
        t = torchvision.io.read_image(t)
    if isinstance(t, np.ndarray): t = torch.from_numpy(t)
    if isinstance(t, (list,tuple)): t = torch.stack(t)
        
    if t.dim() == 1:
        height = height or width
        plt.figure(figsize = (width, height))
        plt.title(title)
        plt.plot(t)
    else:
        while t.dim() < 4: t = t[None]
        t = t.view(-1, *t.shape[-3:])
        n,c,h,w = t.shape
        t = disp(t, **kwargs)
        grid = make_grid(t, cols or n)
        height = height or (width * grid.shape[-2] / grid.shape[-1])
        plt.figure(figsize = (width, height))
        plt.title(title)
        plt.axis(axis)
        plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.show()
    plt.close()

def save(t, filename, cols=None, normalize=False, **kwargs):
    if isinstance(t, np.ndarray): t = torch.from_numpy(t)
    while t.dim() < 4: t = t[None]
    n,c,h,w = t.shape
    if normalize:
        t = norm(t)
    t = disp(t, dtype=torch.uint8, **kwargs)
    if n > 1:
        t = make_grid(t, cols or n)
    else:
        t = t[0]
    if filename.endswith(".png"):
        torchvision.io.write_png(t.cpu(), filename)
    elif filename.endswith(".jpg"):
        torchvision.io.write_jpeg(t.cpu(), filename)
    else:
        assert False
class Plot(nn.Module):
    def __init__(self, title="", width=20, cols=None, axis="off", **kwargs):
        super().__init__()
        self.title = title
        self.width = width
        self.axis = axis
        self.cols = cols
        self.kwargs = kwargs
        
    def forward(self, x):
        plot(x, self.title, self.width, self.cols, self.axis,**self.kwargs)
        return x

    def extra_repr(self):
        return f"title={self.title}, width={self.width}, axis={self.axis}, cols={self.cols}"

class Save(nn.Module):
    def __init__(self, filename, cols=None, **kwargs):
        super().__init__()
        self.filename = filename
        self.cols = cols
        self.kwargs = kwargs
        self.count = 0

    def forward(self, x):
        fn = self.filename.format(self.count)
        self.count += 1
        save(x, fn, self.cols, **self.kwargs)
        return x

    def extra_repr(self):
        return f"title={self.title}, width={self.width}, axis={self.axis}, cols={self.cols}"
    
class NoGrad(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.body = seq(args)
    
    def forward(self, x, *args):
        with torch.no_grad():
            return self.body(x, *args)

class Mean(nn.Module):
    def __init__(self, dim, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        
    def forward(self, x):
        y = x.mean(self.dim, keepdim=self.keepdim)
        return y

    def extra_repr(self):
        return f"{self.dim}, keepdim={self.keepdim}"
    
class Std(nn.Module):
    def __init__(self, dim, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
    def forward(self, x):
        y = x.std(self.dim, keepdim=self.keepdim)
        return y

    def extra_repr(self):
        return f"{self.dim}, keepdim={self.keepdim}"
    
class Log(nn.Module):
    def forward(self,x):
        return x.log()

class Exp(nn.Module):
    def forward(self,x):
        return x.exp()

class Conjugate(nn.Module):
    def forward(self, x):
        return x.conj()

    
class FFT(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        y = torch.fft.fft(x, dim=self.dim)
        return y

    def extra_repr(self):
        return f"dim={self.dim}"
    
class ChannelAttention(nn.Module):
    def __init__(self, num_channels, kernel_size=3, reduction=8, padding=None, bias=False, dropout=0,  act=nn.ReLU):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.bias = bias
        self.padding = padding if padding is not None else kernel_size // 2
        rc = max(num_channels//reduction, 1)
        
        self.channel_max = nn.Sequential(
            nn.MaxPool2d(kernel_size, stride=1, padding=self.padding),
            nn.Conv2d(num_channels, rc, 1),
            act(),
            nn.Conv2d(rc, num_channels, 1, bias=bias))
        
        self.channel_avg = nn.Sequential(
            nn.AvgPool2d(kernel_size, stride=1, padding=self.padding),
            nn.Conv2d(num_channels, rc, 1),
            act(),
            nn.Conv2d(rc, num_channels, 1, bias=bias))
            
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x):
        a = self.dropout(x)
        a = self.channel_max(a) + self.channel_avg(a)
        return x * torch.sigmoid(a)

class ChannelAttention2d(ChannelAttention):...
class ChannelAttention3d(nn.Module):
    def __init__(self, num_channels, kernel_size=3, reduction=8, padding=None, bias=False, dropout=0,  act=nn.ReLU):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.bias = bias
        self.padding = padding if padding is not None else kernel_size // 2
        rc = max(num_channels//reduction, 1)
        
        self.channel_max = nn.MaxPool3d(kernel_size, stride=1, padding=self.padding)
        self.channel_avg = nn.AvgPool3d(kernel_size, stride=1, padding=self.padding)
        self.mlp = nn.Sequential(
            nn.Conv3d(num_channels, rc, 1),
            act(),
            nn.Conv3d(rc, num_channels, 1, bias=bias))
            
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        a = self.dropout(x)
        a = self.mlp(self.channel_max(a)) + self.mlp(self.channel_avg(a))
        return x * torch.sigmoid(a)


class SqueezeExcitation(nn.Module):
    def __init__(self, num_channels, reduction=None, act=nn.ReLU):
        super().__init__()
        self.reduction =  reduction or int(math.ceil(num_channels ** 0.5))
        self.conv1 = nn.Conv2d(num_channels, num_channels // self.reduction, 1)
        self.conv2 = nn.Conv2d(num_channels // self.reduction, num_channels, 1)
        self.act = act()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.num_channels = num_channels
        
    def forward(self, x):
        s = self.pool(x)
        s = self.act(self.conv1(s))
        s = torch.sigmoid(self.conv2(s))
        return x * s

    def extra_repr(self):
        r = f"num_channels={self.num_channels}, reduction={self.reduction}, act={self.act}"
        return r

class SqueezeExcitation2d(SqueezeExcitation):...
class SqueezeExcitation3d(nn.Module):
    def __init__(self, num_channels, reduction=None, act=nn.ReLU):
        super().__init__()
        self.reduction =  reduction or int(math.ceil(num_channels ** 0.5))
        self.conv1 = nn.Conv3d(num_channels, num_channels // self.reduction, 1)
        self.conv2 = nn.Conv3d(num_channels // self.reduction, num_channels, 1)
        self.act = act()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.num_channels = num_channels
        
    def forward(self, x):
        if isinstance(x, list):
            s = []
            for v in x:
                s.append(self.pool(v))
            s = torch.cat(s,1)
        else:
            s = self.pool(x)
        
        s = self.act(self.conv1(s))
        s = torch.sigmoid(self.conv2(s))
        if isinstance(x,list):
            r = []
            i = 0
            for v in x:
                n = v.shape[1]
                r.append(v * s[:,i:i+n])
                i += n
        else:    
            r = x * s
        return r
        
    def extra_repr(self):
        r = f"num_channels={self.num_channels}, reduction={self.reduction}, act={self.act}"
        return r

class SpatialAttention2d(nn.Module):
    def __init__(self, groups=1, kernel_size=3, padding=None, bias=False, dropout=0):
        super().__init__()
        self.groups = groups
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = padding if padding is not None else kernel_size // 2
        
        self.maxpool = nn.AdaptiveMaxPool3d((groups, None, None))
        self.avgpool = nn.AdaptiveAvgPool3d((groups, None, None))
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(2 * groups, 1, kernel_size, padding=self.padding, bias=bias)
        
    def forward(self, x):
        a = torch.cat((self.maxpool(x), self.avgpool(x)), dim=-3)
        a = self.dropout(a)
        a = self.conv(a)
        return x * torch.sigmoid(a)

class SpatialAttention3d(nn.Module):
    def __init__(self, groups=1, kernel_size=3, padding=None, bias=False, dropout=0):
        super().__init__()
        self.groups = groups
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = padding if padding is not None else kernel_size // 2

        if groups > 1:
            self.maxpool = nn.AdaptiveMaxPool3d((groups, None, None))
            self.avgpool = nn.AdaptiveAvgPool3d((groups, None, None))
        
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv3d(2 * groups, 1, kernel_size, padding=self.padding, bias=bias)
        
    def forward(self, x):
        if self.groups > 1:
            y = x.transpose(-3,-4)
            a = torch.cat((self.maxpool(y), self.avgpool(y)), dim=-3)
            a = a.transpose(-3,-4)
        else:
            a = torch.stack((x.max(1)[0], x.mean(1)), dim=1)
        a = self.dropout(a)
        a = self.conv(a)
        return x * torch.sigmoid(a)
        
class AdaptiveBoxBlur2d(nn.Module):
    def forward(self, x, kernel_sizes, eps=1e-5):
        mean = x.mean((-1,-2), keepdim=True)
        std = x.std((-1,-2), keepdim=True)
        I = x.sub(mean).div(std.add(eps)).cumsum(-1).cumsum(-2)
        h,w = x.shape[-2:]
        grid_x = torch.linspace(-1,1,w).repeat(kernel_sizes.shape[0],h,1) - 1/w
        grid_y = torch.linspace(-1,1,h)[:,None].repeat(kernel_sizes.shape[0],1,w) - 1/h
        grid = torch.stack((grid_x, grid_y), -1)
        
        if kernel_sizes.dim() == x.dim()-1: 
            kernel_sizes = kernel_sizes[...,None].repeat(1,1,1,1,2)
        else:
            assert kernel_sizes.dim() == x.dim()
            assert kernel_sizes.shape[-1] == 2
        
        K = torch.stack((
            kernel_sizes[...,0] / w,
            kernel_sizes[...,1] / h), -1)
        gridA = grid + K
        gridD = grid - K
        
        K = torch.stack((
            K[...,0],
            -K[...,1]
        ), -1)
        gridB = grid + K
        gridC = grid - K
        
        A = F.grid_sample(I, gridA, padding_mode="border",align_corners=True)
        B = F.grid_sample(I, gridB, padding_mode="border",align_corners=True)
        C = F.grid_sample(I, gridC, padding_mode="border",align_corners=True)
        D = F.grid_sample(I, gridD, padding_mode="border",align_corners=True)
        result = A - B - C + D
        area = kernel_sizes.prod(-1)
        result = (result / area.add(eps)).mul(std).add(mean)
        return result

class AdaptiveBoxBlur3d(nn.Module):
    def forward(self, x, kernel_sizes, out_size=None, eps=1e-5):
        mean = x.mean((-1,-2,-3), keepdim=True)
        std = x.std((-1,-2,-3), keepdim=True)
        I = x.sub(mean).div(std.add(eps)).cumsum(-1).cumsum(-2).cumsum(-3)
        out_size = out_size or x.shape[-3:]
        d,h,w = out_size
        grid_x = torch.linspace(-1,1,w, device=x.device) [None,None,:].repeat(kernel_sizes.shape[0],d,h,1) - 1/w
        grid_y = torch.linspace(-1,1,h, device=x.device)[None,:,None].repeat(kernel_sizes.shape[0],d,1,w) - 1/h
        grid_z = torch.linspace(-1,1,d, device=x.device)[:,None,None].repeat(kernel_sizes.shape[0],1,h,w) - 1/d
        grid = torch.stack((grid_x, grid_y, grid_z), -1)
        
        if kernel_sizes.dim() == x.dim()-1: 
            kernel_sizes = kernel_sizes[...,None].repeat(1,1,1,1,1,3)
        else:
            assert kernel_sizes.dim() == x.dim()
            assert kernel_sizes.shape[-1] == 3
        
        K = torch.stack((
            kernel_sizes[...,0] / w,
            kernel_sizes[...,1] / h,
            kernel_sizes[...,2] / d), -1)
        
        # A - B - C - D + E + F + G - H
        gridA = grid + K
        gridH = grid - K
        k0, k1, k2 = K[...,0], K[...,1], K[...,2]
        
        gridB = grid + torch.stack((k0,k1,-k2),-1)
        gridC = grid + torch.stack((k0,-k1,k2),-1)
        gridD = grid + torch.stack((-k0,k1,k2),-1)
        
        gridE = grid + torch.stack((k0,-k1,-k2),-1)
        gridF = grid + torch.stack((-k0,-k1,k2),-1)
        gridG = grid + torch.stack((-k0,k1,-k2),-1)

        A = F.grid_sample(I, gridA, padding_mode="border",align_corners=True)
        B = F.grid_sample(I, gridB, padding_mode="border",align_corners=True)
        C = F.grid_sample(I, gridC, padding_mode="border",align_corners=True)
        D = F.grid_sample(I, gridD, padding_mode="border",align_corners=True)
        E = F.grid_sample(I, gridE, padding_mode="border",align_corners=True)
        _F = F.grid_sample(I, gridF, padding_mode="border",align_corners=True)
        G = F.grid_sample(I, gridG, padding_mode="border",align_corners=True)
        H = F.grid_sample(I, gridH, padding_mode="border",align_corners=True)
        result = A - B - C - D + E + _F + G - H
        area = kernel_sizes.prod(-1)[:,None]
        result = (result / area.add(eps)).mul(std).add(mean)
        return result

class AdaptiveBoxBlurNd(nn.Module):
    def __init__(self, kernel_sizes=None, channel_dim=1):
        super().__init__()
        assert channel_dim != 0
        self.channel_dim = channel_dim
        self.flows = None
        self.signs = None
        self.update_kernel_sizes(kernel_sizes)
            
    @staticmethod
    def _calculate_flows(k):
        dimensions = len(k.shape)-2
        assert k.shape[-1] == dimensions
        corners = all_combinations(*[[-1,1]]*dimensions)
        corners = torch.tensor(corners, device=k.device)
        grid = []
        flow = []
        for d in range(dimensions):
            s = k.shape[-d-2]
            i = [None]*dimensions
            i[-d-1] = slice(0,s)
            g = torch.linspace(-1,1,s, device=k.device)[tuple(i)]
            g = g.expand_as(k[...,0])
            g = g - 1/s
            grid.append(g)
            flow.append(k[...,d]/s)
        grid = torch.stack(grid, -1)
        flow = torch.stack(flow, -1)
        flows = []
        for c in corners:
            f = grid + c * flow 
            flows.append(f)
        signs = corners.prod(-1)
        return flows, signs
    
    @staticmethod
    def _window_mean(x, flows, signs, areas, channel_dim=-1, 
                         padding_mode="reflection", eps=1e-5):
    
        assert channel_dim != 0
        
        if channel_dim < 0: channel_dim = x.dim() + channel_dim
        excl_channel_dim = (i for i in range(x.dim()) if i != channel_dim)
        excl_channel_dim = tuple(excl_channel_dim)
        mean = x.mean(excl_channel_dim, keepdim=True)
        std = x.std(excl_channel_dim, keepdim=True)
        x = x.sub(mean).div(std.add(eps))
        for d in excl_channel_dim[1:]:
            x = x.cumsum(d)
        
        s = 0
        if channel_dim != 1:
            to_channels_first = (0,channel_dim) + excl_channel_dim[1:]
            x = x.permute(*to_channels_first)
        for f,sign in zip(flows, signs):
            s += sign * F.grid_sample(x, f, padding_mode=padding_mode, 
                                      align_corners=True)
        if channel_dim != 1:
            to_channels_orig = (to_channels_first.index(i) 
                                for i in range(x.dim()))
            s = s.permute(*to_channels_orig)
        
        return s.div(areas.add(eps)).mul(std).add(mean)
    
    def update_kernel_sizes(self, kernel_sizes):
        if kernel_sizes is not None:
            self.flows, self.signs = self._calculate_flows(kernel_sizes)
            self.kernel_sizes = kernel_sizes
    
    def forward(self, x, kernel_sizes=None):
        self.update_kernel_sizes(kernel_sizes)
        assert self.flows is not None
        assert self.signs is not None
        areas = self.kernel_sizes.prod(-1).unsqueeze(self.channel_dim)
        return self._window_mean(x, self.flows, self.signs, areas, 
                                 channel_dim=self.channel_dim)


class AdaptiveSpatialNorm(nn.Module):
    def __init__(self, channels, k_bias=3, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.xk = nn.Linear(channels, channels + 2)
        self.blur = AdaptiveBoxBlurNd(channel_dim=-1)
        self.k_bias = k_bias
        self.eps = eps
        
    def forward(self, x):
        x = x.permute(0,2,3,1)
        x,k = self.xk(x.contiguous()).split([self.channels, 2], dim=-1)
        k = k.add(self.k_bias).exp().add(1)
        mean = self.blur(x,k)
        stds = self.blur((x - mean).pow(2)).sqrt()
        return x.sub(mean).div(stds.add(self.eps)).permute(0,3,1,2)

class AdaptiveSpatialNorm2d(AdaptiveSpatialNorm): ...
class AdaptiveSpatialNorm3d(nn.Module):
    def __init__(self, channels, k_bias=3, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.xk = nn.Linear(channels, channels + 3)
        self.blur = AdaptiveBoxBlurNd(channel_dim=-1)
        self.k_bias = k_bias
        self.eps = eps
        
    def forward(self, x):
        x = x.permute(0,2,3,4,1)
        x,k = self.xk(x.contiguous()).split([self.channels, 3], dim=-1)
        k = k.add(self.k_bias).exp().add(1)
        mean = self.blur(x,k)
        stds = self.blur((x - mean).pow(2)).sqrt()
        return x.sub(mean).div(stds.add(self.eps)).permute(0,4,1,2,3)
        
class SpatialNorm2d(nn.Module):
    def __init__(self, channels, kernel_size, eps=1e-5, 
                 mode=None, affine=False):
        assert kernel_size % 2 == 1
        assert not affine, "TODO"
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.mode = mode or "pool"
        
        if mode == "gaussian":
            g = gaussian1d(kernel_size)
            g = g[None,:] * g[:,None]
            self.register_buffer("gaussian", g)
        else:
            self.register_buffer("gaussian", None)

    def forward(self, x):
        px = F.pad(x, (self.kernel_size//2,)*4, "reflect")
        if self.mode == "gaussian":
            mean = F.conv2d(px, self.gaussian.repeat(x.shape[0],1,1,1), groups=x.shape[0])
        elif self.mode == "pool":
            mean = F.avg_pool2d(px, self.kernel_size, stride=1)
            
        vars = x.sub(mean).pow(2)
        vars = F.pad(vars, (self.kernel_size//2,)*4, "reflect")
        
        if self.mode == "gaussian":
            vars = F.conv2d(vars, self.gaussian.repeat(x.shape[0],1,1,1), groups=x.shape[0])
        elif self.mode=="pool":
            vars = F.avg_pool2d(vars, self.kernel_size, stride=1)
        
        stds = vars.sqrt()
        return x.sub(mean).div(stds.add(self.eps))

class SpatialNorm3d(nn.Module):
    def __init__(self, channels, kernel_size, eps=1e-5, mode=None, affine=False):
        assert kernel_size % 2 == 1
        assert not affine, "TODO"
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.mode = mode or "pool"
        
        if mode == "gaussian":
            g = gaussian1d(kernel_size)
            g = g[None,None,:] * g[None,:,None] * g[:,None,None]
            self.register_buffer("gaussian", g)
        else:
            self.register_buffer("gaussian", None)

    def forward(self, x):
        px = F.pad(x, (self.kernel_size//2,)*6, "reflect")
        if self.mode == "gaussian":
            mean = F.conv3d(px, self.gaussian.repeat(x.shape[0],1,1,1,1), groups=x.shape[0])
        elif self.mode == "pool":
            mean = F.avg_pool3d(px, self.kernel_size, stride=1)
            
        vars = x.sub(mean).pow(2)
        vars = F.pad(vars, (self.kernel_size//2,)*6, "reflect")
        
        if self.mode == "gaussian":
            vars = F.conv3d(vars, self.gaussian.repeat(x.shape[0],1,1,1,1), groups=x.shape[0])
        elif self.mode=="pool":
            vars = F.avg_pool3d(vars, self.kernel_size, stride=1)
        
        stds = vars.sqrt()
        return x.sub(mean).div(stds.add(self.eps))

class SoftEmbedding(nn.Module):
    def __init__(self, dim, resolution=16, eye=False):
        assert resolution <= dim
        super().__init__()
        self.resolution = resolution
        embedding = torch.randn(dim, dim)
        svd = torch.linalg.svd(embedding)
        ortho = svd[0] @ svd[2]
        ortho = F.layer_norm(ortho, (dim,))
        self.embedding = nn.parameter.Parameter(ortho[:resolution])

    def forward(self, x):
        assert x.shape[1] == 1
        x = x.clamp(0,1)
        x = x * self.resolution
        x = x.squeeze(1)
        a = x.long()
        f = (x - a).unsqueeze(-1)
        va = self.embedding[a.cpu()] 
        vb = self.embedding[a.add(1).clamp(0,len(self.embedding)-1).cpu()]
        v = va * (1-f) + vb * f
        permute = [0, -1] + list(range(1,x.dim()))
        v = v.permute(*permute)
        return v
        
