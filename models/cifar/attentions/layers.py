from collections import OrderedDict
from typing import Tuple, Union
from math import pi, log
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
import torch.nn.init as init


# class QuickGELU(nn.Module):
#     def __init__(self, dim=1, alpha=1.0, epsilon=1e-7):
#         super().__init__()
#         self.dim = dim
#         self.alpha = alpha
#         self.epsilon = epsilon
    
#     def forward(self, x):
#         squared_norm = x.pow(2).sum(dim=self.dim, keepdim=True)
#         norm = torch.sqrt(squared_norm + self.epsilon)
        
#         # squash_factor = squared_norm / (1. + squared_norm)
#         # unit_vector = x / norm
        
#         return self.alpha * torch.sigmoid(1.702 * norm) * x 


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        x = rearrange(x, 'b l c -> b c l')
        x = self.norm(x)
        x = rearrange(x, 'b c l -> b l c')
        
        return x
    
class LayerNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        
        return x
      

class Squash(nn.Module):
    def __init__(self, dim, epsilon=1e-7):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon
    
    def forward(self, x, alpha=1.0):
        # squared_norm = x.pow(2).sum(dim=self.dim, keepdim=True)
        # norm = torch.sqrt(squared_norm + self.epsilon)
        
        # squash_factor = squared_norm / (1. + squared_norm)
        
        # # print(squash_factor.shape)
        # # print(alpha.shape)
        # unit_vector = x / norm
        
        return alpha * x #squash_factor * unit_vector  
      
      
class LMHCA(nn.Module):
    def __init__(self, dim, num_queries=16, num_heads=4, head_dim=64, attn_window_size=3, stride=1, qkv_bias=False, dropout=0.):
        super().__init__()
        padding=attn_window_size//2
        self.padding = padding
        self.attn_window_size = attn_window_size
        self.stride = stride
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        proj_dim = head_dim * num_heads
        self.proj_dim = proj_dim
        self.kv = nn.Conv2d(dim, proj_dim*2, kernel_size=1, bias=qkv_bias)
        self.head_proj = nn.Sequential(
            nn.Conv2d(proj_dim, head_dim, kernel_size=1),
            nn.Dropout(dropout)
        )
      
        # Calculate dot product between keys and learnbale queries, Here the parameters of the 1x1 kernel are  queries
        self.dot_product = nn.Conv2d(proj_dim, num_heads*num_queries, kernel_size=1, groups=num_heads, bias=False)
        
        # self.squash = Squash(dim=1)
        
        self.query_proj = nn.Sequential(
            nn.Conv2d(head_dim*num_queries, dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        
    # # def reset_parameters(self):
    # #     init.normal_(self.q, -1, 1)
      
    def unfold(self, x):
        x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        x = x.unfold(2, self.attn_window_size, self.stride).unfold(3, self.attn_window_size, self.stride)
        
        return x
      
    def forward(self, x):
        b, c, h, w = x.shape
        
        k, v = self.kv(x).chunk(2, dim=1)
        
        attn = self.dot_product(k)
        attn = self.unfold(attn)
        attn = rearrange(attn, 'b (n k) h w ... -> b n h w k (...)', n=self.num_heads)
        
        # squash_factor = torch.sigmoid(attn)
        # squash_factor = squash_factor.mean(dim=[-1, 1], keepdim=True)
        # squash_factor = rearrange(squash_factor, 'b f h w k e -> (b k) (e f) h w')
        # squash_factor = torch.where(squash_factor > torch.Tensor([0.5]).cuda(), torch.Tensor([1.]).cuda(), torch.Tensor([0.]).cuda())
        
        
        v = self.unfold(v)
        v = rearrange(v, 'b (n d) h w ... -> b n h w (...) d', n=self.num_heads)

        attn = attn.softmax(dim=-1) 
        out  = einsum('b n h w k l, b n h w l d -> b n h w k d', attn, v)
        
        out = rearrange(out, 'b n h w k d -> (b k) (n d) h w')
        out = self.head_proj(out)
        
        # out = squash_factor * out #self.squash(out, alpha=squash_factor)
        out = rearrange(out, '(b k) d h w -> b (k d) h w', k=self.num_queries)
        out = self.query_proj(out)
        
        return out
    

class LMHSA(nn.Module):
    def __init__(self, dim, num_heads=4, head_dim=64, attn_window_size=3, stride=1, qkv_bias=False, dropout=0.):
        super().__init__()
        padding=attn_window_size//2
        self.padding = padding
        self.attn_window_size = attn_window_size
        self.stride = stride
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        proj_dim = head_dim * num_heads
        self.proj_dim = proj_dim
        self.qkv = nn.Conv2d(dim, proj_dim*3, kernel_size=1, bias=qkv_bias)
        self.head_proj = nn.Sequential(
            nn.Conv2d(proj_dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )
      
    def unfold(self, x):
        x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        x = x.unfold(2, self.attn_window_size, self.stride).unfold(3, self.attn_window_size, self.stride)
        
        return x
      
    def forward(self, x):
        b, c, h, w = x.shape
        
        k, v = self.kv(x).chunk(2, dim=1)
        
        attn = self.dot_product(k)
        attn = self.unfold(attn)
        attn = rearrange(attn, 'b (n k) h w ... -> b n h w k (...)', n=self.num_heads)
        
        # squash_factor = torch.sigmoid(attn)
        # squash_factor = squash_factor.mean(dim=[-1, 1], keepdim=True)
        # squash_factor = rearrange(squash_factor, 'b f h w k e -> (b k) (e f) h w')
        # squash_factor = torch.where(squash_factor > torch.Tensor([0.5]).cuda(), torch.Tensor([1.]).cuda(), torch.Tensor([0.]).cuda())
        
        
        v = self.unfold(v)
        v = rearrange(v, 'b (n d) h w ... -> b n h w (...) d', n=self.num_heads)

        attn = attn.softmax(dim=-1) 
        out  = einsum('b n h w k l, b n h w l d -> b n h w k d', attn, v)
        
        out = rearrange(out, 'b n h w k d -> (b k) (n d) h w')
        out = self.head_proj(out)
        
        # out = squash_factor * out #self.squash(out, alpha=squash_factor)
        out = rearrange(out, '(b k) d h w -> b (k d) h w', k=self.num_queries)
        out = self.query_proj(out)
        
        return out    

      
class GMHCA(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads=4, head_dim=64, qkv_bias=False, dropout=0., attention_type='dot'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = (head_dim ** -0.5)
        proj_dim = head_dim * num_heads
        self.q = nn.Linear(query_dim, proj_dim, bias=qkv_bias)
        self.kv = nn.Linear(context_dim, proj_dim*2, bias=qkv_bias)
        self.mh_proj = nn.Sequential(
            nn.Linear(proj_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.attention_type = attention_type
        
    def forward(self, query, context):
        q = self.q(query)
        k, v = self.kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b l (n d) -> (b n) l d', n=self.num_heads), (q, k, v))
        
        if self.attention_type == 'dot':
            attn = einsum('b i d, b j d -> b i j', q, k) * self.scale
        elif self.attention_type == 'dist':
            attn = torch.cdist(q, k, p=2.0) * self.scale * (-1.0)
        else:
            raise ValueError(f" Attention type not found!")

        attn = attn.softmax(dim=-1)
        
        out  = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b n) l d -> b l (n d)', n=self.num_heads)
        out = self.mh_proj(out)
        
        return out, attn