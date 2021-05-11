import torch
from torch import nn
from einops import rearrange, repeat
from .layers import GMHCA, BatchNorm1d, QuickGELU, LayerNorm2d, LMHCA
import torch.nn.functional as F
import math

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
            
        self.act = QuickGELU()

    def forward(self, x):
        out =  self.act(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.act(self.bn2(out)))
        out += shortcut
        return out


class ResidualLocalAttentionBlock(nn.Module):
    def __init__(self, dim, num_queries=16, num_heads=4, head_dim=64, multiplier=4, attn_window_size=3, stride=1, padding=1, qkv_bias=False, dropout=0.):
        super().__init__()
        self.attend = LMHCA(dim, num_queries, num_heads, head_dim, attn_window_size, 1, qkv_bias, dropout)
        self.bn1 = nn.BatchNorm2d(dim)
        proj_dim = dim*multiplier
        # self.proj = nn.Sequential(
        #     nn.Conv2d(dim, proj_dim, kernel_size=1),
        #     nn.BatchNorm2d(proj_dim),
        #     QuickGELU(),
        #     nn.Conv2d(proj_dim, dim, kernel_size=1)
        # )
        # self.proj = nn.Sequential(
        #     nn.Conv2d(dim, proj_dim, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(proj_dim),
        #     QuickGELU(),
        #     nn.Conv2d(proj_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        # )
        # self.bn2 = nn.BatchNorm2d(dim)
        # if stride > 1:
        #     self.shortcut = nn.Sequential(
        #         nn.BatchNorm2d(dim),
        #         QuickGELU(),
        #         nn.Conv2d(dim, dim, kernel_size=1, stride=stride, bias=False)
        #     )
        # else:
        #     self.shortcut = nn.Sequential(
        #         nn.BatchNorm2d(dim),
        #         QuickGELU(),
        #     )
        
        self.proj = PreActBlock(in_planes=dim, planes=dim, stride=stride)
          

    def forward(self, x: torch.Tensor):
        # x = self.shortcut(x) + self.attend(self.bn1(x))
        # x = x + self.proj(self.bn2(x))
        # return x
        x = self.proj(x)
        x = x + self.attend(self.bn1(x))
        return x
      
class CrossAttentionBlock(nn.Module):
    def __init__(self, num_queries, query_dim, context_dim, num_heads=4, multiplier=4, head_dim=64, qkv_bias=False, dropout=0., attention_type='dot'):
        super().__init__()
        self.attend = GMHCA(query_dim, context_dim, num_heads, head_dim, qkv_bias, dropout, attention_type)
        self.norm_query = BatchNorm1d(query_dim)
        self.norm_context = BatchNorm1d(context_dim)
        proj_dim = query_dim*multiplier
        self.proj = nn.Sequential(
            nn.Linear(query_dim, proj_dim),
            BatchNorm1d(proj_dim),
            # nn.LayerNorm(proj_dim),
            QuickGELU(),
            nn.Linear(proj_dim, query_dim)
        )
        self.proj_norm = BatchNorm1d(query_dim)
        self.query = nn.Parameter(torch.randn(num_queries, query_dim))
    
    def forward(self, context: torch.Tensor):
        b = context.shape[0]
        
        query = repeat(self.query, 'l d -> b l d', b=b)

        x, attn = self.attend(self.norm_query(query), self.norm_context(context))
        x = self.proj(self.proj_norm(x))
        return x, attn

class Attender(nn.Module):
    def __init__(self, num_classes: int, width: int=64, num_layers=3):
        super().__init__()
        self.conv1 = nn.Sequential(
          # nn.Conv2d(in_channels=3, out_channels=width, kernel_size=7, padding=3, stride=4),
          nn.Conv2d(in_channels=3, out_channels=width, kernel_size=3, stride=2,  padding=1),
          # nn.BatchNorm2d(width),
          # QuickGELU()
        )
        
        self.local_attn_layers = nn.ModuleList([])  
        s = 1
        for i in range(num_layers):
            if i == 1:
              s = 2
            else:
              s = 1
            self.local_attn_layers.append(nn.ModuleList([
                ResidualLocalAttentionBlock(dim=width, num_queries=16, num_heads=4, head_dim=32, multiplier=4, attn_window_size=3, stride=s, padding=1, qkv_bias=False, dropout=0.),
                ResidualLocalAttentionBlock(dim=width, num_queries=16, num_heads=4, head_dim=32, multiplier=4, attn_window_size=3, stride=1, padding=1, qkv_bias=False, dropout=0.),
                ResidualLocalAttentionBlock(dim=width, num_queries=16, num_heads=4, head_dim=32, multiplier=4, attn_window_size=3, stride=1, padding=1, qkv_bias=False, dropout=0.)
            ]))
            
        self.final_layer = nn.Sequential(
            ResidualLocalAttentionBlock(dim=width, num_queries=16, num_heads=4, head_dim=32, multiplier=4, attn_window_size=3, stride=1, padding=1, qkv_bias=False, dropout=0.),
            ResidualLocalAttentionBlock(dim=width, num_queries=16, num_heads=4, head_dim=32, multiplier=4, attn_window_size=3, stride=1, padding=1, qkv_bias=False, dropout=0.)
        )
        self.final_layer2 = CrossAttentionBlock(num_queries=16, query_dim=width, context_dim=width, num_heads=4, multiplier=4, head_dim=32, qkv_bias=False, dropout=0., attention_type='dot')
        
        # self.pool = nn.MaxPool2d(2, stride=2)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(width),
            QuickGELU(),
            nn.Linear(width, num_classes)
        )

    def forward(self, x: torch.Tensor):

        x  = self.conv1(x)
        
        # counter = 0
        for layer1, layer2, layer3 in self.local_attn_layers:
          x = layer3(layer2(layer1((x))))
          # if counter < 2:
          #   x = self.pool(x)
          # counter += 1
        
        x = self.final_layer(x)
        # print(x.shape)
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        x, _ = self.final_layer2(x)
        
        x = torch.mean(x, dim=1)
        
        x = self.classifier(x.squeeze(1))

        return x
    