import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def _spatial_pool(x, pool_type='avg'):
    batch, channel, height, width = x.size()
    if pool_type == 'avg':
        return F.avg_pool2d(x, (height, width))
    elif pool_type == 'max':
        return F.max_pool2d(x, (height, width))
    else:
        raise ValueError(f"Unknown pooling type: {pool_type}")

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        # Add an extra MLP layer for more expressiveness
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Enhanced pooling: combine max and avg pool with learnable weights
        avg_pool = F.avg_pool2d(x, kernel_size=(height, width)).view(batch_size, channels)
        max_pool = F.max_pool2d(x, kernel_size=(height, width)).view(batch_size, channels)
        
        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)
        
        attention = self.sigmoid(avg_out + max_out)
        return attention.view(batch_size, channels, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Enhanced channel pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pool_out = torch.cat([avg_out, max_out], dim=1)
        
        attention = self.conv(pool_out)
        return self.sigmoid(attention)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel)
        
        # Add global context
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Apply spatial attention
        sa = self.spatial_attention(x)
        x = x * sa
        
        # Add global context
        gc = self.global_context(x)
        x = x * gc
        
        return x