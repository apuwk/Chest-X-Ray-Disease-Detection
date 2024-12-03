import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        """
        Channel attention module that helps the model focus on relevant feature channels.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for channel compression (default: 16)
        """
        super().__init__()
        
        # Calculate reduced channels
        reduced_channels = max(in_channels // reduction_ratio, 1)
        
        # Shared MLP for both max and avg pooled features
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Average pooling
        avg_pool = F.avg_pool2d(x, kernel_size=(height, width)).view(batch_size, channels)
        avg_out = self.shared_mlp(avg_pool)
        
        # Max pooling
        max_pool = F.max_pool2d(x, kernel_size=(height, width)).view(batch_size, channels)
        max_out = self.shared_mlp(max_pool)
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)
        
        # Reshape attention to match input dimensions
        attention = attention.view(batch_size, channels, 1, 1)
        
        return x * attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """
        Spatial attention module that helps the model focus on relevant image regions.
        
        Args:
            kernel_size: Size of the convolutional kernel (default: 7)
        """
        super().__init__()
        
        if not kernel_size % 2:
            raise ValueError("Kernel size must be odd")
            
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # Max pooling across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate pooled features
        pooled_features = torch.cat([avg_out, max_out], dim=1)
        
        # Generate spatial attention map
        attention_map = self.conv(pooled_features)
        attention_map = self.sigmoid(attention_map)
        
        return x * attention_map

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel=7):
        """
        Convolutional Block Attention Module (CBAM) combining both attention mechanisms.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for channel attention
            spatial_kernel: Kernel size for spatial attention
        """
        super().__init__()
        
        self.channel_attention = ChannelAttention(
            in_channels=in_channels,
            reduction_ratio=reduction_ratio
        )
        
        self.spatial_attention = SpatialAttention(
            kernel_size=spatial_kernel
        )

    def forward(self, x):
        # Apply channel attention first
        x = self.channel_attention(x)
        
        # Then apply spatial attention
        x = self.spatial_attention(x)
        
        return x

class CBAMResNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        """
        ResNet model with CBAM attention for X-ray classification.
        
        Args:
            num_classes: Number of output classes (default: 1 for binary)
            pretrained: Whether to use pretrained weights (default: True)
        """
        super().__init__()
        
        # Load base ResNet model
        self.resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Get number of features from final layer
        num_features = self.resnet.fc.in_features
        
        # Remove final fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Add CBAM after features
        self.cbam = CBAM(in_channels=2048)  # ResNet50's final channel count
        
        # Global average pooling and classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features
        x = self.features(x)
        
        # Apply attention
        x = self.cbam(x)
        
        # Global average pooling
        x = self.gap(x)
        
        # Classification
        x = self.classifier(x)
        
        return x

    def get_attention_maps(self, x):
        """
        Get intermediate attention maps for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary containing channel and spatial attention maps
        """
        # Get feature maps
        features = self.features(x)
        
        # Get channel attention
        channel_attention = self.cbam.channel_attention(features)
        
        # Get spatial attention
        spatial_attention = self.cbam.spatial_attention(channel_attention)
        
        return {
            'channel_attention': channel_attention,
            'spatial_attention': spatial_attention
        }