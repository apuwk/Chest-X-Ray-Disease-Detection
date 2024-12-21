import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .attention import CBAM

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class SimpleFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                # Added 3x3 conv before 1x1 for better feature adaptation
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                # 1x1 conv for channel reduction
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for in_channels in in_channels_list
        ])

    def forward(self, features):
        results = []
        for feature, conv in zip(features, self.lateral_convs):
            results.append(conv(feature))
        return results

class ChestXrayModel(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super().__init__()
        # Replace ResNet with DenseNet for both paths
        self.densenet_global = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
        self.densenet_local = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Freeze BatchNorm layers in DenseNet
        for model in [self.densenet_global, self.densenet_local]:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    module.weight.requires_grad_(False)
                    module.bias.requires_grad_(False)
        
        # DenseNet121 feature channels
        self.feature_channels = [256, 512, 1024, 1024]
        
        # FPN for both paths
        self.fpn_global = SimpleFPN(self.feature_channels)
        self.fpn_local = SimpleFPN(self.feature_channels)
        
        # CBAM modules for different scales
        self.cbam_global = nn.ModuleList([
            CBAM(256) for _ in range(4)
        ])
        
        self.cbam_local = nn.ModuleList([
            CBAM(256) for _ in range(4)
        ])
        
        # Global pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Feature fusion with batch norm
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(256 * 8),  # 8 = 4 scales * 2 paths
            nn.Linear(256 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512)
        )
        
        # Final classifier
        self.classifier = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights using Xavier initialization with larger gain"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1.4)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Apply to non-pretrained parts
        self.fpn_global.apply(init_weights)
        self.fpn_local.apply(init_weights)
        self.fusion.apply(init_weights)
        init_weights(self.classifier)

    def _get_densenet_features(self, x, densenet):
        """Extract features from DenseNet"""
        features = []
        
        # Initial convolution and pooling
        x = densenet.features.conv0(x)
        x = densenet.features.norm0(x)
        x = densenet.features.relu0(x)
        x = densenet.features.pool0(x)
        
        # Dense blocks
        x = densenet.features.denseblock1(x)
        features.append(x)
        x = densenet.features.transition1(x)
        
        x = densenet.features.denseblock2(x)
        features.append(x)
        x = densenet.features.transition2(x)
        
        x = densenet.features.denseblock3(x)
        features.append(x)
        x = densenet.features.transition3(x)
        
        x = densenet.features.denseblock4(x)
        features.append(x)
        
        return features

    def forward(self, x_global, x_local):
        # Get features from both paths using DenseNet
        global_features = self._get_densenet_features(x_global, self.densenet_global)
        local_features = self._get_densenet_features(x_local, self.densenet_local)
        
        # Apply FPN
        global_fpn = self.fpn_global(global_features)
        local_fpn = self.fpn_local(local_features)
        
        # Apply CBAM and pooling for global path
        global_attended = []
        for feat, cbam in zip(global_fpn, self.cbam_global):
            attended = cbam(feat)
            pooled = self.gap(attended)
            global_attended.append(pooled)
        
        # Apply CBAM and pooling for local path
        local_attended = []
        for feat, cbam in zip(local_fpn, self.cbam_local):
            attended = cbam(feat)
            pooled = self.gap(attended)
            local_attended.append(pooled)
        
        # Concatenate all scales from both paths
        multi_scale = torch.cat(global_attended + local_attended, dim=1)
        multi_scale = torch.flatten(multi_scale, 1)
        
        # Fusion and classification
        fused = self.fusion(multi_scale)
        output = self.classifier(fused)
        
        return output

    def get_attention_maps(self, x_global, x_local):
        """Generate attention maps for visualization"""
        attention_maps = {'global': {}, 'local': {}}
        
        # Get global features and attention maps
        global_features = self._get_densenet_features(x_global, self.densenet_global)
        global_fpn = self.fpn_global(global_features)
        for i, (feat, cbam) in enumerate(zip(global_fpn, self.cbam_global)):
            attention_maps['global'][f'scale_{i}'] = {
                'feature': feat,
                'attention': cbam(feat)
            }
            
        # Get local features and attention maps
        local_features = self._get_densenet_features(x_local, self.densenet_local)
        local_fpn = self.fpn_local(local_features)
        for i, (feat, cbam) in enumerate(zip(local_fpn, self.cbam_local)):
            attention_maps['local'][f'scale_{i}'] = {
                'feature': feat,
                'attention': cbam(feat)
            }
        
        return attention_maps