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
            nn.Conv2d(in_channels, out_channels, 1)
            for in_channels in in_channels_list
        ])

    def forward(self, features):
        results = []
        for i, (feature, conv) in enumerate(zip(features, self.lateral_convs)):
            results.append(conv(feature))
        return results

class ChestXrayModel(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super().__init__()
        # Base ResNet for global features
        self.resnet_global = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        
        # ResNet for local (zoomed) features
        self.resnet_local = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Get feature channels for both paths
        self.feature_channels = [256, 512, 1024, 2048]
        
        # FPN for both paths
        self.fpn_global = SimpleFPN(self.feature_channels)
        self.fpn_local = SimpleFPN(self.feature_channels)
        
        # CBAM modules for different scales (global path)
        self.cbam_global = nn.ModuleList([
            CBAM(256) for _ in range(4)
        ])
        
        # CBAM modules for different scales (local path)
        self.cbam_local = nn.ModuleList([
            CBAM(256) for _ in range(4)
        ])
        
        # Global pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(256 * 8, 1024),  # 8 = 4 scales * 2 paths
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Final classifier
        self.classifier = nn.Linear(512, num_classes)

    def _get_resnet_features(self, x, resnet):
        features = []
        x = resnet.conv1(x)
        x = resnet.bn1(x)
        x = resnet.relu(x)
        x = resnet.maxpool(x)

        x = resnet.layer1(x)
        features.append(x)
        x = resnet.layer2(x)
        features.append(x)
        x = resnet.layer3(x)
        features.append(x)
        x = resnet.layer4(x)
        features.append(x)
        
        return features

    def forward(self, x_global, x_local):
        # Get features from both paths
        global_features = self._get_resnet_features(x_global, self.resnet_global)
        local_features = self._get_resnet_features(x_local, self.resnet_local)
        
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
        attention_maps = {'global': {}, 'local': {}}
        
        # Get global features and attention maps
        global_features = self._get_resnet_features(x_global, self.resnet_global)
        global_fpn = self.fpn_global(global_features)
        for i, (feat, cbam) in enumerate(zip(global_fpn, self.cbam_global)):
            attention_maps['global'][f'scale_{i}'] = {
                'feature': feat,
                'attention': cbam(feat)
            }
            
        # Get local features and attention maps
        local_features = self._get_resnet_features(x_local, self.resnet_local)
        local_fpn = self.fpn_local(local_features)
        for i, (feat, cbam) in enumerate(zip(local_fpn, self.cbam_local)):
            attention_maps['local'][f'scale_{i}'] = {
                'feature': feat,
                'attention': cbam(feat)
            }
        
        return attention_maps