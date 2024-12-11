import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .attention import CBAM

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
        self.alpha_pos = torch.tensor([
            2.7808, 2.9208, 2.7749, 2.6414, 2.9183,
            2.8821, 2.9722, 2.9082, 2.9116, 2.9663,
            2.9540, 2.9465, 2.9285, 2.9943
        ])
        # Negative class weights (when condition is absent)
        self.alpha_neg = 1.0 - self.alpha_pos/3.0  # Example weighting scheme

    def forward(self, inputs, targets):
        self.alpha_pos = self.alpha_pos.to(inputs.device)
        self.alpha_neg = self.alpha_neg.to(inputs.device)
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        eps = 1e-7
        
        inputs_sigmoid = torch.sigmoid(inputs)
        pt = targets * inputs_sigmoid + (1 - targets) * (1 - inputs_sigmoid)
        pt = torch.clamp(pt, min=eps, max=1-eps)
        
        # Apply different weights for positive and negative cases
        alpha_t = targets * self.alpha_pos[None, :] + (1 - targets) * self.alpha_neg[None, :]
        
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = alpha_t * focal_weight * bce_loss
        
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
        self.resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Get intermediate features
        self.feature_channels = [256, 512, 1024, 2048]  # ResNet50 channel sizes
        
        # Add FPN
        self.fpn = SimpleFPN(self.feature_channels)
        
        # Multiple CBAM modules for different scales
        self.cbam_modules = nn.ModuleList([
            CBAM(256) for _ in range(4)  # One for each FPN level
        ])
        
        # Global pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Get intermediate features
        features = []
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        features.append(x)
        x = self.resnet.layer2(x)
        features.append(x)
        x = self.resnet.layer3(x)
        features.append(x)
        x = self.resnet.layer4(x)
        features.append(x)

        # Apply FPN
        fpn_features = self.fpn(features)
        
        # Apply CBAM at each scale
        attended_features = []
        for feat, cbam in zip(fpn_features, self.cbam_modules):
            attended = cbam(feat)
            pool = self.gap(attended)
            attended_features.append(pool)
        
        # Concatenate all scales
        multi_scale = torch.cat(attended_features, dim=1)
        
        # Classification
        output = self.classifier(multi_scale)
        return output

    def get_attention_maps(self, x):
        # Modified to return attention maps from all scales
        features = []
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        features.append(x)
        x = self.resnet.layer2(x)
        features.append(x)
        x = self.resnet.layer3(x)
        features.append(x)
        x = self.resnet.layer4(x)
        features.append(x)

        fpn_features = self.fpn(features)
        attention_maps = {}
        
        for i, (feat, cbam) in enumerate(zip(fpn_features, self.cbam_modules)):
            attention_maps[f'scale_{i}'] = {
                'feature': feat,
                'attention': cbam(feat)
            }
        
        return attention_maps