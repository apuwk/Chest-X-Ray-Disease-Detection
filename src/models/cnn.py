import torch
import torch.nn as nn
import torchvision.models as models
from .attention import CBAM


class ChestXrayModel(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        num_features = self.resnet.fc.in_features 
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.cbam = CBAM(in_channels=2048)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, num_classes),
        )
        
        
    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.gap(x)
        x = self.classifier(x)
        
        return x
    
    
    def get_attention_maps(self, x):
        features = self.features(x)
        channel_attention = self.cbam.channel_attention(features)
        spatial_attention = self.cbam.spatial_attention(channel_attention)
        
        return {
            'channel_attention': channel_attention,
            'spatial_attention': spatial_attention
        }



    
if __name__ == "__main__":
    sample_input = torch.randn(2, 3, 224, 224)

    model = ChestXrayModel(num_classes=1, pretrained=True)
    model.eval()  # Set to evaluation mode

    # Forward pass
    with torch.no_grad():
        output = model(sample_input)

    # Print shapes and values
    print("Input shape:", sample_input.shape)
    print("Output shape:", output.shape)
    print("Output values:", output)