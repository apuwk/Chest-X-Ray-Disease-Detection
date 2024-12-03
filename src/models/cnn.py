import torch
import torch.nn as nn
import torchvision.models as models

class BasicAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),  # Reduce channels
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),  # Restore channels
            nn.Sigmoid()  # Output attention weights between 0-1
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights  # Element-wise multiplication

class ChestXrayModel(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        num_features = self.resnet.fc.in_features 
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.classifier = nn.Linear(num_features, num_classes)
        
        
    def forward(self, x):
        x = self.resnet(x)
        
        x = x.flatten(1)
        
        x = self.classifier(x)
        
        x = torch.sigmoid(x)
        
        return x
    
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