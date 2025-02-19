import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # If dimensions change, adjust the identity connection.
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out += identity
        out = self.relu(out)
        return out

class ResNet1DForParams(nn.Module):
    def __init__(self, num_params=2):
        super(ResNet1DForParams, self).__init__()
        # Initial convolution: preserves length (32 x 512 output) with padding=1.
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Stack a few residual blocks with downsampling to capture higher-level features.
        self.layer1 = ResidualBlock1D(32, 64, stride=2)  # Output length halved
        self.layer2 = ResidualBlock1D(64, 128, stride=2) # Halved again
        self.layer3 = ResidualBlock1D(128, 256, stride=2) # And halved once more
        
        # Global average pooling to collapse the temporal dimension.
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layer to predict the two parameters (lambda and alpha).
        self.fc = nn.Linear(256, num_params)
    
    def forward(self, x):
        # x: [batch_size, 1, 512]
        out = self.conv1(x)    # -> [batch_size, 32, 512]
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out) # -> [batch_size, 64, 256]
        out = self.layer2(out) # -> [batch_size, 128, 128]
        out = self.layer3(out) # -> [batch_size, 256, 64]
        
        out = self.avg_pool(out)  # -> [batch_size, 256, 1]
        out = out.squeeze(-1)     # -> [batch_size, 256]
        out = self.fc(out)        # -> [batch_size, 2] (lambda and alpha)
        return out

if __name__ == '__main__':
    model = ResNet1DForParams()
    # Create a dummy input: batch_size=2, 1 channel, length=512.
    x = torch.randn(2, 1, 512)
    params = model(x)
    print("Predicted parameters shape:", params.shape)  # Should print: torch.Size([2, 2])
