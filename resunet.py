import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    """
    A basic 1D residual block:
    Conv1d -> BatchNorm -> ReLU -> Conv1d -> BatchNorm
    with a learnable shortcut if channels/stride differ.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut for matching dimensions when stride != 1 or channel dims differ
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                      stride=stride)
            self.shortcut_bn = nn.BatchNorm1d(out_channels)
        else:
            self.shortcut = nn.Identity()
            self.shortcut_bn = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        sc = self.shortcut(identity)
        sc = self.shortcut_bn(sc)

        out += sc
        out = self.relu(out)
        return out

class ResUNet1D(nn.Module):
    """
    A minimal ResUNet-like architecture for 1D signals.
    Roughly divided into:
      - Encoder (repeated residual blocks + downsampling via stride>1)
      - Bottleneck (a deeper residual block)
      - Decoder (transposed convolutions + residual blocks)
      - Final Flatten or Linear layer (depending on your output requirement)
    """
    def __init__(self):
        super().__init__()

        # -----------------------------
        # Encoder ("Block x4" concept)
        # -----------------------------
        # Each block may reduce the temporal dimension by stride=2
        self.enc1 = ResidualBlock1D(1,   32, stride=2)   # e.g. 1 -> 32 channels
        self.enc2 = ResidualBlock1D(32,  48, stride=2)   # 32 -> 48 channels
        self.enc3 = ResidualBlock1D(48,  48, stride=1)   # repeated block
        self.enc4 = ResidualBlock1D(48,  48, stride=1)   # repeated block

        # --------------------------------
        # Bottleneck ("Block x1" concept)
        # --------------------------------
        # Often with bigger jump in channels
        self.bottleneck = ResidualBlock1D(48, 243, stride=2)  # 48 -> 243 channels

        # -----------------------------
        # Decoder ("Block x5" concept)
        # -----------------------------
        # Transposed convs for upsampling, then residual blocks
        self.up1 = nn.ConvTranspose1d(243, 48, kernel_size=2, stride=2)
        self.res_up1 = ResidualBlock1D(48, 48)

        self.up2 = nn.ConvTranspose1d(48, 48, kernel_size=2, stride=2)
        self.res_up2 = ResidualBlock1D(48, 48)

        self.up3 = nn.ConvTranspose1d(48, 48, kernel_size=2, stride=2)
        self.res_up3 = ResidualBlock1D(48, 48)

        self.up4 = nn.ConvTranspose1d(48, 32, kernel_size=2, stride=2)
        self.res_up4 = ResidualBlock1D(32, 32)

        self.up5 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)
        self.res_up5 = ResidualBlock1D(16, 16)

        # -----------------------
        # Final projection/flatten
        # -----------------------
        # For a regression or classification head, you might replace this
        # with a linear layer. Here, we simply reduce to 1 channel, then flatten.
        self.final_conv = nn.Conv1d(16, 1, kernel_size=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # x shape: [batch_size, 1, signal_length]

        # ----- Encoder -----
        e1 = self.enc1(x)   # shape: [batch_size, 32, L/2]
        e2 = self.enc2(e1)  # shape: [batch_size, 48, L/4]
        e3 = self.enc3(e2)  # shape: [batch_size, 48, L/4]
        e4 = self.enc4(e3)  # shape: [batch_size, 48, L/4]

        # ----- Bottleneck -----
        b = self.bottleneck(e4)  # shape: [batch_size, 243, L/8]

        # ----- Decoder -----
        d1 = self.up1(b)         # [batch_size, 48, L/4]
        d1 = self.res_up1(d1)    # residual block
        d2 = self.up2(d1)        # [batch_size, 48, L/2]
        d2 = self.res_up2(d2)
        d3 = self.up3(d2)        # [batch_size, 48, L]
        d3 = self.res_up3(d3)
        d4 = self.up4(d3)        # [batch_size, 32, 2L]
        d4 = self.res_up4(d4)
        d5 = self.up5(d4)        # [batch_size, 16, 4L]
        d5 = self.res_up5(d5)

        out = self.final_conv(d5)  # [batch_size, 1, 4L]
        out = self.flatten(out)    # flatten to [batch_size, 1 * 4L]
        return out

if __name__ == "__main__":
    # Quick test
    model = ResUNet1D()
    # Example: batch_size=2, 1 channel, length=512
    x = torch.randn(2, 1, 512)
    y = model(x)
    print("Output shape:", y.shape)
