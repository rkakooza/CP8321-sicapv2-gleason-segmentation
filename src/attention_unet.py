import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        conv_out = self.double_conv(x)
        pooled = self.pool(conv_out)
        return conv_out, pooled


class AttentionGate(nn.Module):
    def __init__(self, in_channels_x, in_channels_g, inter_channels):
        super().__init__()

        self.Wx = nn.Sequential(
            nn.Conv2d(in_channels_x, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels)
        )

        self.Wg = nn.Sequential(
            nn.Conv2d(in_channels_g, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels)
        )

        # psi(ψ): produces attention coefficients 
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        gx = self.Wg(g)
        xx = self.Wx(x)

        if gx.size() != xx.size():
            gx = F.interpolate(gx, size=xx.shape[2:], mode="bilinear", align_corners=True)

        psi = self.relu(gx + xx)
        psi = self.psi(psi)

        return x * psi
    

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_channels_skip):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        inter_channels = in_channels_skip
        self.attention = AttentionGate(
            in_channels_x=in_channels_skip,
            in_channels_g=out_channels,
            inter_channels=inter_channels
        )

        self.double_conv = DoubleConv(out_channels + in_channels_skip, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = self.conv1x1(x)

        skip = self.attention(skip, x)

        x = torch.cat([x, skip], dim=1)

        return self.double_conv(x)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super().__init__()

        # Encoder
        self.enc1 = DownBlock(in_channels, 64)
        self.enc2 = DownBlock(64, 128)
        self.enc3 = DownBlock(128, 256)
        self.enc4 = DownBlock(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = UpBlock(1024, 512, in_channels_skip=512)
        self.up3 = UpBlock(512, 256, in_channels_skip=256)
        self.up2 = UpBlock(256, 128, in_channels_skip=128)
        self.up1 = UpBlock(128, 64, in_channels_skip=64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, p1 = self.enc1(x)
        x2, p2 = self.enc2(p1)
        x3, p3 = self.enc3(p2)
        x4, p4 = self.enc4(p3)

        # Bottleneck
        bottleneck = self.bottleneck(p4)

        # Decoder
        d4 = self.up4(bottleneck, x4)
        d3 = self.up3(d4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)

        # Output
        return self.final_conv(d1)