import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x_up = self.upsample(x)
        out = self.conv(x_up)
        skip = self.skip(x_up)
        return F.relu(out + skip)


class EdgeGuidance(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_ch, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        edge = self.edge_conv(x)
        return edge


class MSAFusion(nn.Module):
    def __init__(self, ch1, ch2, out_ch):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(ch1 + ch2, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.Sigmoid()
        )
        self.fuse = nn.Conv2d(ch1 + ch2, out_ch, 1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        a = self.att(x)
        x = self.fuse(x)
        return x * a


class MiaDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 主人你给的是 [64, 128, 320, 512] 喵～
        self.edge = EdgeGuidance(64)

        self.up4 = ResidualUpBlock(512, 320)      # x4 -> x3 size
        self.fuse3 = MSAFusion(320, 320, 320)

        self.up3 = ResidualUpBlock(320, 128)      # -> x2 size
        self.fuse2 = MSAFusion(128, 128, 128)

        self.up2 = ResidualUpBlock(128, 64)       # -> x1 size
        self.fuse1 = MSAFusion(64, 64, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x1, x2, x3, x4):
        edge = self.edge(x1)

        x = self.up4(x4)          # 11x11 -> 22x22
        x = self.fuse3(x, x3)     # 和x3融合

        x = self.up3(x)           # 22x22 -> 44x44
        x = self.fuse2(x, x2)

        x = self.up2(x)           # 44x44 -> 88x88
        x = self.fuse1(x, x1)

        out = self.out(x)         # 最终显著图预测

        return out, edge
