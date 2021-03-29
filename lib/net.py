"""
other methods
"""
from torch import nn
from torch.nn import functional as F
import torch


class SRGAN(nn.Module):
    """
    original SRGAN
    """
    def __init__(self,block_num = 5):
        super(SRGAN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,9,padding=4),
            nn.PReLU()
        )
        blocks = [ResidualBlock(64) for _ in range(block_num)]
        self.blocks = nn.Sequential(*blocks)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64)
        )
        self.up1 = UpsampleBLock(64, 2)
        self.conv3 = nn.Conv2d(64, 3, 9, padding=4)

    def forward(self,x):
        conv1 = self.conv1(x)
        blocks = self.blocks(conv1)
        conv2 = self.conv2(blocks)
        up1 = self.up1(conv1+conv2)
        conv3 = self.conv3(up1)
        return (torch.tanh(conv3)+1)/2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    """
    使用BN层的残差块
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # need to be identified in every layer
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        return x + out


class UpsampleBLock(nn.Module):
    """
    上采样模块，pixelshuffle
    """

    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class NMUnet(nn.Module):
    """
    implement of NM Deep-Z
    """
    def __init__(self, sr=False):
        super(NMUnet, self).__init__()
        self.sr = sr
        self.conv11 = self.get_conv(4, 25, 3, 1, 1)
        self.conv12 = self.get_conv(25, 48, 3, 1, 1)
        self.conv21 = self.get_conv(48, 72, 3, 1, 1)
        self.conv22 = self.get_conv(72, 96, 3, 1, 1)
        self.conv31 = self.get_conv(96, 144, 3, 1, 1)
        self.conv32 = self.get_conv(144, 192, 3, 1, 1)
        self.conv41 = self.get_conv(192, 288, 3, 1, 1)
        self.conv42 = self.get_conv(288, 384, 3, 1, 1)
        self.conv51 = self.get_conv(384, 576, 3, 1, 1)
        self.conv52 = self.get_conv(576, 768, 3, 1, 1)
        self.up1 = self.get_up_conv(768, 384)
        self.conv61 = self.get_conv(768, 576, 3, 1, 1)
        self.conv62 = self.get_conv(576, 384, 3, 1, 1)
        self.up2 = self.get_up_conv(384, 192)
        self.conv71 = self.get_conv(384, 288, 3, 1, 1)
        self.conv72 = self.get_conv(288, 192, 3, 1, 1)
        self.up3 = self.get_up_conv(192, 96)
        self.conv81 = self.get_conv(192, 144, 3, 1, 1)
        self.conv82 = self.get_conv(144, 96, 3, 1, 1)
        self.up4 = self.get_up_conv(96, 48)
        self.conv91 = self.get_conv(96, 72, 3, 1, 1)
        self.conv92 = self.get_conv(72, 48, 3, 1, 1)
        if sr:
            self.upsample = UpsampleBLock(48, 2)
        self.out = self.get_conv(48, 3, 3, 1, 1)

    def forward(self, x):
        conv11 = self.conv11(x)
        conv12 = self.conv12(conv11)
        residual1 = torch.cat([x for _ in range(12)], dim=1)+conv12
        down1 = F.max_pool2d(residual1, 2, 2)

        conv21 = self.conv21(down1)
        conv22 = self.conv22(conv21)
        residual2 = torch.cat([down1, down1], dim=1) + conv22
        down2 = F.max_pool2d(residual2, 2, 2)

        conv31 = self.conv31(down2)
        conv32 = self.conv32(conv31)
        residual3 = torch.cat([down2, down2], dim=1) + conv32
        down3 = F.max_pool2d(residual3, 2, 2)

        conv41 = self.conv41(down3)
        conv42 = self.conv42(conv41)
        residual4 = torch.cat([down3, down3], dim=1) + conv42
        down4 = F.max_pool2d(residual4, 2, 2)

        conv51 = self.conv51(down4)
        conv52 = self.conv52(conv51)
        residual5 = torch.cat([down4, down4], dim=1) + conv52

        up1 = self.up1(residual5)
        cat1 = torch.cat([residual4, up1], dim=1)
        conv61 = self.conv61(cat1)
        conv62 = self.conv62(conv61)
        residual6 = up1+conv62

        up2 = self.up2(residual6)
        cat2 = torch.cat([residual3, up2], dim=1)
        conv71 = self.conv71(cat2)
        conv72 = self.conv72(conv71)
        residual7 = up2 + conv72

        up3 = self.up3(residual7)
        cat3 = torch.cat([residual2, up3], dim=1)
        conv81 = self.conv81(cat3)
        conv82 = self.conv82(conv81)
        residual8 = up3 + conv82

        up4 = self.up4(residual8)
        cat4 = torch.cat([residual1, up4], dim=1)
        conv91 = self.conv91(cat4)
        conv92 = self.conv92(conv91)
        residual9 = up4 + conv92
        if self.sr:
            residual9 = self.upsample(residual9)
        out = self.out(residual9)
        return out

    @staticmethod
    def get_conv(in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p),
            nn.ReLU()
        )

    @staticmethod
    def get_up_conv(in_c, out_c):
        return nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_c, out_c, 3, 1, 1)
        )


class NMD(nn.Module):
    def __init__(self):
        super(NMD, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 48, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(96, 192, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(192, 384, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(384, 768, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(768, 768, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(768, 1536, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(1536, 1536, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(1536, 3072, 3, 2, 1),

        )

        self.linear = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.Linear(3072, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        y = F.adaptive_avg_pool2d(x, (1, 1))
        y = torch.squeeze(y, dim=2).squeeze(dim=2)
        x = self.linear(y)
        return x


if __name__ == '__main__':
    net = NMUnet(sr=True)
    data = torch.randn((12, 4, 64, 64))
    yy = net(data)
    print(yy.shape)