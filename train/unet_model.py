import torch.nn as nn
import torchvision.models as models
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class Resnet34(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        old = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.resnet.conv1.weight[:, :3] = old.weight
            self.resnet.conv1.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
        self.initial = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )

        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

    def forward(self, x):
        x0 = self.initial(x)
        x = self.resnet.maxpool(x0)
        x1 = self.layer1(x) # 64
        x2 = self.layer2(x1) # 128
        x3 = self.layer3(x2) # 256
        x4 = self.layer4(x3) # 512

        return x0, x1, x2, x3, x4

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.encoder = Resnet34(in_channels)

        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = DoubleConv(512 + 256, 256)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = DoubleConv(256 + 128, 128)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DoubleConv(128 + 64, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = DoubleConv(64 + 64, 64)
        self.up0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec0 = DoubleConv(64, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        ec0, ec1, ec2, ec3, ec4 = self.encoder(x)

        up4 = self.up4(ec4)
        dec4 = self.dec4(torch.cat([up4, ec3], dim=1))
        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([up3, ec2], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, ec1], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, ec0], dim=1))
        up0 = self.up0(dec1)
        dec0 = self.dec0(up0)

        out = self.out(dec0)

        return out