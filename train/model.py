import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBackbone(nn.Module):
    def __init__(self):
        super(SimpleBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 下採樣
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 下採樣
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out1 = self.relu1(self.conv1(x))  # 特徵圖 1 (較大尺寸)
        out2 = self.relu2(self.conv2(out1))  # 特徵圖 2 (中等尺寸)
        out3 = self.relu3(self.conv3(out2))  # 特徵圖 3 (較小尺寸)
        return out1, out2, out3


class FeatureFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeatureFusion, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_c, out_channels, kernel_size=1) for in_c in in_channels_list]
        )
        self.relu = nn.ReLU()

    def forward(self, features):
        fused_features = []
        target_size = features[-1].size()[-2:]  # 以最小的特徵圖尺寸為目標
        for i, feat in enumerate(features):
            if feat.size()[-2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode="bilinear", align_corners=False
                )
            fused_features.append(self.convs[i](feat))
        return self.relu(torch.cat(fused_features, dim=1))  # 在通道維度上拼接


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels * (scale_factor**2),
            kernel_size=3,
            stride=scale_factor,
            padding=1,
            output_padding=1,
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.pixel_shuffle(self.upsample(x))
        x = self.relu(self.conv(x))
        return x


class SRModel(nn.Module):
    def __init__(self):
        super(SRModel, self).__init__()
        self.backbone = SimpleBackbone()
        fusion_channels = 64  # 你可以調整這個值
        self.feature_fusion = FeatureFusion([64, 128, 256], fusion_channels)
        self.upsample1 = UpsampleBlock(
            3 * fusion_channels, 256, scale_factor=2
        )  # 256 -> 512
        self.upsample2 = UpsampleBlock(256, 128, scale_factor=2)  # 512 -> 1024
        self.final_conv = nn.Conv2d(128, 3, kernel_size=3, padding=1)

    def forward(self, x):
        feat1, feat2, feat3 = self.backbone(x)
        fused_features = self.feature_fusion([feat1, feat2, feat3])
        upsampled = self.upsample1(fused_features)
        upsampled = self.upsample2(upsampled)
        out = self.final_conv(upsampled)
        return out


# 創建模型實例
# model = SRModel()
# print(model)

# 測試輸入
# low_res_input = torch.randn(1, 3, 256, 256) # 假設輸入是 256x256 (調整以匹配你的實際輸入)
# high_res_output = model(low_res_input)
# print("Output size:", high_res_output.size()) # 預期輸出尺寸接近目標尺寸
