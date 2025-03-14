import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAdaptiveEdgeEnhance3D(nn.Module):
    """
    3D Spatially-Adaptive Edge Enhancement Module

    参数：
        in_channels (int): 输入通道数
        reduction_ratio (int): 通道缩减比例，默认为16
        kernel_size (int): 边缘检测核大小，默认为3
    """

    def __init__(self, in_channels, reduction_ratio=16, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        # 边缘特征提取分支
        self.edge_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, 1, kernel_size=1)  # 输出单通道边缘图
        )

        # 空间自适应权重生成
        self.weight_net = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=7, padding=3),  # 保持空间维度
            nn.ReLU(inplace=True),
            nn.Conv3d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()  # 输出0-1的增强权重
        )

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv_final = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        输入：
            x (Tensor): 形状为 (B, C, D, H, W)
        输出：
            enhanced (Tensor): 形状与输入相同
        """
        identity = x

        # 边缘特征提取
        edge_map = self.edge_conv(x)  # (B, 1, D, H, W)

        # 空间自适应权重
        spatial_weights = self.weight_net(edge_map)  # (B, 1, D, H, W)

        # 通道注意力
        channel_weights = self.channel_attention(x)  # (B, C, 1, 1, 1)

        # 增强处理
        enhanced_edge = edge_map * spatial_weights  # 空间增强
        enhanced = x * enhanced_edge  # 边缘特征融合

        # 通道调整
        enhanced = enhanced * channel_weights

        # 残差连接
        out = self.conv_final(enhanced) + identity

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    # 测试参数
    B, C, D, H, W = 2, 64, 32, 32, 32
    x = torch.randn(B, C, D, H, W)
    # 初始化模块
    enhancer = SpatialAdaptiveEdgeEnhance3D(in_channels=C)
    # 前向传播
    enhanced = enhancer(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {enhanced.shape}")
    # 应该输出：
    # 输入形状: torch.Size([2, 64, 32, 32, 32])
    # 输出形状: torch.Size([2, 64, 32, 32, 32])