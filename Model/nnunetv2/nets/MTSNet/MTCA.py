import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from monai.losses import DiceLoss, FocalLoss
from monai.networks.blocks import Convolution, UpSample


# ------------------- 模型架构 -------------------
class SharedEncoder(nn.Module):
    """共享编码器"""

    def __init__(self, in_channels=1, base_dim=32):
        super().__init__()
        self.conv1 = Convolution(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=base_dim,
            kernel_size=3,
            act="leakyrelu",
            norm="instance"
        )
        self.down1 = Convolution(
            spatial_dims=3,
            in_channels=base_dim,
            out_channels=base_dim * 2,
            kernel_size=2,
            strides=2,
            act="leakyrelu",
            norm="instance"
        )
        self.down2 = Convolution(
            spatial_dims=3,
            in_channels=base_dim * 2,
            out_channels=base_dim * 4,
            kernel_size=2,
            strides=2,
            act="leakyrelu",
            norm="instance"
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        return x1, x2, x3


class Task1Decoder(nn.Module):
    """任务1解码器：主动脉成分分割"""

    def __init__(self, in_channels, out_channels=4):
        super().__init__()
        self.up1 = UpSample(spatial_dims=3, in_channels=in_channels, out_channels=in_channels // 2, scale_factor=2)
        self.conv1 = Convolution(3, in_channels // 2, in_channels // 2, kernel_size=3, act="leakyrelu")
        self.up2 = UpSample(3, in_channels // 2, in_channels // 4, scale_factor=2)
        self.conv2 = Convolution(3, in_channels // 4, in_channels // 4, kernel_size=3, act="leakyrelu")
        self.final_conv = nn.Conv3d(in_channels // 4, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        return self.final_conv(x)


class Task2Decoder(nn.Module):
    """任务2解码器：主动脉病变分割"""

    def __init__(self, in_channels, out_channels=4):
        super().__init__()
        self.up1 = UpSample(3, in_channels, in_channels // 2, scale_factor=2)
        self.conv1 = Convolution(3, in_channels // 2, in_channels // 2, kernel_size=3, act="leakyrelu")
        self.up2 = UpSample(3, in_channels // 2, in_channels // 4, scale_factor=2)
        self.conv2 = Convolution(3, in_channels // 4, in_channels // 4, kernel_size=3, act="leakyrelu")
        self.final_conv = nn.Conv3d(in_channels // 4, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        return self.final_conv(x)


class DualTaskModel(nn.Module):
    """双任务分割模型"""

    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder()
        self.task1_decoder = Task1Decoder(in_channels=128)  # 任务1：成分分割
        self.task2_decoder = Task2Decoder(in_channels=128)  # 任务2：病变分割
        self.cross_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4)

    def forward(self, x):
        # 编码器提取特征
        x1, x2, x3 = self.encoder(x)

        # 任务1解码
        task1_feat = self.task1_decoder(x3)

        # 任务2解码
        task2_feat = self.task2_decoder(x3)

        # 跨任务特征交互
        B, C, D, H, W = task1_feat.shape
        task1_flat = task1_feat.view(B, C, -1).permute(2, 0, 1)  # (seq_len, B, C)
        task2_flat = task2_feat.view(B, C, -1).permute(2, 0, 1)
        att_out, _ = self.cross_attention(task1_flat, task2_flat, task2_flat)
        att_out = att_out.permute(1, 2, 0).view(B, C, D, H, W)

        # 任务2特征增强
        task2_feat = task2_feat + att_out

        return task1_feat, task2_feat


# ------------------- 损失函数 -------------------
class Task1Loss(nn.Module):
    """任务1损失：成分分割"""

    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(include_background=False)
        self.focal = FocalLoss(gamma=2.0)

    def forward(self, pred, target):
        return 0.7 * self.dice(pred, target) + 0.3 * self.focal(pred, target)


class Task2Loss(nn.Module):
    """任务2损失：病变分割"""

    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(include_background=False)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return 0.5 * self.dice(pred, target) + 0.5 * self.cross_entropy(pred, target)


# ------------------- 训练流程 -------------------
class AortaDataset(Dataset):
    """自定义数据集"""

    def __init__(self, data_dir):
        # 实现数据加载逻辑
        pass

    def __getitem__(self, index):
        # 返回图像、任务1标签、任务2标签
        return ct_volume, task1_mask, task2_mask


def train():
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualTaskModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)

    # 数据加载
    train_dataset = AortaDataset("./data/train")
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # 损失函数
    task1_loss = Task1Loss()
    task2_loss = Task2Loss()

    # 训练循环
    for epoch in range(100):
        model.train()
        total_loss = 0.0
        for batch_idx, (ct, mask1, mask2) in enumerate(train_loader):
            ct = ct.to(device)
            mask1 = mask1.to(device)
            mask2 = mask2.to(device)

            # 前向传播
            pred1, pred2 = model(ct)

            # 计算损失
            loss1 = task1_loss(pred1, mask1)
            loss2 = task2_loss(pred2, mask2)
            loss = 0.6 * loss1 + 0.4 * loss2

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # 调整学习率
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        # 保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")


# ------------------- 主程序 -------------------
if __name__ == "__main__":
    train()