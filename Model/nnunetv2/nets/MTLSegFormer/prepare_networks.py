import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerModel

class EfficientSelfAttention(nn.Module):
    """高效自注意力模块，包含键的合并操作"""

    def __init__(self, embed_dim, num_heads, reduction_ratio=1):
        super().__init__()
        # self.num_heads = num_heads
        # self.reduction_ratio = reduction_ratio
        self.scale = (embed_dim // num_heads) ** -0.5
        # self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x=self.proj(x)
        return x

class MultiTaskTransformerBlock(nn.Module):
    """多任务Transformer块"""

    def __init__(self, embed_dim, num_heads, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks
        self.norm1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_tasks)])
        self.attn1 = EfficientSelfAttention(embed_dim, num_heads)
        self.attn2 = EfficientSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_tasks)])
        self.mlp1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, task_features):
        x = self.norm1[0](task_features[0])
        k=v = self.norm1[0](task_features[0])

        xx = self.norm1[1](task_features[1])
        kk=vv = self.norm1[1](task_features[1])

        attn_output1 = self.attn1(xx,k,v)
        x = x + attn_output1
        x = x + self.mlp1(self.norm2[0](x))

        attn_output2 = self.attn2(x,kk,vv)
        xx = xx + attn_output2
        xx = xx + self.mlp2(self.norm2[1](xx))
        return x,xx

class MTLSegFormer(nn.Module):
    def __init__(self, num_classes_per_task, num_tasks):
        super().__init__()
        # 编码器配置
        config = SegformerConfig(
            num_channels=1,
            depths=[2, 3, 4, 2],#depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            hidden_sizes=[32, 64, 160, 256],  # 编码器各阶段输出通道 [64, 128, 320, 512]
            num_attention_heads=[1, 2, 5, 8],#1 2 5 8
            mlp_ratios=[4, 4, 4, 4],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
        )

        # 初始化模型（无需手动修改patch_embed）
        self.encoder = SegformerModel(config)

        # 解码器参数
        decoder_channels = 64 #256
        self.num_tasks = num_tasks
        encoder_hidden_sizes = config.hidden_sizes[::-1]  # 反转后为 [512, 320, 128, 64]
        # 各阶段需要的上采样比例（7→56需要8倍，14→56需要4倍，28→56需要2倍，56保持）
        upscale_factors = [8, 4, 2, 1]

        # MLP层（通道转换+空间上采样）
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(enc_ch, decoder_channels, 1),  # 通道数统一为64
                nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
            ) for enc_ch, scale in zip(encoder_hidden_sizes, upscale_factors)
        ])

        # 多任务Transformer块
        self.multi_task_blocks = nn.ModuleList([
            MultiTaskTransformerBlock(decoder_channels * 4, num_heads=8, num_tasks=num_tasks)
            for _ in range(2)
        ])

        # 任务头
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(decoder_channels * 4, num_classes, 1),
                nn.Upsample(scale_factor=4, mode='bilinear')
            ) for num_classes in num_classes_per_task
        ])

    def forward(self, x):
        # 直接输入单通道图像（无需复制通道）
        encoder_outputs = self.encoder(x, output_hidden_states=True).hidden_states[::-1]

        # MLP处理
        mlp_features = []
        for i, (layer, feat) in enumerate(zip(self.mlp_layers, encoder_outputs)):
            mlp_features.append(layer(feat))

        # 特征融合
        fused_feature = torch.cat([f for f in mlp_features], dim=1)  # [B, 4*256, H/4, W/4]
        b, c, h, w = fused_feature.shape

        # 任务特征初始化
        task_features = [fused_feature.clone() for _ in range(self.num_tasks)]

        # 多任务处理
        for block in self.multi_task_blocks:
            task_features = block([f.view(b, h * w, c) for f in task_features])
            task_features = [f.view(b, c, h, w) for f in task_features]

        # 任务输出
        outputs = []
        for head, feat in zip(self.heads, task_features):
            outputs.append(head(feat))

        return outputs

# 示例测试
if __name__ == "__main__":
    model = MTLSegFormer(num_classes_per_task=[4, 4], num_tasks=2)
    x = torch.randn(2, 1, 224, 224)
    outputs = model(x)
    print(f"任务1输出形状：{outputs[0].shape}")  # [2, 4, 224, 224]
    print(f"任务2输出形状：{outputs[1].shape}")