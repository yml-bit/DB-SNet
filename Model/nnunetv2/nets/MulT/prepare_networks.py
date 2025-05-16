import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.swin_transformer import SwinTransformer
from opt_einsum import contract
from torch.utils.checkpoint import checkpoint

class SharedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, chunk_size=64):  # 缩小分块尺寸
        super().__init__()
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.scale = (dim // num_heads) ** -0.5

        # 共享参数初始化
        self.ref_qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.task_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def _compute_attn_chunk(self, q, k, v):
        """梯度检查点封装的计算函数"""
        # attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = contract('bhid,bhjd->bhij', q, k) * self.scale
        return torch.einsum('bhij,bhjd->bhid', F.softmax(attn, dim=-1), v)

    def _chunked_attention(self, q, k, v):
        B, H, L, D = q.shape
        output = torch.zeros_like(v, dtype=torch.float16)  # 半精度输出
        for i in range(0, L, self.chunk_size):
            q_chunk = q[:, :, i:i + self.chunk_size].half()
            k = k.half()
            v = v.half()

            # 梯度检查点
            chunk_out = checkpoint(
                self._compute_attn_chunk,
                q_chunk,
                k,
                v,
                use_reentrant=False
            )
            output[:, :, i:i + self.chunk_size] = chunk_out
            del q_chunk, chunk_out

        return output.float()  # 返回单精度

    def forward(self, x_ref, x_tasks):
        B, L, C = x_ref.shape

        # 生成共享QK
        qk = self.ref_qk(x_ref).reshape(B, L, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q_ref, k_ref = qk[0], qk[1]  # [B, H, L, D]

        # 分块计算注意力矩阵
        # with torch.cuda.amp.autocast():  # 混合精度
        #     attn = self._chunked_attention(q_ref, k_ref, k_ref)  # 示例用k_ref作为v计算

        outputs = []
        for x in x_tasks:
            # 生成当前任务V
            v = self.task_v(x).reshape(B, L, self.num_heads, -1).permute(0, 2, 1, 3)

            # 分块聚合
            x_out = self._chunked_attention(q_ref, k_ref, v)
            x_out = x_out.permute(0, 2, 1, 3).reshape(B, L, C)
            outputs.append(self.proj(x_out) + x)

        return outputs

class SharedCrossAttention11(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True):  # 移除chunk_size
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.ref_qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.task_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x_ref, x_tasks):
        B, L, C = x_ref.shape
        qk = self.ref_qk(x_ref).reshape(B, L, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q_ref, k_ref = qk[0], qk[1]  # [B, H, L, D]

        # 处理任务特征
        x_tasks_stacked = torch.stack(x_tasks, dim=0)  # [T, B, L, C]
        v = self.task_v(x_tasks_stacked).reshape(
            len(x_tasks), B, L, self.num_heads, -1
        ).permute(0, 1, 3, 2, 4)  # [T, B, H, L, D]

        # 计算核心注意力（无任务维度）
        attn = contract('bhid,bhjd->bhij', q_ref, k_ref) * self.scale  # [B, H, L, L]
        attn = F.softmax(attn, dim=-1)

        # 扩展注意力到任务维度
        attn = attn.unsqueeze(0).expand(len(x_tasks), -1, -1, -1, -1)  # [T, B, H, L, L]

        # 应用注意力到各任务的值向量
        x_out = torch.einsum('bhij,bhjd->bhid', attn, v)  # [T, B, H, L, D]
        x_out = x_out.permute(1, 3, 0, 2, 4).reshape(B, L, len(x_tasks), -1)  # [B, L, T, C]

        outputs = [self.proj(x_out[:, :, t]) + x_tasks[t] for t in range(len(x_tasks))]
        return outputs

class TaskSpecificDecoder(nn.Module):
    def __init__(self, dim, num_heads, num_tasks, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(dim // 2),
                'cross_attn': SharedCrossAttention(dim // 2, num_heads),
                'norm2': nn.LayerNorm(dim // 2),  # 单LayerNorm
                'ffn': nn.Sequential(  # 单FFN
                    nn.Linear(dim // 2, dim * 2),
                    nn.GELU(),
                    nn.Linear(dim * 2, dim // 2)
                )
            }) for _ in range(depth)])

        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim // 2 * 4, 3, padding=1),
            nn.PixelShuffle(2)
        )

    def forward(self, x_tasks, skip_conn):
        B, L, C = x_tasks[0].shape
        H, W = int(L ** 0.5), int(L ** 0.5)

        # 空间维度恢复
        x_tasks = [x.view(B, H, W, C).permute(0, 3, 1, 2) for x in x_tasks]
        x_tasks = [self.upsample(x) for x in x_tasks]
        x_tasks = [rearrange(x, 'b c h w -> b (h w) c') for x in x_tasks]

        if skip_conn is not None:
            x_ref = rearrange(skip_conn, 'b c h w -> b (h w) c')
            for blk in self.blocks:
                # 共享归一化
                x_tasks_norm = [blk['norm1'](x) for x in x_tasks]

                # 交叉注意力
                x_tasks = blk['cross_attn'](x_ref, x_tasks_norm)

                # 共享FFN + 残差 (关键修正)
                x_tasks = [blk['ffn'](blk['norm2'](x)) + x for x in x_tasks]

                # 跨任务残差连接
                x_tasks = [x + x_ref for x in x_tasks]

        return x_tasks

class MulT(nn.Module): #embed_dim=192
    def __init__(self, tasks_config, num_tasks=5, embed_dim=96, depths=[2, 2, 4, 2], num_heads=[6, 12, 24, 48]):#[6, 12, 24, 48]
        super().__init__()
        # ------------ 编码器 ------------
        self.encoder = SwinTransformer(
            # 其他参数保持不变
            in_chans=1,
            embed_dim=embed_dim,
            depths=depths,  # 原[2,2,18,2] -> 改为[2,2,6,2]
            num_heads=num_heads  # 原[6,12,24,48] -> 改为[3,6,12,24]
        )

        # ------------ 解码器 ------------
        self.decoders = nn.ModuleList([
            TaskSpecificDecoder(
                dim=embed_dim * (2 ** i),
                num_heads=max(num_heads)//(2**i),  # 动态减少头数
                num_tasks=num_tasks,
                depth=1  # 减少解码器深度
            ) for i in range(len(depths))
        ])
        # 记录各阶段特征图尺寸（假设输入为224x224）
        dimm=224//4
        self.feature_sizes = [dimm // (2 ** i) for i in range(4)]  # [224, 112, 56, 28]
        # ------------ 任务头 ------------
        self.heads = nn.ModuleDict()
        for task_name, config in tasks_config.items():
            if config['type'] == 'segmentation1':
                self.heads[task_name] = nn.Sequential(
                    nn.Conv2d(embed_dim // 2, 128, 3, padding=1),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.Conv2d(128, config['num_classes'], 1)
                )
            elif config['type'] == 'segmentation2':
                self.heads[task_name] = nn.Sequential(
                    nn.Conv2d(embed_dim // 2, 128, 3, padding=1),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.Conv2d(128, config['num_classes'], 1)
                )

    def forward(self, x):
        # ------------ 编码器 ------------
        x_enc = self.encoder.patch_embed(x)
        x_enc = self.encoder.pos_drop(x_enc)
        skip_connections = []
        B, L, C = x_enc.shape
        H = W = self.feature_sizes[0]  # 动态获取当前特征图尺寸
        x_enc_reshaped = x_enc.view(B, H, W, C)
        skip_connections.append(x_enc_reshaped.permute(0, 3, 1, 2))  # [B, C, H, W]

        for i, layer in enumerate(self.encoder.layers):
            x_enc = layer(x_enc)
            if i < len(self.encoder.layers) - 1:#最后一层没有进行下采样
                # 恢复空间维度 [B, L, C] -> [B, H, W, C]
                B, L, C = x_enc.shape
                H = W = self.feature_sizes[i+1]  # 动态获取当前特征图尺寸
                x_enc_reshaped = x_enc.view(B, H, W, C)
                skip_connections.append(x_enc_reshaped.permute(0, 3, 1, 2))  # [B, C, H, W]

        # ------------ 解码器 ------------
        # 初始化任务特征（使用编码器最终输出）
        task_feats = [x_enc.clone() for _ in range(len(self.heads))]

        # 分层解码
        for i, decoder in enumerate(self.decoders[::-1]):  # 反向连接编码器阶段
            if i<3:
                task_feats = decoder(
                task_feats,
                skip_conn=skip_connections[-(i+2)]
            )
            else:
                task_feats = decoder(
                    task_feats,
                    skip_conn=None
                )

        # ------------ 任务输出 ------------
        outputs = {}
        for i, (task_name, head) in enumerate(self.heads.items()):
            B, L, C = task_feats[i].shape
            H = W = self.feature_sizes[0]*2  # 动态获取当前特征图尺寸
            seg = task_feats[i].view(B, H, W, C)
            # 恢复空间维度 [B, L, C] -> [B, C, H, W]
            feat = seg.permute(0, 3, 1, 2)
            outputs[i] = head(feat)
        return outputs[0],outputs[1]

if __name__ == '__main__':
    tasks_config = {
        'sem_seg1': {'type': 'segmentation1', 'num_classes': 4},
        'sem_seg2': {'type': 'segmentation2', 'num_classes': 4}
        # 其他任务...
    }

    model = MulT(
        tasks_config=tasks_config,
        num_tasks=2,
        embed_dim=192,
        depths=[2, 2,6, 2],
        num_heads=[6, 12, 24, 48]
    )

    x = torch.randn(2, 1, 224, 224)
    outputs = model(x)
    print("Semantic Segmentation Shape:", outputs['sem_seg'].shape)  # [2, 21, 224, 224]
    print("Depth Estimation Shape:", outputs['depth'].shape)  # [2, 1, 224, 224]