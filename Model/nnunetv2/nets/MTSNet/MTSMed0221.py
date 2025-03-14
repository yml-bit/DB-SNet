import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
import functools

from .blocks import *

#上采样共享
class MedNeXt12(nn.Module):

    def __init__(self,
                 in_channels: int,
                 n_channels: int,
                 n_classes: int,
                 exp_r: int = 4,  # Expansion ratio as in Swin Transformers
                 kernel_size: int = 7,  # Ofcourse can test kernel_size
                 enc_kernel_size: int = None,
                 dec_kernel_size: int = None,
                 deep_supervision: bool = False,  # Can be used to test deep supervision
                 do_res: bool = False,  # Can be used to individually test residual connection
                 do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
                 checkpoint_style: bool = None,  # Either inside block or outside block
                 block_counts: list = [2, 2, 2, 2, 2, 2, 2, 2, 2],  # Can be used to test staging ratio:
                 # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
                 norm_type='group',
                 dim='3d',  # 2d or 3d
                 grn=False
                 ):

        super().__init__()

        self.do_ds = deep_supervision
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        assert dim in ['2d', '3d']

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d

        self.stem = conv(in_channels, n_channels, kernel_size=1)
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[0])]
                                         )

        self.down_0 = MedNeXtDownBlock(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim
        )

        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[1],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[1])]
                                         )

        self.down_1 = MedNeXtDownBlock(
            in_channels=2 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[2],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[2])]
                                         )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[3],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[3])]
                                         )

        self.down_3 = MedNeXtDownBlock(
            in_channels=8 * n_channels,
            out_channels=16 * n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 16,
                out_channels=n_channels * 16,
                exp_r=exp_r[4],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[4])]
                                        )

        self.up_3 = MedNeXtUpBlock(
            in_channels=16 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[5])]
                                         )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[6])]
                                         )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[7])]
                                         )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[8])]
                                         )

        # self.up_33 = MedNeXtUpBlock(
        #     in_channels=16 * n_channels,
        #     out_channels=8 * n_channels,
        #     exp_r=exp_r[5],
        #     kernel_size=dec_kernel_size,
        #     do_res=do_res_up_down,
        #     norm_type=norm_type,
        #     dim=dim,
        #     grn=grn
        # )

        self.dec_block_33 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[5])]
                                          )

        # self.up_22 = MedNeXtUpBlock(
        #     in_channels=8 * n_channels,
        #     out_channels=4 * n_channels,
        #     exp_r=exp_r[6],
        #     kernel_size=dec_kernel_size,
        #     do_res=do_res_up_down,
        #     norm_type=norm_type,
        #     dim=dim,
        #     grn=grn
        # )

        self.dec_block_22 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[6])]
                                          )

        # self.up_11 = MedNeXtUpBlock(
        #     in_channels=4 * n_channels,
        #     out_channels=2 * n_channels,
        #     exp_r=exp_r[7],
        #     kernel_size=dec_kernel_size,
        #     do_res=do_res_up_down,
        #     norm_type=norm_type,
        #     dim=dim,
        #     grn=grn
        # )

        self.dec_block_11 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[7])]
                                          )

        self.dec_block_00 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[8])]
                                          )
        self.cross_attention = CrossAttention3D(channel=n_channels, reduction=16)
        self.enhancer0 = SpatialAdaptiveEdgeEnhance3D(in_channels=n_channels)
        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)

        self.enhancer00 = SpatialAdaptiveEdgeEnhance3D(in_channels=n_channels)
        self.out_00 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)
        # self.cov_out_0 = nn.Conv3d(in_channels=n_channels, out_channels=n_classes,kernel_size=1)
        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)
        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels * 2, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=n_channels * 4, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=n_channels * 8, n_classes=n_classes, dim=dim)
            # self.out_4 = OutBlock(in_channels=n_channels*16, n_classes=n_classes, dim=dim)

            self.out_11 = OutBlock(in_channels=n_channels * 2, n_classes=n_classes, dim=dim)
            self.out_22 = OutBlock(in_channels=n_channels * 4, n_classes=n_classes, dim=dim)
            self.out_33 = OutBlock(in_channels=n_channels * 8, n_classes=n_classes, dim=dim)
            # self.out_44 = OutBlock(in_channels=n_channels*16, n_classes=n_classes, dim=dim)
        self.block_counts = block_counts

    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x

    def forward(self, x):

        x = self.stem(x)  # (2 32 56 160 256)
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor)  # 28
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor)  # 14
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor)  # 7
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor)  #

            x = self.iterative_checkpoint(self.bottleneck, x)
            # if self.do_ds:
            #     x_ds_4 = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor)

            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_3
            x1 = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.do_ds:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x1, self.dummy_tensor)

            x_up_33 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_33
            x2 = self.iterative_checkpoint(self.dec_block_33, dec_x)
            if self.do_ds:
                x_ds_33 = checkpoint.checkpoint(self.out_33, x2, self.dummy_tensor)

            del x_res_3, x_up_3, x_up_33

            x_up_2 = checkpoint.checkpoint(self.up_2, x1, self.dummy_tensor)
            dec_x = x_res_2 + x_up_2
            x1 = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x1, self.dummy_tensor)

            x_up_22 = checkpoint.checkpoint(self.up_2, x2, self.dummy_tensor)
            dec_x = x_res_2 + x_up_22
            x2 = self.iterative_checkpoint(self.dec_block_22, dec_x)
            if self.do_ds:
                x_ds_22 = checkpoint.checkpoint(self.out_22, x2, self.dummy_tensor)
            del x_res_2, x_up_2, x_up_22

            x_up_1 = checkpoint.checkpoint(self.up_1, x1, self.dummy_tensor)
            dec_x = x_res_1 + x_up_1
            x1 = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x1, self.dummy_tensor)

            x_up_11 = checkpoint.checkpoint(self.up_1, x2, self.dummy_tensor)
            dec_x = x_res_1 + x_up_11
            x2 = self.iterative_checkpoint(self.dec_block_11, dec_x)
            if self.do_ds:
                x_ds_11 = checkpoint.checkpoint(self.out_11, x2, self.dummy_tensor)
            del x_res_1, x_up_1, x_up_11

            x_up_0 = checkpoint.checkpoint(self.up_0, x1, self.dummy_tensor)
            dec_x = x_res_0 + x_up_0
            x1 = self.iterative_checkpoint(self.dec_block_0, dec_x)

            x_up_00 = checkpoint.checkpoint(self.up_0, x2, self.dummy_tensor)
            dec_x = x_res_0 + x_up_00
            x2 = self.iterative_checkpoint(self.dec_block_00, dec_x)
            del x_res_0, x_up_0, dec_x, x_up_00

            x1,x2= self.cross_attention(x1, x2)
            x1=self.enhancer0(x1)
            x2 = self.enhancer00(x2)
            x1 = checkpoint.checkpoint(self.out_0, x1, self.dummy_tensor)
            x2 = checkpoint.checkpoint(self.out_00, x2, self.dummy_tensor)

        if self.do_ds:
            return [x1, x_ds_1, x_ds_2, x_ds_3], [x2, x_ds_11, x_ds_22, x_ds_33]
        else:
            return x1, x2

#上采样部分共享
class MedNeXt13(nn.Module):

    def __init__(self,
                 in_channels: int,
                 n_channels: int,
                 n_classes: int,
                 exp_r: int = 4,  # Expansion ratio as in Swin Transformers
                 kernel_size: int = 7,  # Ofcourse can test kernel_size
                 enc_kernel_size: int = None,
                 dec_kernel_size: int = None,
                 deep_supervision: bool = False,  # Can be used to test deep supervision
                 do_res: bool = False,  # Can be used to individually test residual connection
                 do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
                 checkpoint_style: bool = None,  # Either inside block or outside block
                 block_counts: list = [2, 2, 2, 2, 2, 2, 2, 2, 2],  # Can be used to test staging ratio:
                 # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
                 norm_type='group',
                 dim='3d',  # 2d or 3d
                 grn=False
                 ):

        super().__init__()

        self.do_ds = deep_supervision
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        assert dim in ['2d', '3d']

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d

        self.stem = conv(in_channels, n_channels, kernel_size=1)
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[0])]
                                         )

        self.down_0 = MedNeXtDownBlock(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim
        )

        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[1],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[1])]
                                         )

        self.down_1 = MedNeXtDownBlock(
            in_channels=2 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[2],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[2])]
                                         )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[3],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[3])]
                                         )

        self.down_3 = MedNeXtDownBlock(
            in_channels=8 * n_channels,
            out_channels=16 * n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 16,
                out_channels=n_channels * 16,
                exp_r=exp_r[4],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[4])]
                                        )

        self.up_3 = MedNeXtUpBlock(
            in_channels=16 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[5])]
                                         )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[6])]
                                         )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[7])]
                                         )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[8])]
                                         )

        self.up_33 = MedNeXtUpBlock(
            in_channels=16 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_33 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[5])]
                                          )

        self.up_22 = MedNeXtUpBlock(
            in_channels=8 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_22 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[6])]
                                          )

        # self.up_11 = MedNeXtUpBlock(
        #     in_channels=4 * n_channels,
        #     out_channels=2 * n_channels,
        #     exp_r=exp_r[7],
        #     kernel_size=dec_kernel_size,
        #     do_res=do_res_up_down,
        #     norm_type=norm_type,
        #     dim=dim,
        #     grn=grn
        # )

        self.dec_block_11 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[7])]
                                          )

        # self.up_00 = MedNeXtUpBlock(
        #     in_channels=2 * n_channels,
        #     out_channels=n_channels,
        #     exp_r=exp_r[8],
        #     kernel_size=dec_kernel_size,
        #     do_res=do_res_up_down,
        #     norm_type=norm_type,
        #     dim=dim,
        #     grn=grn
        # )

        self.dec_block_00 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[8])]
                                          )
        self.cross_attention = CrossAttention3D(channel=n_channels, reduction=16)
        self.enhancer0 = SpatialAdaptiveEdgeEnhance3D(in_channels=n_channels)
        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)

        self.enhancer00 = SpatialAdaptiveEdgeEnhance3D(in_channels=n_channels)
        self.out_00 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)
        # self.cov_out_0 = nn.Conv3d(in_channels=n_channels, out_channels=n_classes,kernel_size=1)
        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)
        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels * 2, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=n_channels * 4, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=n_channels * 8, n_classes=n_classes, dim=dim)
            # self.out_4 = OutBlock(in_channels=n_channels*16, n_classes=n_classes, dim=dim)

            self.out_11 = OutBlock(in_channels=n_channels * 2, n_classes=n_classes, dim=dim)
            self.out_22 = OutBlock(in_channels=n_channels * 4, n_classes=n_classes, dim=dim)
            self.out_33 = OutBlock(in_channels=n_channels * 8, n_classes=n_classes, dim=dim)
            # self.out_44 = OutBlock(in_channels=n_channels*16, n_classes=n_classes, dim=dim)
        self.block_counts = block_counts

    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x

    def forward(self, x):

        x = self.stem(x)  # (2 32 56 160 256)
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor)  # 28
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor)  # 14
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor)  # 7
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor)  #

            x = self.iterative_checkpoint(self.bottleneck, x)
            # if self.do_ds:
            #     x_ds_4 = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor)

            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_3
            x1 = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.do_ds:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x1, self.dummy_tensor)

            x_up_33 = checkpoint.checkpoint(self.up_33, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_33
            x2 = self.iterative_checkpoint(self.dec_block_33, dec_x)
            if self.do_ds:
                x_ds_33 = checkpoint.checkpoint(self.out_33, x2, self.dummy_tensor)

            del x_res_3, x_up_3, x_up_33

            x_up_2 = checkpoint.checkpoint(self.up_2, x1, self.dummy_tensor)
            dec_x = x_res_2 + x_up_2
            x1 = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x1, self.dummy_tensor)

            x_up_22 = checkpoint.checkpoint(self.up_22, x2, self.dummy_tensor)
            dec_x = x_res_2 + x_up_22
            x2 = self.iterative_checkpoint(self.dec_block_22, dec_x)
            if self.do_ds:
                x_ds_22 = checkpoint.checkpoint(self.out_22, x2, self.dummy_tensor)
            del x_res_2, x_up_2, x_up_22

            x_up_1 = checkpoint.checkpoint(self.up_1, x1, self.dummy_tensor)
            dec_x = x_res_1 + x_up_1
            x1 = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x1, self.dummy_tensor)

            x_up_11 = checkpoint.checkpoint(self.up_1, x2, self.dummy_tensor)
            dec_x = x_res_1 + x_up_11
            x2 = self.iterative_checkpoint(self.dec_block_11, dec_x)
            if self.do_ds:
                x_ds_11 = checkpoint.checkpoint(self.out_11, x2, self.dummy_tensor)
            del x_res_1, x_up_1, x_up_11

            x_up_0 = checkpoint.checkpoint(self.up_0, x1, self.dummy_tensor)
            dec_x = x_res_0 + x_up_0
            x1 = self.iterative_checkpoint(self.dec_block_0, dec_x)

            x_up_00 = checkpoint.checkpoint(self.up_0, x2, self.dummy_tensor)
            dec_x = x_res_0 + x_up_00
            x2 = self.iterative_checkpoint(self.dec_block_00, dec_x)
            del x_res_0, x_up_0, dec_x, x_up_00

            x1,x2= self.cross_attention(x1, x2)
            x1=self.enhancer0(x1)
            x2 = self.enhancer00(x2)
            x1 = checkpoint.checkpoint(self.out_0, x1, self.dummy_tensor)
            x2 = checkpoint.checkpoint(self.out_00, x2, self.dummy_tensor)

        if self.do_ds:
            return [x1, x_ds_1, x_ds_2, x_ds_3], [x2, x_ds_11, x_ds_22, x_ds_33]
        else:
            return x1, x2

class CrossAttention3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.channel = channel
        self.reduction = reduction

        # Channel attention for x1 and x2
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.channel_excitation_x1 = nn.Sequential(
            nn.Linear(channel, int(channel // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel // reduction), channel)
        )
        self.channel_excitation_x2 = nn.Sequential(
            nn.Linear(channel, int(channel // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel // reduction), channel)
        )

        # Spatial attention for x1 and x2
        self.spatial_se_x1 = nn.Conv3d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.spatial_se_x2 = nn.Conv3d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x1, x2):
        B, C, D, H, W = x1.size()
        # Channel attention for x1
        chn_se_x1 = self.avg_pool(x1).view(B, C)
        chn_se_x1 = torch.sigmoid(self.channel_excitation_x1(chn_se_x1).view(B, C, 1, 1, 1))
        chn_se_x1 = torch.mul(x1, chn_se_x1)

        # Channel attention for x2
        chn_se_x2 = self.avg_pool(x2).view(B, C)
        chn_se_x2 = torch.sigmoid(self.channel_excitation_x2(chn_se_x2).view(B, C, 1, 1, 1))
        chn_se_x2 = torch.mul(x2, chn_se_x2)

        # Spatial attention for x1
        spa_se_x1 = torch.sigmoid(self.spatial_se_x1(x1))
        spa_se_x1 = torch.mul(x1, spa_se_x1)

        # Spatial attention for x2
        spa_se_x2 = torch.sigmoid(self.spatial_se_x2(x2))
        spa_se_x2 = torch.mul(x2, spa_se_x2)

        net_out1 = spa_se_x1 + chn_se_x1 +spa_se_x2 + chn_se_x2+x1  # Enhanced x1 with cross-attention
        # Combine features
        net_out2 = spa_se_x1 + chn_se_x1 +spa_se_x2 + chn_se_x2 +x2  # Enhanced x1 with cross-attention

        return net_out1,net_out2

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

    network = MedNeXt12(
            in_channels = 1, 
            n_channels = 32,
            n_classes = 13,
            exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
            # exp_r = 2,
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            # block_counts = [2,2,2,2,2,2,2,2,2],
            block_counts = [3,4,8,8,8,8,8,4,3],
            checkpoint_style = None,
            dim = '2d',
            grn=True
            
        ).cuda()


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(network))

    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import parameter_count_table

    # model = ResTranUnet(img_size=128, in_channels=1, num_classes=14, dummy=False).cuda()
    x = torch.zeros((1,1,64,64,64), requires_grad=False).cuda()
    flops = FlopCountAnalysis(network, x)
    print(flops.total())
    
    with torch.no_grad():
        print(network)
        x = torch.zeros((1, 1, 128, 128, 128)).cuda()
        print(network(x)[0].shape)
