import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
import functools
from .blocks import *

#粗暴双分支，非深度监督
class DBSNet0(nn.Module):
    def __init__(self,
        in_channels: int,
        n_channels: int,
        n_classes: int,
        exp_r: int = 4,                            # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,                      # Ofcourse can test kernel_size
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,             # Can be used to test deep supervision
        do_res: bool = False,                       # Can be used to individually test residual connection
        do_res_up_down: bool = False,             # Additional 'res' connection on up and down convs
        checkpoint_style: bool = None,            # Either inside block or outside block
        block_counts: list = [2,2,2,2,2,2,2,2,2], # Can be used to test staging ratio:
                                            # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
        norm_type = 'group',
        dim = '3d',                                # 2d or 3d
        grn = False
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
            out_channels=2*n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim
        )

        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
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
            in_channels=2*n_channels,
            out_channels=4*n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
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
            in_channels=4*n_channels,
            out_channels=8*n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*8,
                out_channels=n_channels*8,
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
            in_channels=8*n_channels,
            out_channels=16*n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*16,
                out_channels=n_channels*16,
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
            in_channels=16*n_channels,
            out_channels=8*n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*8,
                out_channels=n_channels*8,
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
            in_channels=8*n_channels,
            out_channels=4*n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
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
            in_channels=4*n_channels,
            out_channels=2*n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
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
            in_channels=2*n_channels,
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

        self.up_11 = MedNeXtUpBlock(
            in_channels=4 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

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

        self.up_00 = MedNeXtUpBlock(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
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

        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)
        self.out_00 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)
        # self.cov_out_0 = nn.Conv3d(in_channels=n_channels, out_channels=n_classes,kernel_size=1)
        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)
        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels*2, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=n_channels*4, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=n_channels*8, n_classes=n_classes, dim=dim)
            # self.out_4 = OutBlock(in_channels=n_channels*16, n_classes=n_classes, dim=dim)

            self.out_11 = OutBlock(in_channels=n_channels*2, n_classes=n_classes, dim=dim)
            self.out_22 = OutBlock(in_channels=n_channels*4, n_classes=n_classes, dim=dim)
            self.out_33 = OutBlock(in_channels=n_channels*8, n_classes=n_classes, dim=dim)
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

        x = self.stem(x) #(2 32 56 160 256)
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor) #28
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor) #14
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor) #7
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor) #

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

            del x_res_3, x_up_3,x_up_33

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
            del x_res_2, x_up_2,x_up_22

            x_up_1 = checkpoint.checkpoint(self.up_1, x1, self.dummy_tensor)
            dec_x = x_res_1 + x_up_1
            x1 = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x1, self.dummy_tensor)

            x_up_11 = checkpoint.checkpoint(self.up_11, x2, self.dummy_tensor)
            dec_x = x_res_1 + x_up_11
            x2 = self.iterative_checkpoint(self.dec_block_11, dec_x)
            if self.do_ds:
                x_ds_11 = checkpoint.checkpoint(self.out_11, x2, self.dummy_tensor)
            del x_res_1, x_up_1,x_up_11

            x_up_0 = checkpoint.checkpoint(self.up_0, x1, self.dummy_tensor)
            dec_x = x_res_0 + x_up_0
            x1 = self.iterative_checkpoint(self.dec_block_0, dec_x)

            x_up_00 = checkpoint.checkpoint(self.up_00, x2, self.dummy_tensor)
            dec_x = x_res_0 + x_up_00
            x2 = self.iterative_checkpoint(self.dec_block_00, dec_x)
            del x_res_0, x_up_0, dec_x,x_up_00

            x1 = checkpoint.checkpoint(self.out_0, x1, self.dummy_tensor)
            x2 = checkpoint.checkpoint(self.out_00, x2, self.dummy_tensor)
        if self.do_ds:
            return [x1, x_ds_1, x_ds_2, x_ds_3], [x2, x_ds_11, x_ds_22, x_ds_33]
        else:
            return x1, x2

#3处简单拼接concaten
class DBSNet1(nn.Module):
    def __init__(self,
        in_channels: int,
        n_channels: int,
        n_classes: int,
        exp_r: int = 4,                            # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,                      # Ofcourse can test kernel_size
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,             # Can be used to test deep supervision
        do_res: bool = False,                       # Can be used to individually test residual connection
        do_res_up_down: bool = False,             # Additional 'res' connection on up and down convs
        checkpoint_style: bool = None,            # Either inside block or outside block
        block_counts: list = [2,2,2,2,2,2,2,2,2], # Can be used to test staging ratio:
                                            # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
        norm_type = 'group',
        dim = '3d',                                # 2d or 3d
        grn = False
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
            out_channels=2*n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim
        )

        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
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
            in_channels=2*n_channels,
            out_channels=4*n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
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
            in_channels=4*n_channels,
            out_channels=8*n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*8,
                out_channels=n_channels*8,
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
            in_channels=8*n_channels,
            out_channels=16*n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*16,
                out_channels=n_channels*16,
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
            in_channels=16*n_channels,
            out_channels=8*n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 8,  # 第一个块输入不同
                out_channels=n_channels * 8,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[5])
        ])

        self.up_2 = MedNeXtUpBlock(
            in_channels=8*n_channels,
            out_channels=4*n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*8 if i == 0 else n_channels * 4,
                out_channels=n_channels*4,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=False if i==0 else do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[6])]
        )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4*n_channels,
            out_channels=2*n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*4 if i == 0 else n_channels * 2,
                out_channels=n_channels*2,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=False if i==0 else do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[7])]
        )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2*n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=False,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*2 if i == 0 else n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=False if i==0 else do_res,
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
                in_channels=n_channels * 8 if i == 0 else n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=False if i==0 else do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[6])]
                                         )

        self.up_11 = MedNeXtUpBlock(
            in_channels=4 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_11 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 4 if i == 0 else n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=False if i==0 else do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[7])]
                                         )

        self.up_00 = MedNeXtUpBlock(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_00 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*2 if i == 0 else n_channels * 1,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=False if i==0 else do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[8])]
                                         )
        # self.m1 = conv(16 * n_channels, 8 * n_channels, kernel_size=1)
        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)
        self.out_00 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)
        # self.cov_out_0 = nn.Conv3d(in_channels=n_channels, out_channels=n_classes,kernel_size=1)
        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)
        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels*2, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=n_channels*4, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=n_channels*8, n_classes=n_classes, dim=dim)
            # self.out_4 = OutBlock(in_channels=n_channels*16, n_classes=n_classes, dim=dim)

            self.out_11 = OutBlock(in_channels=n_channels*2, n_classes=n_classes, dim=dim)
            self.out_22 = OutBlock(in_channels=n_channels*4, n_classes=n_classes, dim=dim)
            self.out_33 = OutBlock(in_channels=n_channels*8, n_classes=n_classes, dim=dim)
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

        x = self.stem(x) #(2 32 56 160 256)
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor) #28
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor) #14
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor) #7
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor) #

            x = self.iterative_checkpoint(self.bottleneck, x)
            # if self.do_ds:
            #     x_ds_4 = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor)

            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_3
            x_up_33 = checkpoint.checkpoint(self.up_33, x, self.dummy_tensor)
            dec_x2 = x_res_3 + x_up_33

            # dec_x=torch.concatenate([dec_x,dec_x2],dim=1)
            x1 = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.do_ds:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x1, self.dummy_tensor)

            x2 = self.iterative_checkpoint(self.dec_block_33, dec_x2)
            if self.do_ds:
                x_ds_33 = checkpoint.checkpoint(self.out_33, x2, self.dummy_tensor)

            del x_res_3, x_up_3,x_up_33

            x_up_2 = checkpoint.checkpoint(self.up_2, x1, self.dummy_tensor)
            dec_x = x_res_2 + x_up_2
            x_up_22 = checkpoint.checkpoint(self.up_22, x2, self.dummy_tensor)
            dec_x2 = x_res_2 + x_up_22
            dec_x = torch.concatenate([dec_x, dec_x2], dim=1)

            x1 = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x1, self.dummy_tensor)

            x2 = self.iterative_checkpoint(self.dec_block_22, dec_x)
            if self.do_ds:
                x_ds_22 = checkpoint.checkpoint(self.out_22, x2, self.dummy_tensor)
            del x_res_2, x_up_2,x_up_22

            x_up_1 = checkpoint.checkpoint(self.up_1, x1, self.dummy_tensor)
            dec_x = x_res_1 + x_up_1
            x_up_11 = checkpoint.checkpoint(self.up_11, x2, self.dummy_tensor)
            dec_x2 = x_res_1 + x_up_11
            dec_x = torch.concatenate([dec_x, dec_x2], dim=1)
            x1 = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x1, self.dummy_tensor)

            x2 = self.iterative_checkpoint(self.dec_block_11, dec_x)
            if self.do_ds:
                x_ds_11 = checkpoint.checkpoint(self.out_11, x2, self.dummy_tensor)
            del x_res_1, x_up_1,x_up_11

            x_up_0 = checkpoint.checkpoint(self.up_0, x1, self.dummy_tensor)
            dec_x = x_res_0 + x_up_0
            x_up_00 = checkpoint.checkpoint(self.up_00, x2, self.dummy_tensor)
            dec_x2 = x_res_0 + x_up_00
            dec_x = torch.concatenate([dec_x, dec_x2], dim=1)

            x1 = self.iterative_checkpoint(self.dec_block_0, dec_x)
            x2 = self.iterative_checkpoint(self.dec_block_00, dec_x)
            del x_res_0, x_up_0, dec_x,x_up_00

            x1 = checkpoint.checkpoint(self.out_0, x1, self.dummy_tensor)
            x2 = checkpoint.checkpoint(self.out_00, x2, self.dummy_tensor)
        if self.do_ds:
            return [x1, x_ds_1, x_ds_2, x_ds_3], [x2, x_ds_11, x_ds_22, x_ds_33]
        else:
            return x1, x2

#末端3处双分支信息交互 MTLSegFormer，运行提示超出内存
class DBSNet2(nn.Module):
    def __init__(self,
        in_channels: int,
        n_channels: int,
        n_classes: int,
        exp_r: int = 4,                            # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,                      # Ofcourse can test kernel_size
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,             # Can be used to test deep supervision
        do_res: bool = False,                       # Can be used to individually test residual connection
        do_res_up_down: bool = False,             # Additional 'res' connection on up and down convs
        checkpoint_style: bool = None,            # Either inside block or outside block
        block_counts: list = [2,2,2,2,2,2,2,2,2], # Can be used to test staging ratio:
                                            # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
        norm_type = 'group',
        dim = '3d',                                # 2d or 3d
        grn = False
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
            out_channels=2*n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim
        )

        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
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
            in_channels=2*n_channels,
            out_channels=4*n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
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
            in_channels=4*n_channels,
            out_channels=8*n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*8,
                out_channels=n_channels*8,
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
            in_channels=8*n_channels,
            out_channels=16*n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*16,
                out_channels=n_channels*16,
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
            in_channels=16*n_channels,
            out_channels=8*n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*8,
                out_channels=n_channels*8,
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
            in_channels=8*n_channels,
            out_channels=4*n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
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
            in_channels=4*n_channels,
            out_channels=2*n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
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
            in_channels=2*n_channels,
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

        self.up_11 = MedNeXtUpBlock(
            in_channels=4 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

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

        self.up_00 = MedNeXtUpBlock(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
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
        self.cross_attention0 = DualTargetCrossAttention(in_channels=n_channels)
        self.cross_attention1= DualTargetCrossAttention(in_channels=n_channels*2)
        self.cross_attention2 = DualTargetCrossAttention(in_channels=n_channels * 4)
        # self.cross_attention3 = DualTargetCrossAttention(in_channels=n_channels * 8)

        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)
        self.out_00 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)
        # self.cov_out_0 = nn.Conv3d(in_channels=n_channels, out_channels=n_classes,kernel_size=1)
        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)
        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels*2, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=n_channels*4, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=n_channels*8, n_classes=n_classes, dim=dim)
            # self.out_4 = OutBlock(in_channels=n_channels*16, n_classes=n_classes, dim=dim)

            self.out_11 = OutBlock(in_channels=n_channels*2, n_classes=n_classes, dim=dim)
            self.out_22 = OutBlock(in_channels=n_channels*4, n_classes=n_classes, dim=dim)
            self.out_33 = OutBlock(in_channels=n_channels*8, n_classes=n_classes, dim=dim)
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

        x = self.stem(x) #(2 32 56 160 256)
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor) #28
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor) #14
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor) #7
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor) #

            x = self.iterative_checkpoint(self.bottleneck, x)
            # if self.do_ds:
            #     x_ds_4 = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor)

            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_3
            x1 = self.iterative_checkpoint(self.dec_block_3, dec_x)

            x_up_33 = checkpoint.checkpoint(self.up_33, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_33
            x2 = self.iterative_checkpoint(self.dec_block_33, dec_x)
            # x1, x2 = checkpoint.checkpoint(self.cross_attention3, (x1, x2))
            if self.do_ds:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x1, self.dummy_tensor)
            if self.do_ds:
                x_ds_33 = checkpoint.checkpoint(self.out_33, x2, self.dummy_tensor)

            del x_res_3, x_up_3,x_up_33

            x_up_2 = checkpoint.checkpoint(self.up_2, x1, self.dummy_tensor)
            dec_x = x_res_2 + x_up_2
            x1 = self.iterative_checkpoint(self.dec_block_2, dec_x)
            x_up_22 = checkpoint.checkpoint(self.up_22, x2, self.dummy_tensor)
            dec_x = x_res_2 + x_up_22
            x2 = self.iterative_checkpoint(self.dec_block_22, dec_x)
            x1, x2 = self.cross_attention2(x1, x2)
            # x1, x2 = checkpoint.checkpoint(self.cross_attention2, (x1, x2))
            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x1, self.dummy_tensor)

            if self.do_ds:
                x_ds_22 = checkpoint.checkpoint(self.out_22, x2, self.dummy_tensor)
            del x_res_2, x_up_2,x_up_22

            x_up_1 = checkpoint.checkpoint(self.up_1, x1, self.dummy_tensor)
            dec_x = x_res_1 + x_up_1
            x1 = self.iterative_checkpoint(self.dec_block_1, dec_x)
            x_up_11 = checkpoint.checkpoint(self.up_11, x2, self.dummy_tensor)
            dec_x = x_res_1 + x_up_11
            x2 = self.iterative_checkpoint(self.dec_block_11, dec_x)
            x1, x2 = self.cross_attention1(x1, x2)
            # x1, x2 = checkpoint.checkpoint(self.cross_attention1, (x1, x2))
            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x1, self.dummy_tensor)
            if self.do_ds:
                x_ds_11 = checkpoint.checkpoint(self.out_11, x2, self.dummy_tensor)
            del x_res_1, x_up_1,x_up_11

            x_up_0 = checkpoint.checkpoint(self.up_0, x1, self.dummy_tensor)
            dec_x = x_res_0 + x_up_0
            x1 = self.iterative_checkpoint(self.dec_block_0, dec_x)

            x_up_00 = checkpoint.checkpoint(self.up_00, x2, self.dummy_tensor)
            dec_x = x_res_0 + x_up_00
            x2 = self.iterative_checkpoint(self.dec_block_00, dec_x)
            del x_res_0, x_up_0, dec_x,x_up_00
            x1,x2=self.cross_attention0(x1,x2)
            # x1, x2 = checkpoint.checkpoint(self.cross_attention0, (x1, x2))
            x1 = checkpoint.checkpoint(self.out_0, x1, self.dummy_tensor)
            x2 = checkpoint.checkpoint(self.out_00, x2, self.dummy_tensor)
        if self.do_ds:
            return [x1, x_ds_1, x_ds_2, x_ds_3], [x2, x_ds_11, x_ds_22, x_ds_33]
        else:
            return x1, x2


#Shared_attention_DecoderBlock
class DBSNet4(nn.Module):
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

        # self.up_3=Shared_attention_DecoderBlock(in_channels=16 * n_channels, n_filters=64,
        #                               rla_channel=8 * n_channels,SE=True, ECA_size=5, reduction=16)
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

        # self.up_2 = MedNeXtUpBlock(
        #     in_channels=8 * n_channels,
        #     out_channels=4 * n_channels,
        #     exp_r=exp_r[6],
        #     kernel_size=dec_kernel_size,
        #     do_res=do_res_up_down,
        #     norm_type=norm_type,
        #     dim=dim,
        #     grn=grn
        # )
        self.up_2 = Shared_attention_DecoderBlock(in_channels=8 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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

        self.up_1=Shared_attention_DecoderBlock(in_channels=4 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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

        self.up_0=Shared_attention_DecoderBlock(in_channels=2 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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

        # self.up_22 = MedNeXtUpBlock(
        #     in_channels=8 * n_channels,
        #     out_channels=4 * n_channels,
        #     exp_r=exp_r[4],
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

        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)
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

            x_up_33 = checkpoint.checkpoint(self.up_33, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_33
            x2 = self.iterative_checkpoint(self.dec_block_33, dec_x)

            if self.do_ds:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x1, self.dummy_tensor)
            if self.do_ds:
                x_ds_33 = checkpoint.checkpoint(self.out_33, x2, self.dummy_tensor)

            del x_res_3, x_up_3, x_up_33
            x_up_2,x_up_22=self.up_2(x1,x2)
            # x_up_2 = checkpoint.checkpoint(self.up_2, x1, self.dummy_tensor)
            dec_x = x_res_2 + x_up_2
            x1 = self.iterative_checkpoint(self.dec_block_2, dec_x)
            # x_up_22 = checkpoint.checkpoint(self.up_22, x2, self.dummy_tensor)
            dec_x = x_res_2 + x_up_22
            x2 = self.iterative_checkpoint(self.dec_block_22, dec_x)

            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x1, self.dummy_tensor)

            if self.do_ds:
                x_ds_22 = checkpoint.checkpoint(self.out_22, x2, self.dummy_tensor)
            del x_res_2, x_up_2, x_up_22
            x_up_1, x_up_11 = self.up_1(x1, x2)
            dec_x = x_res_1 + x_up_1
            x1 = self.iterative_checkpoint(self.dec_block_1, dec_x)
            dec_x = x_res_1 + x_up_11
            x2 = self.iterative_checkpoint(self.dec_block_11, dec_x)

            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x1, self.dummy_tensor)
            if self.do_ds:
                x_ds_11 = checkpoint.checkpoint(self.out_11, x2, self.dummy_tensor)
            del x_res_1, x_up_1, x_up_11
            x_up_0, x_up_00 = self.up_0(x1, x2)
            dec_x = x_res_0 + x_up_0
            x1 = self.iterative_checkpoint(self.dec_block_0, dec_x)
            dec_x = x_res_0 + x_up_00
            x2 = self.iterative_checkpoint(self.dec_block_00, dec_x)
            del x_res_0, x_up_0, dec_x, x_up_00

            x1 = checkpoint.checkpoint(self.out_0, x1, self.dummy_tensor)
            x2 = checkpoint.checkpoint(self.out_00, x2, self.dummy_tensor)
        if self.do_ds:
            return [x1, x_ds_1, x_ds_2, x_ds_3], [x2, x_ds_11, x_ds_22, x_ds_33]
        else:
            return x1, x2

class ChannelGate3D(nn.Module):
    """3D通道注意力门控"""

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.mlp1 = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel))
        self.mlp2 = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel))

    def forward(self, x):
        B, C, _, _, _ = x.shape
        avg_out = self.mlp1(self.avg_pool(x).view(B, C))
        max_out = self.mlp2(self.max_pool(x).view(B, C))
        return (avg_out + max_out).view(B, C, 1, 1, 1)

class SpatialGate3D(nn.Module):
    """3D空间注意力（强制维度保持）"""
    def __init__(self, channel, kernel_size=5):
        super().__init__()
        kernel_size = kernel_size if kernel_size%2==1 else kernel_size+1
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size, padding=padding),
            nn.BatchNorm3d(1)
        )

    def forward(self, x):
        B, _, D, H, W = x.shape
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(cat)
        assert out.shape[2:] == (D, H, W), f"空间维度不匹配 {out.shape} vs {(D, H, W)}"
        return torch.sigmoid(out)

#参考MTLSegFormer，自注意力+交叉注意力+空间注意力
#分支+跨分支+MLP+空间增强
class DualTargetCrossAttention(nn.Module):
    """优化后的双分支交互模块"""

    def __init__(self, in_channels, num_heads=4, expansion=2, drop_rate=0.1):
        super().__init__()
        # 独立归一化层
        self.norm1 = nn.InstanceNorm3d(in_channels)  # 分支1归一化
        self.norm2 = nn.InstanceNorm3d(in_channels)  # 分支2归一化

        # 交叉注意力机制
        self.cross_attn1 = EfficientCrossAttention(in_channels)
        self.cross_attn2 = EfficientCrossAttention(in_channels)

        # 独立MLP路径
        self.mlp1 = self._make_mlp(in_channels, expansion, drop_rate)
        self.mlp2 = self._make_mlp(in_channels, expansion, drop_rate)

        # 空间增强
        self.spatial_conv = nn.Conv3d(in_channels, in_channels, 3, padding=1)

    def _make_mlp(self, channels, expansion, dropout):
        return nn.Sequential(
            nn.Conv3d(channels, channels * expansion, 1),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channels * expansion, channels, 1)
        )

    def forward(self, x1, x2):
        # x1, x2 = x

        # 交叉注意力
        x1_cross = x1 + self.cross_attn1(self.norm1(x1), self.norm2(x2))  # 分支1查询分支2
        x2_cross = x2 + self.cross_attn2(self.norm2(x2), self.norm1(x1))  # 分支2查询分支1

        # 特征融合
        x1_out = x1_cross + self.mlp1(self.norm1(x1_cross))
        x2_out = x2_cross + self.mlp2(self.norm2(x2_cross))

        # 空间信息增强
        x1_out = x1_out + self.spatial_conv(x1_out)
        x2_out = x2_out + self.spatial_conv(x2_out)

        return x1_out, x2_out

class EfficientCrossAttention(nn.Module):
    """优化后的交叉注意力模块"""

    def __init__(self, in_channels):
        super().__init__()
        self.channel_att = ChannelGate3D(in_channels)
        self.spatial_att = SpatialGate3D(in_channels)

        # 可分离卷积
        self.dw_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.InstanceNorm3d(in_channels)
        )

    def forward(self, query, key_value):
        """
        query: 当前分支特征 [B,C,D,H,W]
        key_value: 另一分支特征 [B,C,D,H,W]
        """
        # 特征增强
        kv = self.dw_conv(key_value)

        # 双路注意力
        channel_att = self.channel_att(kv)
        spatial_att = self.spatial_att(kv)

        # 注意力融合
        att = torch.sigmoid(channel_att + spatial_att)
        return query * att  # 返回注意力加权的查询特征

class DualTargetCrossAttention11(nn.Module):
    def __init__(self, in_channels,embed_dim=64, n_heads=8, spatial_scale=4):
        """
        双目标3D交叉注意力模块
        参数:
            embed_dim: 特征嵌入维度
            n_heads: 注意力头数
            spatial_scale: 空间降采样比例
        """
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim必须能被n_heads整除"

        # 核心参数
        self.n_heads = n_heads
        self.d_k = embed_dim // n_heads
        self.spatial_scale = spatial_scale

        # 特征投影层（提升通道数）
        self.embed_dim = embed_dim
        self.proj = None  # 动态初始化 proj 层

        # 多尺度处理（参考C2FNAS[9]）
        self.downsample = nn.MaxPool3d(kernel_size=spatial_scale)
        self.upsample = nn.Upsample(scale_factor=spatial_scale, mode='trilinear', align_corners=False)

        # 双向交叉注意力
        self.cross_attn1 = CrossAttentionUnit(in_channels, n_heads)  # Branch1 -> Branch2
        self.cross_attn2 = CrossAttentionUnit(in_channels, n_heads)  # Branch2 -> Branch1

        # 自适应融合（参考G-SWPA[8]）
        self.fusion_gate = nn.Sequential(
            nn.Conv3d(in_channels * 2, embed_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(embed_dim // 2, 2, 1),  # 输出2个权重通道
            nn.Sigmoid()
        )

        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.GELU()
        )
    def forward(self, x):
        x1 = x[0]
        x2 = x[1]

        # 特征投影
        f1 = self.proj(x1)  # [B, embed_dim, D, H, W]
        f2 = self.proj(x2)

        # 多尺度处理
        f1_ds = self.downsample(f1)
        f2_ds = self.downsample(f2)

        # 双向交叉注意力
        attn1 = self.cross_attn1(f1_ds, f2_ds)  # Branch1关注Branch2
        attn2 = self.cross_attn2(f2_ds, f1_ds)  # Branch2关注Branch1

        # 上采样注意力结果，确保形状与输入一致
        attn1 = F.interpolate(attn1, size=x1.shape[2:], mode='trilinear', align_corners=False)
        attn2 = F.interpolate(attn2, size=x2.shape[2:], mode='trilinear', align_corners=False)

        # 门控融合
        gate = self.fusion_gate(torch.cat([f1, f2], dim=1))  # [B, 2, D, H, W]

        # 加权融合
        out1 = gate[:, 0:1] * x1 + (1 - gate[:, 0:1]) * attn1
        out2 = gate[:, 1:2] * x2 + (1 - gate[:, 1:2]) * attn2

        return out1, out2

class CrossAttentionUnit(nn.Module):
    """ 单向交叉注意力单元 """

    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = embed_dim // n_heads

        # 查询/键/值投影
        self.W_q = nn.Conv3d(embed_dim, embed_dim, 1)
        self.W_kv = nn.Conv3d(embed_dim, embed_dim * 2, 1)

    def forward(self, query_feat, context_feat):
        B, C, D, H, W = query_feat.shape
        # 投影Q/K/V
        Q = self.W_q(query_feat)  # [B, C, D, H, W]
        K, V = self.W_kv(context_feat).split([C, C], dim=1)
        # 多头拆分
        Q = Q.view(B, self.n_heads, self.d_k, D, H, W)
        K = K.view(B, self.n_heads, self.d_k, D, H, W)
        V = V.view(B, self.n_heads, self.d_k, D, H, W)
        # 3D注意力计算（空间维度展平）
        Q = Q.flatten(3)  # [B, h, d_k, D*H*W]
        K = K.flatten(3).transpose(-1, -2)  # [B, h, D*H*W, d_k]
        V = V.flatten(3)  # [B, h, d_k, D*H*W]
        # 注意力分数
        scores = torch.matmul(Q, K) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        # 特征聚合
        output = torch.matmul(attn, V)
        # 恢复形状
        output = output.view(B, C, D, H, W)
        return output

#MTSegFormer #分支+跨分支+空间，和 MTLSegFormer 非常相似
# class MTSegFormer(nn.Module):
#     """3D医学图像双分支Transformer模块"""
#     def __init__(self, in_channels, num_heads, reduction_ratios=[2, 4]):
#         super().__init__()
#
#         # 共享参数
#         self.mlp = nn.Sequential(
#             nn.Conv3d(in_channels, in_channels * 2, 1),
#             nn.GELU(),
#             nn.Conv3d(in_channels * 2, in_channels, 1)
#         )
#
#         # 分支特定模块
#         self.norm1_b1 = nn.InstanceNorm3d(in_channels)
#         self.attn_b1 = EfficientSelfAttention(in_channels, num_heads, reduction_ratios[0])
#
#         self.norm1_b2 = nn.InstanceNorm3d(in_channels)
#         self.attn_b2 = EfficientSelfAttention(in_channels, num_heads, reduction_ratios[1])
#
#         # 跨分支注意力
#         self.cross_attn = EfficientSelfAttention(in_channels*2, num_heads)
#
#         # 输出归一化
#         self.norm2 = nn.InstanceNorm3d(in_channels)
#
#     def forward(self, x1, x2):
#         # 分支1处理
#         identity1 = x1
#         x1 = self.norm1_b1(x1)
#         x1 = self.attn_b1(x1) + x1
#
#         # 分支2处理
#         identity2 = x2
#         x2 = self.norm1_b2(x2)
#         x2 = self.attn_b2(x2) + x2
#
#         # 跨分支交互
#         cross_feat = self.cross_attn(torch.cat([x1, x2], dim=1))  # 通道拼接
#         x1 = x1 + cross_feat[:, :x1.shape[1]]
#         x2 = x2 + cross_feat[:, x1.shape[1]:]
#
#         # MLP处理
#         x1 = identity1 + self.mlp(self.norm2(x1))
#         x2 = identity2 + self.mlp(self.norm2(x2))
#
#         return x1, x2

#I2U
#decoder_block = Shared_attention_DecoderBlock(in_channels=in_channels, n_filters=64,
        # rla_channel=rla_channel,SE=True, ECA_size=5, reduction=16)
class Shared_attention_DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, SE=False, ECA_size=5, reduction=16):
        super(Shared_attention_DecoderBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels*2, in_channels // 2, kernel_size=1)
        self.norm1 = nn.BatchNorm3d(in_channels // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=3,
                                          stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm3d(in_channels // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(in_channels // 2, n_filters, kernel_size=1)
        self.norm3 = nn.BatchNorm3d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)
        self.expansion = 1

        self.deconv_h = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=3,
                                           stride=2, padding=1, output_padding=1)
        self.deconv_x = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=3,
                                           stride=2, padding=1, output_padding=1)

        self.se = None
        if SE:#3D通道注意力机制
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(n_filters * self.expansion, n_filters * self.expansion // reduction, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(n_filters * self.expansion // reduction, n_filters * self.expansion, kernel_size=1),
                nn.Sigmoid()
            )

        self.eca = None
        if ECA_size is not None: #3D高效通道注意力
            self.eca = nn.Sequential(
                nn.Conv3d(n_filters * self.expansion, n_filters * self.expansion, kernel_size=(1, ECA_size, ECA_size),
                          padding=(0, ECA_size // 2, ECA_size // 2), groups=n_filters * self.expansion),
                nn.Sigmoid()
            )

        self.conv_out = nn.Conv3d(n_filters, in_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)

    def forward(self, x, h):
        identity = x
        # 确保输入通道数正确
        x = torch.cat((x, h), dim=1)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.deconv(out)#上采样
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.se is not None:#通道注意力
            se_out = self.se(out)
            out = out+out * se_out

        if self.eca is not None:
            eca_out = self.eca(out)
            out = out+out * eca_out
        y_out = self.conv_out(out)#??

        identity = self.deconv_x(identity)#上采样
        # identity=identity+y_out

        h = self.deconv_h(h)#上采样
        # h = h + y_out

        return identity, h

if __name__ == "__main__":

    network = DBSNet0(
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
