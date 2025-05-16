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

#DBSNet3=DBSNet4+注意力
class Attention_DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, SE=False, ECA_size=5, reduction=16):
        super(Attention_DecoderBlock, self).__init__()
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

        self.se = True
        if SE:#3D通道注意力机制
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(n_filters * self.expansion, n_filters * self.expansion // reduction, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(n_filters * self.expansion // reduction, n_filters * self.expansion, kernel_size=1),
                nn.Sigmoid()
            )

        self.eca = True
        if ECA_size is not None: #3D高效通道注意力
            self.eca = nn.Sequential(
                nn.Conv3d(n_filters * self.expansion, n_filters * self.expansion, kernel_size=(1, ECA_size, ECA_size),
                          padding=(0, ECA_size // 2, ECA_size // 2), groups=n_filters * self.expansion),
                nn.Sigmoid()
            )

        self.conv_out = nn.Conv3d(n_filters, in_channels// 2, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)

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
        y_out = self.conv_out(out)#??project

        identity = self.deconv_x(identity)#上采样
        identity=identity+y_out

        h = self.deconv_h(h)#上采样
        h = h + y_out

        return identity, h

class DBSNet3(nn.Module):
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

        # self.up_3=Attention_DecoderBlock(in_channels=16 * n_channels, n_filters=64,
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
        self.up_2 = Attention_DecoderBlock(in_channels=8 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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

        self.up_1=Attention_DecoderBlock(in_channels=4 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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

        self.up_0=Attention_DecoderBlock(in_channels=2 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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

#Shared_attention_DecoderBlock
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
        identity = self.deconv_x(identity)#上采样
        # identity=identity+y_out

        h = self.deconv_h(h)#上采样
        # h = h + y_out

        return identity, h

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

#通道拼接
class DBSNet44(nn.Module):
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

        self.up_1=Shared_attention_DecoderBlock(in_channels=4 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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

        self.up_0=Shared_attention_DecoderBlock(in_channels=2 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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

        self.dec_block_11 = nn.Sequential(*[
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

        self.dec_block_00 = nn.Sequential(*[
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

            # x_up_22 = checkpoint.checkpoint(self.up_22, x2, self.dummy_tensor)
            dec_x2 = x_res_2 + x_up_22
            dec_x = torch.concatenate([dec_x, dec_x2], dim=1)
            x1 = self.iterative_checkpoint(self.dec_block_2, dec_x)
            x2 = self.iterative_checkpoint(self.dec_block_22, dec_x)

            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x1, self.dummy_tensor)

            if self.do_ds:
                x_ds_22 = checkpoint.checkpoint(self.out_22, x2, self.dummy_tensor)
            del x_res_2, x_up_2, x_up_22
            x_up_1, x_up_11 = self.up_1(x1, x2)
            dec_x = x_res_1 + x_up_1
            dec_x2 = x_res_1 + x_up_11
            dec_x = torch.concatenate([dec_x, dec_x2], dim=1)
            x1 = self.iterative_checkpoint(self.dec_block_1, dec_x)
            x2 = self.iterative_checkpoint(self.dec_block_11, dec_x)

            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x1, self.dummy_tensor)
            if self.do_ds:
                x_ds_11 = checkpoint.checkpoint(self.out_11, x2, self.dummy_tensor)
            del x_res_1, x_up_1, x_up_11
            x_up_0, x_up_00 = self.up_0(x1, x2)
            dec_x = x_res_0 + x_up_0
            dec_x2 = x_res_0 + x_up_00
            dec_x = torch.concatenate([dec_x, dec_x2], dim=1)
            x1 = self.iterative_checkpoint(self.dec_block_0, dec_x)
            x2 = self.iterative_checkpoint(self.dec_block_00, dec_x)
            del x_res_0, x_up_0, dec_x, x_up_00

            x1 = checkpoint.checkpoint(self.out_0, x1, self.dummy_tensor)
            x2 = checkpoint.checkpoint(self.out_00, x2, self.dummy_tensor)
        if self.do_ds:
            return [x1, x_ds_1, x_ds_2, x_ds_3], [x2, x_ds_11, x_ds_22, x_ds_33]
        else:
            return x1, x2

######5###### 上采样拼接####################
#group归一化,和55非常相似，所以这个没有训练
class UpBlock(nn.Module):
    """带交互的解码器块"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        # 上采样层
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(out_channels),
            # nn.GroupNorm(num_groups=out_channels, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )
        # 双路径处理
        self.double_path = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            # nn.GroupNorm(num_groups=out_channels, num_channels=out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv3d(out_channels, out_channels, 3, padding=1),
            # nn.GroupNorm(num_groups=out_channels,num_channels=out_channels),
            # nn.LeakyReLU(0.2)
        )

    def forward(self, xx, skip=None):
        x, skip = xx[0], xx[1]
        x1 = self.up(x)
        # 如果有跳跃连接则进行拼接
        if skip is not None:
            x = torch.cat([x1, skip], dim=1)
        # 双分支交互处理
        x = self.double_path(x)
        return x,x1

class DynamicFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 轻量级动态融合
        self.fuse = nn.Sequential(
            nn.Conv3d(channels*2, channels, 3, padding=1, groups=4),  # 分组卷积降参
            nn.GroupNorm(num_groups=channels, num_channels=channels),
            nn.GELU()  # 平滑激活
        )
        # 动态权重参数
        self.alpha = nn.Parameter(torch.tensor([0.7]))  # 初始偏重原始特征

    def forward(self, x1, x2):
        fused = self.fuse(torch.cat([x1, x2], dim=1))
        return self.alpha * fused + (1 - self.alpha) * x1  # 自适应加权

class DuInteraction(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch_fusion = DynamicFusion(channels)
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(num_groups=channels,num_channels=channels),
            # nn.LeakyReLU(0.2)
        )

    def forward(self, xx):
        x1, x2=xx[0],xx[1]
        # 双向特征融合
        fused1 = self.branch_fusion(x1, x2)
        fused2 = self.branch_fusion(x2, x1)
        # 特征增强
        return self.conv(fused1), self.conv(fused2)

class CDecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, SE=False, ECA_size=5, reduction=16):
        super(CDecoderBlock, self).__init__()
        self.branch1_dec1 = UpBlock(in_channels, in_channels// 2 , in_channels // 2)
        self.branch2_dec1 = UpBlock(in_channels, in_channels// 2 , in_channels // 2)

        self.conv1=nn.Sequential(nn.Conv3d(in_channels*2, in_channels // 2, kernel_size=1),
                                 nn.BatchNorm3d(in_channels // 2),
                                 nn.ReLU(inplace=True))

        # self.deconv1 = nn.Sequential(nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=3,
        #                                   stride=2, padding=1, output_padding=1),
        #                             nn.BatchNorm3d(in_channels // 2),
        #                             nn.ReLU(inplace=True))
        # self.deconv2 = nn.Sequential(nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=3,
        #                                                 stride=2, padding=1, output_padding=1),
        #                              nn.BatchNorm3d(in_channels // 2),
        #                              nn.ReLU(inplace=True))

        self.conv31 = nn.Sequential(nn.Conv3d(in_channels // 2, n_filters, kernel_size=1),
                                   nn.BatchNorm3d(n_filters))
        self.conv32 = nn.Sequential(nn.Conv3d(in_channels // 2, n_filters, kernel_size=1),
                                    nn.BatchNorm3d(n_filters))
        self.expansion = 1
        self.attention = DuInteraction(n_filters * self.expansion)

        self.conv_out1= nn.Sequential(nn.Conv3d(n_filters, in_channels// 2, kernel_size=1),
                                    nn.GroupNorm(num_groups=in_channels// 2,num_channels=in_channels// 2))
        self.conv_out2= nn.Sequential(nn.Conv3d(n_filters, in_channels// 2, kernel_size=1),
                                    nn.GroupNorm(num_groups=in_channels// 2,num_channels=in_channels// 2))

        self.dropout = nn.Dropout3d(0.2)

    def forward(self, xx):
        x1, x2, S1, S2=xx[0],xx[1],xx[2],xx[3]
        # x1,out1 = checkpoint.checkpoint(self.branch1_dec1, [x1, S1])
        # x2,out2 = checkpoint.checkpoint(self.branch2_dec1, [x2, S2])
        x1,out1=self.branch1_dec1([x1, S1])
        x2, out2 = self.branch2_dec1([x2, S2])
        out1=self.conv31(out1)
        out2 = self.conv32(out2)
        out1,out2=checkpoint.checkpoint(self.attention,[out1,out2])
        x1 =x1+self.conv_out1(out2)
        x2=x2+self.conv_out2(out1)
        # x1 = x1 + out1
        # x2 = x2 + out2
        return x1, x2

class DBSNet5(nn.Module):
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

        # self.up_3=CDecoderBlock(in_channels=16 * n_channels, n_filters=64,
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
        self.up_2 = CDecoderBlock(in_channels=8 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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

        self.up_1=CDecoderBlock(in_channels=4 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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

        self.up_0=CDecoderBlock(in_channels=2 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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
            dec_x1,dec_x2=self.up_2([x1,x2,x_res_2,x_res_2])
            # dec_x1,dec_x2 = checkpoint.checkpoint(self.up_2, [x1,x2,x_res_2,x_res_2])
            x1 = self.iterative_checkpoint(self.dec_block_2, dec_x1)
            # x_up_22 = checkpoint.checkpoint(self.up_22, x2, self.dummy_tensor)
            x2 = self.iterative_checkpoint(self.dec_block_22, dec_x2)

            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x1, self.dummy_tensor)

            if self.do_ds:
                x_ds_22 = checkpoint.checkpoint(self.out_22, x2, self.dummy_tensor)
            del x_res_2
            dec_x1, dec_x2 = self.up_1([x1, x2, x_res_1, x_res_1])
            # dec_x1,dec_x2 = checkpoint.checkpoint(self.up_1, [x1, x2, x_res_1, x_res_1])
            x1 = self.iterative_checkpoint(self.dec_block_1, dec_x1)
            x2 = self.iterative_checkpoint(self.dec_block_11, dec_x2)

            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x1, self.dummy_tensor)
            if self.do_ds:
                x_ds_11 = checkpoint.checkpoint(self.out_11, x2, self.dummy_tensor)
            del x_res_1
            dec_x1, dec_x2 = self.up_0([x1, x2, x_res_0, x_res_0])
            # dec_x1, dec_x2 = checkpoint.checkpoint(self.up_0, [x1, x2, x_res_0, x_res_0])
            x1 = self.iterative_checkpoint(self.dec_block_0, dec_x1)
            x2 = self.iterative_checkpoint(self.dec_block_00, dec_x2)
            del x_res_0, dec_x1, dec_x2,

            x1 = checkpoint.checkpoint(self.out_0, x1, self.dummy_tensor)
            x2 = checkpoint.checkpoint(self.out_00, x2, self.dummy_tensor)

        if self.do_ds:
            return [x1, x_ds_1, x_ds_2, x_ds_3], [x2, x_ds_11, x_ds_22, x_ds_33]
        else:
            return x1, x2

##55=CBAM+跳转拼接
class DDecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, SE=False, ECA_size=5, reduction=16):
        super(DDecoderBlock, self).__init__()
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

        self.se = True
        if SE:#3D通道注意力机制
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(n_filters * self.expansion, n_filters * self.expansion // reduction, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(n_filters * self.expansion // reduction, n_filters * self.expansion, kernel_size=1),
                nn.Sigmoid()
            )

        self.eca = True
        if ECA_size is not None: #3D高效通道注意力
            self.eca = nn.Sequential(
                nn.Conv3d(n_filters * self.expansion, n_filters * self.expansion, kernel_size=(1, ECA_size, ECA_size),
                          padding=(0, ECA_size // 2, ECA_size // 2), groups=n_filters * self.expansion),
                nn.Sigmoid()
            )

        self.conv_out = nn.Conv3d(n_filters, in_channels// 2, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)

        self.branch1_dec1 = ConcatenDecoderBlock(in_channels, in_channels// 2 , in_channels // 2)
        self.branch2_dec1 = ConcatenDecoderBlock(in_channels, in_channels// 2 , in_channels // 2)
        self.dropout = nn.Dropout3d(0.2)

    def forward(self, xx):
        x1, x2, S1, S2=xx[0],xx[1],xx[2],xx[3]
        identity = x1
        # 确保输入通道数正确
        x = torch.cat((x1, x2), dim=1)
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
        y_out = self.conv_out(out)#??project

        x1 = self.dropout(self.branch1_dec1(x1, S1))+y_out
        x2 = self.dropout(self.branch2_dec1(x2, S2))+y_out
        return x1, x2

class DBSNet55(nn.Module):
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

        # self.up_3=DDecoderBlock(in_channels=16 * n_channels, n_filters=64,
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
        self.up_2 = DDecoderBlock(in_channels=8 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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

        self.up_1=DDecoderBlock(in_channels=4 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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

        self.up_0=DDecoderBlock(in_channels=2 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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
            dec_x1,dec_x2=self.up_2([x1,x2,x_res_2,x_res_2])
            # dec_x1,dec_x2 = checkpoint.checkpoint(self.up_2, [x1,x2,x_res_2,x_res_2])
            x1 = self.iterative_checkpoint(self.dec_block_2, dec_x1)
            # x_up_22 = checkpoint.checkpoint(self.up_22, x2, self.dummy_tensor)
            x2 = self.iterative_checkpoint(self.dec_block_22, dec_x2)

            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x1, self.dummy_tensor)

            if self.do_ds:
                x_ds_22 = checkpoint.checkpoint(self.out_22, x2, self.dummy_tensor)
            del x_res_2
            dec_x1, dec_x2 = self.up_1([x1, x2, x_res_1, x_res_1])
            # dec_x1,dec_x2 = checkpoint.checkpoint(self.up_1, [x1, x2, x_res_1, x_res_1])
            x1 = self.iterative_checkpoint(self.dec_block_1, dec_x1)
            x2 = self.iterative_checkpoint(self.dec_block_11, dec_x2)

            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x1, self.dummy_tensor)
            if self.do_ds:
                x_ds_11 = checkpoint.checkpoint(self.out_11, x2, self.dummy_tensor)
            del x_res_1
            dec_x1, dec_x2 = self.up_0([x1, x2, x_res_0, x_res_0])
            # dec_x1, dec_x2 = checkpoint.checkpoint(self.up_0, [x1, x2, x_res_0, x_res_0])
            x1 = self.iterative_checkpoint(self.dec_block_0, dec_x1)
            x2 = self.iterative_checkpoint(self.dec_block_00, dec_x2)
            del x_res_0, dec_x1, dec_x2,

            x1 = checkpoint.checkpoint(self.out_0, x1, self.dummy_tensor)
            x2 = checkpoint.checkpoint(self.out_00, x2, self.dummy_tensor)

        if self.do_ds:
            return [x1, x_ds_1, x_ds_2, x_ds_3], [x2, x_ds_11, x_ds_22, x_ds_33]
        else:
            return x1, x2

####555=原始归一化+跳转拼接+DynStaF注意力
# #EDecoderBlock11效果不好丢弃

#加权通道拼接
class EDecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, SE=True, ECA_size=5, reduction=16):
        super(EDecoderBlock, self).__init__()
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
        self.conva=nn.Sequential(nn.Conv3d(in_channels, in_channels // 2, kernel_size=1),
                                 nn.BatchNorm3d(in_channels // 2),
                                 nn.ReLU(inplace=True))

        self.deconv_x = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=3,
                                           stride=2, padding=1, output_padding=1)

        self.alpha = nn.Parameter(torch.tensor(0.4))  # 初始偏重原始特征
        self.beta = nn.Parameter(torch.tensor(0.8))

        self.se = True
        if SE:#3D通道注意力机制
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(n_filters * self.expansion, n_filters * self.expansion // reduction, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(n_filters * self.expansion // reduction, n_filters * self.expansion, kernel_size=1),
                nn.Sigmoid()
            )

        self.eca = True
        if ECA_size is not None: #3D高效通道注意力
            self.eca = nn.Sequential(
                nn.Conv3d(n_filters * self.expansion, n_filters * self.expansion, kernel_size=(1, ECA_size, ECA_size),
                          padding=(0, ECA_size // 2, ECA_size // 2), groups=n_filters * self.expansion),
                nn.Sigmoid()
            )

        self.conv_out = nn.Conv3d(n_filters, in_channels// 2, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)

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
        y_out = self.conv_out(out)#??project

        identity = self.deconv_x(identity)#上采样
        identity=self.conva(torch.cat((identity, self.alpha *y_out), dim=1))
        # identity=identity+y_out*self.alpha

        h = self.deconv_h(h)#上采样
        h = h + y_out*self.beta

        return identity, h

class DBSNet555(nn.Module):
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

        # self.up_3=EDecoderBlock(in_channels=16 * n_channels, n_filters=64,
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
        self.up_2 = EDecoderBlock(in_channels=8 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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

        self.up_1=EDecoderBlock(in_channels=4 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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

        self.up_0=EDecoderBlock(in_channels=2 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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

####加权相加
class FDecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, SE=True, ECA_size=5, reduction=16):
        super(FDecoderBlock, self).__init__()
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
        # self.conva=nn.Sequential(nn.Conv3d(in_channels, in_channels // 2, kernel_size=1),
        #                          nn.BatchNorm3d(in_channels // 2),
        #                          nn.ReLU(inplace=True))

        self.deconv_x = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=3,
                                           stride=2, padding=1, output_padding=1)

        self.alpha = nn.Parameter(torch.tensor(0.4))  # 初始偏重原始特征
        self.beta = nn.Parameter(torch.tensor(0.8))

        self.se = True
        if SE:#3D通道注意力机制
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(n_filters * self.expansion, n_filters * self.expansion // reduction, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(n_filters * self.expansion // reduction, n_filters * self.expansion, kernel_size=1),
                nn.Sigmoid()
            )

        self.eca = True
        if ECA_size is not None: #3D高效通道注意力
            self.eca = nn.Sequential(
                nn.Conv3d(n_filters * self.expansion, n_filters * self.expansion, kernel_size=(1, ECA_size, ECA_size),
                          padding=(0, ECA_size // 2, ECA_size // 2), groups=n_filters * self.expansion),
                nn.Sigmoid()
            )

        self.conv_out = nn.Conv3d(n_filters, in_channels// 2, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)

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
        y_out = self.conv_out(out)#??project

        identity = self.deconv_x(identity)#上采样
        # identity=self.conva(torch.cat((identity, self.alpha *y_out), dim=1))
        identity=identity+y_out*self.alpha

        h = self.deconv_h(h)#上采样
        h = h + y_out*self.beta

        return identity, h

class DBSNetF(nn.Module):
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

        # self.up_3=FDecoderBlock(in_channels=16 * n_channels, n_filters=64,
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
        self.up_2 = FDecoderBlock(in_channels=8 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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

        self.up_1=FDecoderBlock(in_channels=4 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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

        self.up_0=FDecoderBlock(in_channels=2 * n_channels, n_filters=64,SE=True, ECA_size=5, reduction=16)

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

#####6###### #简单上采样+交叉注意力
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

class Cross_attention_DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(Cross_attention_DecoderBlock, self).__init__()
        self.cross_attention = DualTargetCrossAttention(in_channels=in_channels)
        self.deconv_h = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=3,
                                           stride=2, padding=1, output_padding=1)
        self.deconv_x = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=3,
                                           stride=2, padding=1, output_padding=1)

    def forward(self, x, h):
        # c=x.shape[1]
        x1, x2 = self.cross_attention(x, h)
        x1 = self.deconv_x(x1)#上采样
        x2 = self.deconv_h(x2)#上采样

        return x1, x2

class DBSNet6(nn.Module):
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

        # self.up_3=Cross_attention_DecoderBlock(in_channels=16 * n_channels, n_filters=64)
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
        self.up_2 = Cross_attention_DecoderBlock(in_channels=8 * n_channels, n_filters=64)

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

        self.up_1=Cross_attention_DecoderBlock(in_channels=4 * n_channels, n_filters=64)

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

        self.up_0=Cross_attention_DecoderBlock(in_channels=2 * n_channels, n_filters=64)

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

#######7###### DynStaF注意力

class EnhancedDynStaFNet(nn.Module):
    """增强版动态融合网络"""
    def __init__(self, in_channels, n_filters):
        super().__init__()
        # 分支1解码器
        self.branch1_dec1 = ConcatenDecoderBlock(in_channels, in_channels// 2 , in_channels // 2)
        self.branch2_dec1 = ConcatenDecoderBlock(in_channels, in_channels// 2 , in_channels // 2)
        # 解码器配置
        self.cross_attention = DuInteraction(in_channels // 2)
        # 带可学习权重的残差
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始权重
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.res_path1 = nn.Sequential(
            nn.Conv3d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.GroupNorm(num_groups=in_channels // 2,num_channels=in_channels // 2),
            nn.LeakyReLU(0.2)
        )
        self.res_path2 = nn.Sequential(
            nn.Conv3d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.GroupNorm(num_groups=in_channels // 2,num_channels=in_channels // 2),
            nn.LeakyReLU(0.2)
        )

    def forward(self, xx):
        x1, x2, S1, S2 = xx[0], xx[1], xx[2], xx[3]
        # c=x.shape[1]
        x1 = self.branch1_dec1(x1, S1)
        x2 = self.branch2_dec1(x2, S2)
        x1, x2 = self.cross_attention(x1, x2)
        x1 = x1 + self.alpha * self.res_path1(x1)  # 可学习权重调节
        x2 = x2 + self.beta * self.res_path2(x2)
        return x1, x2

class ConcatenDecoderBlock(nn.Module):
    """带交互的解码器块"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        # 上采样层
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.GroupNorm(num_groups=out_channels,num_channels=out_channels),
            nn.LeakyReLU(0.2)
        )

        # 双路径处理
        self.double_path = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=out_channels,num_channels=out_channels),
            nn.LeakyReLU(0.2),
            # nn.Conv3d(out_channels, out_channels, 3, padding=1),
            # nn.GroupNorm(num_groups=out_channels,num_channels=out_channels),
            # nn.LeakyReLU(0.2)
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        # 如果有跳跃连接则进行拼接
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        # 双分支交互处理
        x = self.double_path(x)
        return x

class EnhancedDynStaFNet(nn.Module):
    """增强版动态融合网络"""
    def __init__(self, in_channels, n_filters):
        super().__init__()
        # 分支1解码器
        self.branch1_dec1 = ConcatenDecoderBlock(in_channels, in_channels// 2 , in_channels // 2)
        self.branch2_dec1 = ConcatenDecoderBlock(in_channels, in_channels// 2 , in_channels // 2)
        # 解码器配置
        self.cross_attention = DuInteraction(in_channels // 2)
        # 带可学习权重的残差
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始权重
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.res_path1 = nn.Sequential(
            nn.Conv3d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.GroupNorm(num_groups=in_channels // 2,num_channels=in_channels // 2),
            nn.LeakyReLU(0.2)
        )
        self.res_path2 = nn.Sequential(
            nn.Conv3d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.GroupNorm(num_groups=in_channels // 2,num_channels=in_channels // 2),
            nn.LeakyReLU(0.2)
        )

    def forward(self, xx):
        x1, x2, S1, S2 = xx[0], xx[1], xx[2], xx[3]
        # c=x.shape[1]
        x1 = self.branch1_dec1(x1, S1)
        x2 = self.branch2_dec1(x2, S2)
        x1, x2 = self.cross_attention(x1, x2)
        x1 = x1 + self.alpha * self.res_path1(x1)  # 可学习权重调节
        x2 = x2 + self.beta * self.res_path2(x2)
        return x1, x2

class DBSNet7(nn.Module):
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

        # self.up_3=EnhancedDynStaFNet(in_channels=16 * n_channels, n_filters=64,
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
        self.up_2 = EnhancedDynStaFNet(in_channels=8 * n_channels, n_filters=64)

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

        self.up_1=EnhancedDynStaFNet(in_channels=4 * n_channels, n_filters=64)

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

        self.up_0=EnhancedDynStaFNet(in_channels=2 * n_channels, n_filters=64)

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
            # dec_x1,dec_x2=self.up_2(x1,x2,x_res_2,x_res_2)
            dec_x1,dec_x2 = checkpoint.checkpoint(self.up_2, [x1,x2,x_res_2,x_res_2])
            x1 = self.iterative_checkpoint(self.dec_block_2, dec_x1)
            # x_up_22 = checkpoint.checkpoint(self.up_22, x2, self.dummy_tensor)
            x2 = self.iterative_checkpoint(self.dec_block_22, dec_x2)

            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x1, self.dummy_tensor)

            if self.do_ds:
                x_ds_22 = checkpoint.checkpoint(self.out_22, x2, self.dummy_tensor)
            del x_res_2
            # dec_x1, dec_x2 = self.up_1(x1, x2, x_res_1, x_res_1)
            dec_x1, dec_x2 = checkpoint.checkpoint(self.up_1, [x1, x2, x_res_1, x_res_1])
            x1 = self.iterative_checkpoint(self.dec_block_1, dec_x1)
            x2 = self.iterative_checkpoint(self.dec_block_11, dec_x2)

            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x1, self.dummy_tensor)
            if self.do_ds:
                x_ds_11 = checkpoint.checkpoint(self.out_11, x2, self.dummy_tensor)
            del x_res_1
            # dec_x1, dec_x2 = self.up_0(x1, x2, x_res_0, x_res_0)
            dec_x1, dec_x2 = checkpoint.checkpoint(self.up_0, [x1, x2, x_res_0, x_res_0])
            x1 = self.iterative_checkpoint(self.dec_block_0, dec_x1)
            x2 = self.iterative_checkpoint(self.dec_block_00, dec_x2)
            del x_res_0, dec_x1, dec_x2,

            x1 = checkpoint.checkpoint(self.out_0, x1, self.dummy_tensor)
            x2 = checkpoint.checkpoint(self.out_00, x2, self.dummy_tensor)
        if self.do_ds:
            return [x1, x_ds_1, x_ds_2, x_ds_3], [x2, x_ds_11, x_ds_22, x_ds_33]
        else:
            return x1, x2

########8###### DBSNet8+DMF注意力
class CrossModalAttention3D11(nn.Module):
    """基于滑动窗口的3D跨模态交叉注意力模块（最终修正版）"""

    def __init__(self, channels, window_size=(8, 8, 8), num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = channels // num_heads

        # 合并QKV投影
        self.qkv_conv = nn.Conv3d(3 * channels, 3 * channels, 1)

        # 输出层
        self.out_conv = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.GroupNorm(num_groups=channels,num_channels=channels)
        )

    def _window_partition(self, x, window_size):
        B, C, D, H, W = x.shape
        x = x.view(B, C,
                   D // window_size[0], window_size[0],
                   H // window_size[1], window_size[1],
                   W // window_size[2], window_size[2])
        windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        return windows.view(-1, C, window_size[0], window_size[1], window_size[2])

    def _window_reverse(self, windows, window_size, D, H, W):
        B = int(windows.shape[0] / (D * H * W / (window_size[0] * window_size[1] * window_size[2])))
        x = windows.view(B,
                         D // window_size[0], H // window_size[1], W // window_size[2],
                         -1,
                         window_size[0], window_size[1], window_size[2])
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.view(B, -1, D, H, W)

    def forward(self, query, key, value):
        B, C, D, H, W = query.shape
        window_size = self.window_size

        # 自动填充
        pad_d = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_h = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_w = (window_size[2] - W % window_size[2]) % window_size[2]

        query = F.pad(query, (0, pad_w, 0, pad_h, 0, pad_d))
        key = F.pad(key, (0, pad_w, 0, pad_h, 0, pad_d))
        value = F.pad(value, (0, pad_w, 0, pad_h, 0, pad_d))

        # 合并投影
        qkv = self.qkv_conv(torch.cat([query, key, value], dim=1))
        q, k, v = torch.split(qkv, C, dim=1)

        # 窗口划分
        q_windows = self._window_partition(q, window_size)  # [B*N, C, wD, wH, wW]
        k_windows = self._window_partition(k, window_size)
        v_windows = self._window_partition(v, window_size)

        # 多头拆分（统一维度标记）
        q_windows = q_windows.view(-1, self.num_heads, self.head_dim,
                                   window_size[0], window_size[1], window_size[2])
        k_windows = k_windows.view(-1, self.num_heads, self.head_dim,
                                   window_size[0], window_size[1], window_size[2])
        v_windows = v_windows.view(-1, self.num_heads, self.head_dim,
                                   window_size[0], window_size[1], window_size[2])

        # 注意力计算（统一空间维度标记）
        attn = torch.einsum('bhdijk,bhdijl->bhijkl',
                          q_windows, k_windows) / (self.head_dim**0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhijkl,bhdijl->bhdijk', attn, v_windows)

        # 恢复形状
        out = out.contiguous().view(-1, C,
                                    window_size[0],
                                    window_size[1],
                                    window_size[2])
        out = self._window_reverse(out, window_size, D + pad_d, H + pad_h, W + pad_w)
        out = out[:, :, :D, :H, :W]

        return self.out_conv(out) + query[:, :, :D, :H, :W]

class ChannelGate(nn.Module):
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
        aa=(avg_out + max_out).view(B, C, 1, 1, 1)
        return  torch.sigmoid(aa)

class SpatialGate(nn.Module):
    """3D空间注意力（强制维度保持）"""
    def __init__(self, channel, kernel_size=3):
        super().__init__()
        kernel_size = kernel_size if kernel_size%2==1 else kernel_size+1
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size, padding=padding),
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.GELU()
        )

    def forward(self, x):
        B, _, D, H, W = x.shape
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(cat)
        assert out.shape[2:] == (D, H, W), f"空间维度不匹配 {out.shape} vs {(D, H, W)}"
        return torch.sigmoid(out)

class CrossModalAttention3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_att = ChannelGate(in_channels)
        self.spatial_att = SpatialGate(in_channels)

        # 可分离卷积
        self.dw_conv = nn.Sequential(
            nn.Conv3d(in_channels*2, in_channels, 3, padding=1, groups=in_channels),
            nn.GroupNorm(num_groups=in_channels, num_channels=in_channels),
            nn.GELU(),
            nn.Conv3d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.GroupNorm(num_groups=in_channels, num_channels=in_channels),
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始权重
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x1, x2):
        """
        query: 当前分支特征 [B,C,D,H,W]
        key_value: 另一分支特征 [B,C,D,H,W]
        """
        # 特征增强
        key_value=torch.cat((x1, x2), dim=1)
        out = self.dw_conv(key_value)
        # 双路注意力
        out = out + out * self.channel_att(out)*self.alpha
        out = out + out * self.spatial_att(out)*self.beta
        # 注意力融合
        return out  # 返回注意力加权的查询特征

class IterativeFusionBlock3D(nn.Module):
    """迭代式3D特征融合块"""
    def __init__(self, channels, iterations=2):
        super().__init__()
        self.iterations = iterations
        self.attention = CrossModalAttention3D(channels)
        self.covn = nn.Conv3d(channels, channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)

    def forward(self, x1, x2):
        for i in range(self.iterations):
            # 双向注意力交互
            attn = self.attention(x1, x2)
            attn=self.covn(attn)+attn
        return attn

class DMF_DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DMF_DecoderBlock, self).__init__()
        self.branch1_dec1 = ConcatenDecoderBlock(in_channels, in_channels// 2 , in_channels // 2)
        self.branch2_dec1 = ConcatenDecoderBlock(in_channels, in_channels// 2 , in_channels // 2)
        self.dropout = nn.Dropout3d(0.2)
        self.cross_attention = IterativeFusionBlock3D(in_channels // 2)

    def forward(self, xx):
        x1, x2, S1, S2=xx[0],xx[1],xx[2],xx[3]
        x1 = self.dropout(self.branch1_dec1(x1, S1))
        x2 = self.dropout(self.branch2_dec1(x2, S2))
        attn = self.cross_attention(x1, x2)
        # x1 = x1 +x11
        # x2 = x2 +x22
        x1 = x1 + attn  # 可学习权重调节
        x2 = x2 + attn
        return x1, x2

class DBSNet8(nn.Module):
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

        # self.up_3=DMF_DecoderBlock(in_channels=16 * n_channels, n_filters=64,
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
        self.up_2 = DMF_DecoderBlock(in_channels=8 * n_channels, n_filters=64)

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

        self.up_1=DMF_DecoderBlock(in_channels=4 * n_channels, n_filters=64)

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

        self.up_0=DMF_DecoderBlock(in_channels=2 * n_channels, n_filters=64)

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
            # dec_x1,dec_x2=self.up_2(x1,x2,x_res_2,x_res_2)
            dec_x1,dec_x2 = checkpoint.checkpoint(self.up_2, [x1,x2,x_res_2,x_res_2])
            x1 = self.iterative_checkpoint(self.dec_block_2, dec_x1)
            # x_up_22 = checkpoint.checkpoint(self.up_22, x2, self.dummy_tensor)
            x2 = self.iterative_checkpoint(self.dec_block_22, dec_x2)

            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x1, self.dummy_tensor)

            if self.do_ds:
                x_ds_22 = checkpoint.checkpoint(self.out_22, x2, self.dummy_tensor)
            del x_res_2
            # dec_x1, dec_x2 = self.up_1(x1, x2, x_res_1, x_res_1)
            dec_x1,dec_x2 = checkpoint.checkpoint(self.up_1, [x1, x2, x_res_1, x_res_1])
            x1 = self.iterative_checkpoint(self.dec_block_1, dec_x1)
            x2 = self.iterative_checkpoint(self.dec_block_11, dec_x2)

            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x1, self.dummy_tensor)
            if self.do_ds:
                x_ds_11 = checkpoint.checkpoint(self.out_11, x2, self.dummy_tensor)
            del x_res_1
            # dec_x1, dec_x2 = self.up_0(x1, x2, x_res_0, x_res_0)
            dec_x1, dec_x2 = checkpoint.checkpoint(self.up_0, [x1, x2, x_res_0, x_res_0])
            x1 = self.iterative_checkpoint(self.dec_block_0, dec_x1)
            x2 = self.iterative_checkpoint(self.dec_block_00, dec_x2)
            del x_res_0, dec_x1, dec_x2,

            x1 = checkpoint.checkpoint(self.out_0, x1, self.dummy_tensor)
            x2 = checkpoint.checkpoint(self.out_00, x2, self.dummy_tensor)
        if self.do_ds:
            return [x1, x_ds_1, x_ds_2, x_ds_3], [x2, x_ds_11, x_ds_22, x_ds_33]
        else:
            return x1, x2

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
