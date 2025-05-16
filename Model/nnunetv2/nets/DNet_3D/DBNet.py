import torch
import torch.nn as nn
from .blocks import *
import torch.utils.checkpoint as checkpoint

from .DNet_blocks import Encoder, Decoder, Bottleneck, Convblock, DFF

class DBNet(nn.Module):

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
        drop_path=0

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

        self.enc_block_1 = nn.Sequential(*[DLKMEDBlock(dim=n_channels*2, r=exp_r[1],drop_path=drop_path)
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

        self.enc_block_2 = nn.Sequential(*[DLKMEDBlock(dim=n_channels*4, r=exp_r[2],drop_path=drop_path)
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

        self.enc_block_3 = nn.Sequential(*[DLKMEDBlock(dim=n_channels*8, r=exp_r[3],drop_path=drop_path)
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

        self.bottleneck = nn.Sequential(*[DLKMEDBlock(dim=n_channels*16, r=exp_r[4],drop_path=drop_path)
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

        self.dec_block_3 = nn.Sequential(*[DLKMEDBlock(dim=n_channels*8, r=exp_r[5],drop_path=drop_path)
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

        self.dec_block_2 = nn.Sequential(*[DLKMEDBlock(dim=n_channels*4, r=exp_r[6],drop_path=drop_path)
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

        self.dec_block_1 = nn.Sequential(*[DLKMEDBlock(dim=n_channels*2, r=exp_r[7],drop_path=drop_path)
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
        self.dff3 = DFF(n_channels*8)
        self.dff2 = DFF(n_channels * 4)
        self.dff1 = DFF(n_channels * 2)
        self.dff0 = DFF(n_channels * 1)

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

        self.dec_block_33 = nn.Sequential(*[DLKMEDBlock(dim=n_channels*8, r=exp_r[5],drop_path=drop_path)
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

        self.dec_block_22 = nn.Sequential(*[DLKMEDBlock(dim=n_channels*4, r=exp_r[6],drop_path=drop_path)
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

        self.dec_block_11 = nn.Sequential(*[DLKMEDBlock(dim=n_channels*2, r=exp_r[7],drop_path=drop_path)
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

        self.dff33 = DFF(n_channels*8)
        self.dff22 = DFF(n_channels * 4)
        self.dff11 = DFF(n_channels * 2)
        self.dff00 = DFF(n_channels * 1)

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
            dec_x = self.dff3(x_up_3,x_res_3)
            x1 = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.do_ds:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x1, self.dummy_tensor)

            x_up_33 = checkpoint.checkpoint(self.up_33, x, self.dummy_tensor)
            dec_x = self.dff33(x_up_33,x_res_3)
            x2 = self.iterative_checkpoint(self.dec_block_33, dec_x)
            if self.do_ds:
                x_ds_33 = checkpoint.checkpoint(self.out_33, x2, self.dummy_tensor)

            del x_res_3, x_up_3,x_up_33

            x_up_2 = checkpoint.checkpoint(self.up_2, x1, self.dummy_tensor)
            dec_x = self.dff2(x_up_2,x_res_2)
            x1 = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x1, self.dummy_tensor)

            x_up_22 = checkpoint.checkpoint(self.up_22, x2, self.dummy_tensor)
            dec_x = self.dff22(x_up_22,x_res_2)#x_res_2 + x_up_22
            x2 = self.iterative_checkpoint(self.dec_block_22, dec_x)
            if self.do_ds:
                x_ds_22 = checkpoint.checkpoint(self.out_22, x2, self.dummy_tensor)
            del x_res_2, x_up_2,x_up_22

            x_up_1 = checkpoint.checkpoint(self.up_1, x1, self.dummy_tensor)
            dec_x =self.dff1(x_up_1,x_res_1)# x_res_1 + x_up_1
            x1 = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x1, self.dummy_tensor)

            x_up_11 = checkpoint.checkpoint(self.up_11, x2, self.dummy_tensor)
            dec_x = self.dff11(x_up_11,x_res_1)
            x2 = self.iterative_checkpoint(self.dec_block_11, dec_x)
            if self.do_ds:
                x_ds_11 = checkpoint.checkpoint(self.out_11, x2, self.dummy_tensor)
            del x_res_1, x_up_1,x_up_11

            x_up_0 = checkpoint.checkpoint(self.up_0, x1, self.dummy_tensor)
            dec_x = self.dff0(x_up_0,x_res_0)
            x1 = self.iterative_checkpoint(self.dec_block_0, dec_x)

            x_up_00 = checkpoint.checkpoint(self.up_00, x2, self.dummy_tensor)
            dec_x = self.dff00(x_up_00,x_res_0)
            x2 = self.iterative_checkpoint(self.dec_block_00, dec_x)
            del x_res_0, x_up_0, dec_x,x_up_00

            x1 = checkpoint.checkpoint(self.out_0, x1, self.dummy_tensor)
            x2 = checkpoint.checkpoint(self.out_00, x2, self.dummy_tensor)

        else:
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)
            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)
            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)
            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)

            x = self.bottleneck(x)
            if self.do_ds:
                x_ds_4 = self.out_4(x)

            x_up_3 = self.up_3(x)
            dec_x = x_res_3 + x_up_3
            x = self.dec_block_3(dec_x)

            if self.do_ds:
                x_ds_3 = self.out_3(x)
            del x_res_3, x_up_3

            x_up_2 = self.up_2(x)
            dec_x = x_res_2 + x_up_2
            x = self.dec_block_2(dec_x)
            if self.do_ds:
                x_ds_2 = self.out_2(x)
            del x_res_2, x_up_2

            x_up_1 = self.up_1(x)
            dec_x = x_res_1 + x_up_1
            x = self.dec_block_1(dec_x)
            if self.do_ds:
                x_ds_1 = self.out_1(x)
            del x_res_1, x_up_1

            x_up_0 = self.up_0(x)
            dec_x = x_res_0 + x_up_0
            x = self.dec_block_0(dec_x)
            del x_res_0, x_up_0, dec_x

            x = self.out_0(x)

        if self.do_ds:
            # print(f"output[0] grad_fn: {x1.grad_fn}")
            # print(f"output[0] grad_fn: {x2.grad_fn}")
            return [x1, x_ds_1, x_ds_2, x_ds_3],[x2, x_ds_11, x_ds_22, x_ds_33]
        else:
            return x1,x2

if __name__ == '__main__':
    data = torch.rand((2, 1, 128, 128, 128))
    
    model = DBNet(
            in_channels=1,
            out_channels=16,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            bottom_feat = 768,
            drop_path_rate=0
        )
    out=model(data)
    #from torchinfo import summary

    #summary(model, (1, 1, 128, 128, 128))

    #import hiddenlayer as hl
    #g = hl.build_graph(model, data,transforms=None)
    #g.save("network_architecture.pdf")
    #del g