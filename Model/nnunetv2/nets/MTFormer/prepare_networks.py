import torch
import torch.nn as nn

# from swim_transformer2 import SwinTransformer2
# from swim_transformer2_CA_nyud import SwinTransformer3
from .swim_transformer2 import SwinTransformer2
from .swim_transformer2_CA_nyud import SwinTransformer3
import math
import torch.nn.functional as F

class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """
        Draw N samples from multinomial
        :param N: number of samples
        :return: samples
        """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj

class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = l.size(0)
        outputSize = self.memory_l.size(0)
        inputSize = self.memory_l.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        # sample
        weight_l = torch.index_select(self.memory_l, 0, idx.to(self.memory_l.device).view(-1)).detach()
        weight_l = weight_l.view(batchSize, K + 1, inputSize)
        out_ab = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))
        # sample
        weight_ab = torch.index_select(self.memory_ab, 0, idx.to(self.memory_ab.device).view(-1)).detach()
        weight_ab = weight_ab.view(batchSize, K + 1, inputSize)
        out_l = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out_ab = torch.div(out_ab, T)
            out_l = torch.div(out_l, T)
            out_l = out_l.contiguous()
            out_ab = out_ab.contiguous()
        else:
            out_ab = torch.exp(torch.div(out_ab, T))
            out_l = torch.exp(torch.div(out_l, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_l < 0:
                self.params[2] = out_l.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))
            if Z_ab < 0:
                self.params[3] = out_ab.mean() * outputSize
                Z_ab = self.params[3].clone().detach().item()
                print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))
            # compute out_l, out_ab
            out_l = torch.div(out_l, Z_l).contiguous()
            out_ab = torch.div(out_ab, Z_ab).contiguous()

        # # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.to(self.memory_l.device).view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.to(self.memory_ab.device).view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

        return out_l, out_ab

class NCECriterion(nn.Module):
    """
    Eq. (12): L_{NCE}
    """
    def __init__(self, n_data):
        super(NCECriterion, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        eps = 1e-7
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class ProjectionHead(nn.Module):
    def __init__(self, input_channels=256, output_channels=4):
        super().__init__()
        self.project = nn.Sequential(
            # Block 1: 28x28 → 56x56
            nn.Sequential(
                nn.Conv2d(input_channels, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
            ),
            # Block 2: 56x56 → 112x112
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
            ),
            # Block 3: 112x112 → 224x224
            nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
            ),
            # Block 4: 224x224 → 224x224 (仅调整通道)
            nn.Sequential(
                nn.Conv2d(32, output_channels, kernel_size=1, bias=False),
            )
        )

    def forward(self, x):
        return self.project(x)

class MTFormer(nn.Module):
    def __init__(self, input_backbone_channels,output_channel):
        super(MTFormer, self).__init__()
        self.backbone = SwinTransformer2(pretrain_img_size=224, in_chans=input_backbone_channels,window_size=7, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24),
                                   embed_dim=96, drop_path_rate=0.3)

        backbone_channels = 1344#224*6
        backbone_channels_reduce = 256
        self.backbone_channels_reduce = backbone_channels_reduce
        in_chans = backbone_channels_reduce
        embed_dim = backbone_channels_reduce

        self.decoder = nn.Conv2d(backbone_channels, backbone_channels_reduce, 1)

        self.transformer1 = SwinTransformer2(pretrain_img_size=224, window_size=7, depths=(2,), num_heads=(4,),
                                             in_chans=in_chans, embed_dim=embed_dim, drop_path_rate=0.0,
                                             out_indices=(0,))

        self.transformer4 = SwinTransformer2(pretrain_img_size=224, window_size=7, depths=(2,), num_heads=(4,),
                                             in_chans=in_chans, embed_dim=embed_dim, drop_path_rate=0.0,
                                             out_indices=(0,))
        self.transformer5 = SwinTransformer2(pretrain_img_size=224, window_size=7, depths=(2,), num_heads=(4,),
                                             in_chans=in_chans, embed_dim=embed_dim, drop_path_rate=0.0,
                                             out_indices=(0,))
        self.transformer6 = SwinTransformer3(pretrain_img_size=224, window_size=7, depths=(2,), num_heads=(4,),
                                             in_chans=in_chans * 2, embed_dim=embed_dim * 2, drop_path_rate=0.0,
                                             out_indices=(0,))
        self.transformer7 = SwinTransformer3(pretrain_img_size=224, window_size=7, depths=(2,), num_heads=(4,),
                                             in_chans=in_chans * 2, embed_dim=embed_dim * 2, drop_path_rate=0.0,
                                             out_indices=(0,))

        self.output_channel1 = output_channel
        self.output_channel2 = output_channel

        self.project1 = ProjectionHead(input_channels=256, output_channels=self.output_channel1)
        self.project2=ProjectionHead(input_channels=256, output_channels=self.output_channel2)
    def forward(self, x):
        out_size = x.size()[2:]
        shared_representation = self.backbone(x)
        shared_representation = self.decoder(shared_representation)
        shared_representation = self.transformer1.forward2(shared_representation)

        feature_T_task1 = self.transformer4.forward2(shared_representation)
        feature_T_task2 = self.transformer5.forward2(shared_representation)

        feature_T1 = torch.cat([feature_T_task1, feature_T_task2], dim=1)
        feature_T1 = self.transformer6.forward2(feature_T1)
        feature_T_task1_new = feature_T1[:, 0:self.backbone_channels_reduce, :, :]

        feature_T2 = torch.cat([feature_T_task2, feature_T_task1], dim=1)
        feature_T2 = self.transformer7.forward2(feature_T2)
        feature_T_task2_new = feature_T2[:, 0:self.backbone_channels_reduce, :, :]

        output_task1 = self.project1(feature_T_task1_new)
        output_task2 = self.project2(feature_T_task2_new)

        return output_task1,output_task2

if __name__ == '__main__':
    model = MTFormer(1,4)
    x = torch.randn(2, 1, 224, 224)
    outputs = model(x)
    a=1