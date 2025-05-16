import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
import torch.nn.functional as F


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', ignore_index=None):
        """
        Focal Loss for multi-class classification.
        :param alpha: Weighting factor for class imbalance.
        :param gamma: Focusing parameter to reduce the loss contribution from easy examples.
        :param reduction: Reduction method for the loss ('mean', 'sum', or 'none').
        :param ignore_index: Specifies a target value that is ignored and does not contribute to the loss.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (batch_size, num_classes, depth, height, width) for 3D or (batch_size, num_classes, height, width) for 2D
        # targets: (batch_size, depth, height, width) for 3D or (batch_size, height, width) for 2D

        # Compute log softmax
        log_pt = F.log_softmax(inputs, dim=1)  # (batch_size, num_classes, depth, height, width) or (batch_size, num_classes, height, width)

        # Convert targets to int64 for gather
        targets = targets.long()

        # Reshape log_pt and targets for gather
        if log_pt.dim() == 5:  # 3D data
            log_pt = log_pt.permute(0, 2, 3, 4, 1).contiguous()  # (batch_size, depth, height, width, num_classes)
            log_pt = log_pt.view(-1, log_pt.size(-1))  # (batch_size * depth * height * width, num_classes)
            targets = targets.view(-1, 1)  # (batch_size * depth * height * width, 1)
        elif log_pt.dim() == 4:  # 2D data
            log_pt = log_pt.permute(0, 2, 3, 1).contiguous()  # (batch_size, height, width, num_classes)
            log_pt = log_pt.view(-1, log_pt.size(-1))  # (batch_size * height * width, num_classes)
            targets = targets.view(-1, 1)  # (batch_size * height * width, 1)
        else:
            raise ValueError(f"Unsupported input dimension: {log_pt.dim()}")

        # Gather the log probabilities for the target classes
        log_pt = log_pt.gather(1, targets)  # (batch_size * depth * height * width, 1) or (batch_size * height * width, 1)
        log_pt = log_pt.view(-1)  # (batch_size * depth * height * width) or (batch_size * height * width)

        # Compute pt
        pt = torch.exp(log_pt)

        # Compute Focal Loss
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * log_pt

        # Handle ignore_index
        if self.ignore_index is not None:
            mask = targets.view(-1) != self.ignore_index
            focal_loss = focal_loss[mask]

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DC_and_CE_and_Focal_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, focal_kwargs=None, weight_ce=1, weight_dice=1, weight_focal=1,
                 ignore_label=None, dice_class=SoftDiceLoss):
        """
        Combined loss function: Dice Loss + Cross-Entropy Loss + Focal Loss.

        :param soft_dice_kwargs: Arguments for Dice Loss.
        :param ce_kwargs: Arguments for Cross-Entropy Loss.
        :param focal_kwargs: Arguments for Focal Loss (alpha, gamma, reduction).
        :param weight_ce: Weight for Cross-Entropy Loss.
        :param weight_dice: Weight for Dice Loss.
        :param weight_focal: Weight for Focal Loss.
        :param ignore_label: Specifies a target value that is ignored and does not contribute to the loss.
        :param dice_class: Class for Dice Loss implementation.
        """
        super(DC_and_CE_and_Focal_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_focal = weight_focal
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

        # Initialize Focal Loss
        if focal_kwargs is None:
            focal_kwargs = {'alpha': 0.25, 'gamma': 2.0, 'reduction': 'mean', 'ignore_index': ignore_label}
        self.focal = FocalLoss(**focal_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_and_Focal_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        # Compute Dice Loss
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0

        # Compute Cross-Entropy Loss
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        # Compute Focal Loss
        focal_loss = self.focal(net_output, target[:, 0]) \
            if self.weight_focal != 0 and (self.ignore_label is None or num_fg > 0) else 0

        # Combine losses
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_focal * focal_loss
        return result

class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            target_regions = torch.clone(target[:, :-1])
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
