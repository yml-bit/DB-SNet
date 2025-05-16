from nnunetv2.nets.MTSNet.MTSNet import DBSNet4 #原始
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from torch import autocast, nn
import torch
import numpy as np
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss,DC_and_CE_and_Focal_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
import copy

class nnUNetTrainerDSNet4(nnUNetTrainer): #nnUNetTrainer nnUNetTrainerNoDeepSupervision
    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device('cuda')
        ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # self.grad_scaler = None#默认是采用了，不然内存溢出
        self.initial_lr = 3e-4
        self.weight_decay = 1e-5
        self.num_epochs = 400#400 200
        self.current_epoch = 0

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        label_manager = plans_manager.get_label_manager(dataset_json)
        #D-Net与MedNeXt，效果没有获得提升
        model = DBSNet4(in_channels=num_input_channels, n_channels=32,  # 增大通道，减少卷积核l3
                        n_classes=label_manager.num_segmentation_heads, exp_r=[2,2,2,2,2,2,2,2,2],
                        kernel_size=3, deep_supervision=enable_deep_supervision, do_res=True,
                        do_res_up_down=True, block_counts=[2,2,2,2,2,2,2,2,2],  # True False
                        checkpoint_style='outside_block')

        # model = DBNet1(in_channels=1,out_channels=label_manager.num_segmentation_heads,
        #     depths=[2, 2, 2, 2],feat_size=[32, 64, 128, 256],bottom_feat=512,drop_path_rate=0)#[48, 96, 192, 384]
        return model

    def train_step(self, batch: dict) -> dict:
        # self.enable_deep_supervision = False  # False True
        data = batch['data']
        target_task1 = batch['target1']
        target_task2 = batch['target2']
        # target_task1 = batch['target1'][0]
        # target_task2 = batch['target2'][0]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target_task1, list):
            target_task1 = [i.to(self.device, non_blocking=True) for i in target_task1]
        else:
            target_task1 = target_task1.to(self.device, non_blocking=True)

        if isinstance(target_task2, list):
            target_task2 = [i.to(self.device, non_blocking=True) for i in target_task2]
        else:
            target_task2 = target_task2.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l =  0.5 *self.loss(output[0], target_task1) +  0.5 *self.loss(output[1], target_task2)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    # def configure_optimizers(self):
    #     optimizer = AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
    #     scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)
    #     self.print_to_log_file(f"Using optimizer {optimizer}")
    #     self.print_to_log_file(f"Using scheduler {scheduler}")
    #     return optimizer, scheduler

    def configure_optimizers(self):
        optimizer = AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=1.0)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")
        return optimizer, scheduler

    # def _build_loss(self):
    #     if self.label_manager.has_regions:
    #         loss = DC_and_BCE_loss({},
    #                                {'batch_dice': self.configuration_manager.batch_dice,
    #                                 'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
    #                                use_ignore_label=self.label_manager.ignore_label is not None,
    #                                dice_class=MemoryEfficientSoftDiceLoss)
    #     else:
    #         loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
    #                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
    #                               ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
    #         # loss=DC_and_CE_and_Focal_loss({'batch_dice': self.configuration_manager.batch_dice,
    #         #                        'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
    #         #                       ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
    #     # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
    #     # this gives higher resolution outputs more weight in the loss
    #
    #     return loss