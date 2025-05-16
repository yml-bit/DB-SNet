from nnunetv2.nets.MTSNet.MTSMed import MedNeXt2b#原始
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from torch import autocast, nn
import torch
import numpy as np

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
import copy

#128 160 160
class nnUNetTrainerMednext3(nnUNetTrainer):
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

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        # MedNeXt
        label_manager = plans_manager.get_label_manager(dataset_json)
        model = MedNeXt2b(in_channels=num_input_channels, n_channels=32,  # 增大通道，减少卷积核l3
                        n_classes=label_manager.num_segmentation_heads, exp_r=[2,3,4,4,4,4,4,3,2],
                        kernel_size=3, deep_supervision=enable_deep_supervision, do_res=True,
                        do_res_up_down=True, block_counts=[3,4,4,4,4,4,4,4,3],  # True False
                        checkpoint_style='outside_block')
        return model

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target_task1 = batch['target1']
        target_task2 = batch['target2']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target_task1, list):
            target_task1 = [i.to(self.device, non_blocking=True) for i in target_task1]
        else:
            target_task1 = target_task1.to(self.device, non_blocking=True)

        if isinstance(target_task2, list):
            target_task2 = [i.to(self.device, non_blocking=True) for i in target_task2]
        else:
            target_task2 = target_task2.to(self.device, non_blocking=True)

        optimizers, _ = self.configure_optimizers()
        optimizer_common, optimizer_branch1, optimizer_branch2 = optimizers
        # optimizer_common, optimizer_branch1, optimizer_branch2 = self.optimizer

        optimizer_common.zero_grad(set_to_none=True)
        optimizer_branch1.zero_grad(set_to_none=True)
        optimizer_branch2.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l1 = self.loss(output[0], target_task1)
            l2 = self.loss(output[1], target_task2)
            l = 0.6 * l1 + 0.4 * l2
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            # l = self.loss(output[0], target_task1)+self.loss(output[1], target_task2)
            l1 = self.loss(output[0], target_task1)
            l2 = self.loss(output[1], target_task2)
            l = 0.6 * l1 + 0.4 * l2

        if self.grad_scaler is not None:
            # 共同部分的梯度裁剪
            self.grad_scaler.scale(l).backward(retain_graph=True)
            self.grad_scaler.unscale_(optimizer_common)
            torch.nn.utils.clip_grad_norm_(optimizer_common.param_groups[0]['params'], 12)
            self.grad_scaler.step(optimizer_common)

            # 分支1的梯度裁剪
            self.grad_scaler.scale(l1).backward(retain_graph=True)
            self.grad_scaler.unscale_(optimizer_branch1)
            torch.nn.utils.clip_grad_norm_(optimizer_branch1.param_groups[0]['params'], 12)
            self.grad_scaler.step(optimizer_branch1)

            # 分支2的梯度裁剪
            self.grad_scaler.scale(l2).backward()
            self.grad_scaler.unscale_(optimizer_branch2)
            torch.nn.utils.clip_grad_norm_(optimizer_branch2.param_groups[0]['params'], 12)
            self.grad_scaler.step(optimizer_branch2)

            # 更新梯度缩放器
            self.grad_scaler.update()
        else:
            # 共同部分的梯度裁剪
            l.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(optimizer_common.param_groups[0]['params'], 12)
            optimizer_common.step()

            # 分支1的梯度裁剪
            l1.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(optimizer_branch1.param_groups[0]['params'], 12)
            optimizer_branch1.step()

            # 分支2的梯度裁剪
            l2.backward()
            torch.nn.utils.clip_grad_norm_(optimizer_branch2.param_groups[0]['params'], 12)
            optimizer_branch2.step()

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
        # 将网络参数分为三组：共同部分、分支1独立部分和分支2独立部分
        common_params = []
        branch1_params = []
        branch2_params = []

        for name, param in self.network.named_parameters():
            # 共同部分：增强器和交叉注意力模块
            if 'enhancer' in name or 'cross_attention' in name:
                common_params.append(param)

            # 分支2的独立部分。分支2需要放分支1前面，不然报错
            elif 'out_00' in name or 'dec_block_00' in name or 'up_00' in name or \
                    'out_11' in name or 'dec_block_11' in name or 'up_11' in name or \
                    'out_22' in name or 'dec_block_22' in name or 'up_22' in name or \
                    'out_33' in name or 'dec_block_33' in name or 'up_33' in name:
                branch2_params.append(param)
            # 分支1的独立部分
            elif 'out_0' in name or 'dec_block_0' in name or 'up_0' in name or \
                    'out_1' in name or 'dec_block_1' in name or 'up_1' in name or \
                    'out_2' in name or 'dec_block_2' in name or 'up_2' in name or \
                    'out_3' in name or 'dec_block_3' in name or 'up_3' in name:
                branch1_params.append(param)
            # 其他部分（如编码器、下采样、瓶颈层等）归为共同部分
            else:
                common_params.append(param)

        # 为每一组参数分别配置优化器
        optimizer_common = AdamW(common_params, lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
        optimizer_branch1 = AdamW(branch1_params, lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
        optimizer_branch2 = AdamW(branch2_params, lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)

        # 配置学习率调度器
        scheduler_common = PolyLRScheduler(optimizer_common, self.initial_lr, self.num_epochs, exponent=1.0)
        scheduler_branch1 = PolyLRScheduler(optimizer_branch1, self.initial_lr, self.num_epochs, exponent=1.0)
        scheduler_branch2 = PolyLRScheduler(optimizer_branch2, self.initial_lr, self.num_epochs, exponent=1.0)

        # self.print_to_log_file(f"Using optimizer_common {optimizer_common}")
        # self.print_to_log_file(f"Using optimizer_branch1 {optimizer_branch1}")
        # self.print_to_log_file(f"Using optimizer_branch2 {optimizer_branch2}")
        # self.print_to_log_file(f"Using scheduler_common {scheduler_common}")
        # self.print_to_log_file(f"Using scheduler_branch1 {scheduler_branch1}")
        # self.print_to_log_file(f"Using scheduler_branch2 {scheduler_branch2}")

        return [optimizer_common, optimizer_branch1, optimizer_branch2], [scheduler_common, scheduler_branch1, scheduler_branch2]