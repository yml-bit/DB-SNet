from nnunetv2.nets.MaXlCNN.MaCNN import MaCNN#原始
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from torch import autocast, nn
import torch

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
import copy

#segmanba+medxnet
class nnUNetTrainerMaCNN(nnUNetTrainer):
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
        self.initial_lr = 1e-4
        self.weight_decay = 1e-5
        
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        # MedNeXt
        label_manager = plans_manager.get_label_manager(dataset_json)
        model = MaCNN(in_channels=num_input_channels, n_channels=48,  # 增大通道，减少卷积核l3
                        n_classes=label_manager.num_segmentation_heads, exp_r=[2,3,4,4,4,4,4,3,2],
                        kernel_size=3, deep_supervision=enable_deep_supervision, do_res=True,
                        do_res_up_down=True, block_counts=[2,2,2,2,8,2,2,2,2],  # True False
                        checkpoint_style='outside_block')
        return model
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        binary_mask = (copy.deepcopy(data[1]) != 0).type(torch.uint8)
        data_with_mask = data[1].unsqueeze(1) * binary_mask.unsqueeze(1)
        data = torch.cat((data, data_with_mask), dim=1)
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # original
        # output = self.network(data)
        # l = self.loss(output, target)
        # l.backward()
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        # self.optimizer.step()

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)
        self.grad_scaler.scale(l).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

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