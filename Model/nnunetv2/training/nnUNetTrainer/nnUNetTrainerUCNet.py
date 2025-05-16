from nnunetv2.nets.UCTNet.uctnet_3D import UCTNet_3D#原始
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
#     nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
import torch
from torch.optim import AdamW
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch import autocast, nn
from nnunetv2.utilities.helpers import empty_cache, dummy_context

class nnUNetTrainerUCNet(nnUNetTrainer):
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
        # self.grad_scaler = None
        self.initial_lr = 1e-3
        self.weight_decay = 1e-5
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:#False

        label_manager = plans_manager.get_label_manager(dataset_json)

        model = UCTNet_3D(img_size=[128,128,128],
                                    base_num_features=24,
                                    num_classes=label_manager.num_segmentation_heads,
                                    num_pool=5,
                                    image_channels=num_input_channels,
                                    pool_op_kernel_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                                    conv_kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                                    deep_supervision=True,
                                    max_num_features=24*13,
                                    bound_sizes=[11, 7, 3, 3, 3, 3],
                                    dmodels=[768, 768, 768, 768, 768, 768],
                                    depths=[1, 1, 1, 1, 1, 1],
                                    num_heads=[3, 3, 3, 6, 12, 24],
                                    patch_size=[[4,12,12],[4,6,6],[2,3,3],[1,2,2],[1,1,1],[1,1,1]],
                                    dim_head=64,
                                    add_Map=False)#True

        return model
    
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