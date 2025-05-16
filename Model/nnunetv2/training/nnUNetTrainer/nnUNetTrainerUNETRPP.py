from nnunetv2.nets.UNETRPP.unetr_pp import UNETR_PP#原始
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
#     nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import autocast, nn
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

class nnUNetTrainerUNETRPP(nnUNetTrainer):
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
        # self.initial_lr = 1e-2
        # self.weight_decay = 1e-5
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:#False

        label_manager = plans_manager.get_label_manager(dataset_json)

        model = UNETR_PP(in_channels=num_input_channels,
                             out_channels=label_manager.num_segmentation_heads,
                             feature_size=16,
                             num_heads=4,
                             depths=[3, 3, 3, 3],#[3, 3, 3, 3] [2, 2, 2, 2]
                             dims=[32, 64, 128, 256],
                             do_ds=True,
                             )

        return model
    
    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=1.0)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")
        return optimizer, scheduler