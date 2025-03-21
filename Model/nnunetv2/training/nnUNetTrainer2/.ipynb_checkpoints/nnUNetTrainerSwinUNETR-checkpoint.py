from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import autocast, nn

from monai.networks.nets import SwinUNETR
from nnunetv2.utilities.helpers import empty_cache, dummy_context
import copy

class nnUNetTrainerSwinUNETR(nnUNetTrainerNoDeepSupervision):
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
        self.initial_lr = 3e-4
        self.weight_decay = 1e-5
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:

        label_manager = plans_manager.get_label_manager(dataset_json)

        model = SwinUNETR(
            in_channels = num_input_channels,
            out_channels = label_manager.num_segmentation_heads,
            img_size = configuration_manager.patch_size,
            depths = (2, 2, 2, 2),
            num_heads = (3, 6, 12, 24),
            feature_size = 24, ##
            norm_name = "instance",
            drop_rate = 0.0,
            attn_drop_rate = 0.0,
            dropout_path_rate = 0.0,
            normalize = True,
            use_checkpoint = False,
            spatial_dims = len(configuration_manager.patch_size),
            downsample = "merging",
            use_v2 = False,
        )

        return model
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        binary_mask = (copy.deepcopy(data[1]) != 0).type(torch.uint8)
        data_with_mask = data[1].unsqueeze(1) * binary_mask.unsqueeze(1)
        data = torch.cat((data, data_with_mask), dim=1)
        target = batch['target']
        # s=data.shape
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        # original
        output = self.network(data)
        l = self.loss(output, target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()

        # with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
        #     output = self.network(data)
        #     l = self.loss(output, target)
        # self.grad_scaler.scale(l).backward()
        # self.grad_scaler.unscale_(self.optimizer)
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        # self.grad_scaler.step(self.optimizer)
        # self.grad_scaler.update()
        
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
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

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        output = self.network(data)
        del data
        l = self.loss(output, target)

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    def configure_optimizers(self):

        optimizer = AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler
    
    def set_deep_supervision_enabled(self, enabled: bool):
        pass