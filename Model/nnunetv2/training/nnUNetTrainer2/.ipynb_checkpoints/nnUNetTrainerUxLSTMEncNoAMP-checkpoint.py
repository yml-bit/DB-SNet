import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import autocast, nn
from nnunetv2.nets.UxLSTM.UxLSTMEnc_3d import get_uxlstm_enc_3d_from_plans
from nnunetv2.nets.UxLSTM.UxLSTMEnc_2d import get_uxlstm_enc_2d_from_plans
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import empty_cache, dummy_context


class nnUNetTrainerUxLSTMEncNoAMP(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_uxlstm_enc_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            model = get_uxlstm_enc_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")
        print("UxLSTMEnc: {}".format(model))

        return model


    # def train_step(self, batch: dict) -> dict:
    #     data = batch['data']
    #     target = batch['target']
    #
    #     data = data.to(self.device, non_blocking=True)
    #     if isinstance(target, list):
    #         target = [i.to(self.device, non_blocking=True) for i in target]
    #     else:
    #         target = target.to(self.device, non_blocking=True)
    #
    #     self.optimizer.zero_grad(set_to_none=True)
    #
    #     #original
    #     # output = self.network(data)
    #     # l = self.loss(output, target)
    #     # l.backward()
    #     # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
    #     # self.optimizer.step()
    #
    #     with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
    #         output = self.network(data)
    #         l = self.loss(output, target)
    #     self.grad_scaler.scale(l).backward()
    #     self.grad_scaler.unscale_(self.optimizer)
    #     torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
    #     self.grad_scaler.step(self.optimizer)
    #     self.grad_scaler.update()
    #
    #     return {'loss': l.detach().cpu().numpy()}
    #
    # def validation_step(self, batch: dict) -> dict:
    #     data = batch['data']
    #     target = batch['target']
    #
    #     data = data.to(self.device, non_blocking=True)
    #     if isinstance(target, list):
    #         target = [i.to(self.device, non_blocking=True) for i in target]
    #     else:
    #         target = target.to(self.device, non_blocking=True)
    #
    #     output = self.network(data)
    #     del data
    #     l = self.loss(output, target)
    #
    #     output = output[0]
    #     target = target[0]
    #
    #     axes = [0] + list(range(2, output.ndim))
    #
    #     if self.label_manager.has_regions:
    #         predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
    #     else:
    #         output_seg = output.argmax(1)[:, None]
    #         predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
    #         predicted_segmentation_onehot.scatter_(1, output_seg, 1)
    #         del output_seg
    #
    #     if self.label_manager.has_ignore_label:
    #         if not self.label_manager.has_regions:
    #             mask = (target != self.label_manager.ignore_label).float()
    #             target[target == self.label_manager.ignore_label] = 0
    #         else:
    #             mask = 1 - target[:, -1:]
    #             target = target[:, :-1]
    #     else:
    #         mask = None
    #
    #     tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
    #
    #     tp_hard = tp.detach().cpu().numpy()
    #     fp_hard = fp.detach().cpu().numpy()
    #     fn_hard = fn.detach().cpu().numpy()
    #     if not self.label_manager.has_regions:
    #         tp_hard = tp_hard[1:]
    #         fp_hard = fp_hard[1:]
    #         fn_hard = fn_hard[1:]
    #
    #     return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}