import numpy as np
from typing import Tuple, Union, List, Optional
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from tqdm import tqdm
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.base_data_loaders import nnUNetDataLoaderBases
from nnunetv2.training.dataloading.nnunet_datasets import nnUNetDatasets
import torch

class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)

        if data_all.shape[1]==2:
            data=data_all[:, 0, :, :, :][:, np.newaxis,:, :, :]
            seg1 = data_all[:, 1, :, :, :][:, np.newaxis,:, :, :]
            seg = np.concatenate([seg1, seg_all], axis=1)
            return {'data': data, 'seg': seg,'properties': case_properties, 'keys': selected_keys}
        else:
            return {'data': data_all, 'seg': seg_all,'properties': case_properties, 'keys': selected_keys}

class SequenceLoader3D(nnUNetDataLoaderBases):
    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        self.tile_step_size = 0.5
        self.verbose = False
        slicers = []
        if len(self.final_patch_size) < len(image_size):
            assert len(self.final_patch_size) == len(
                image_size) - 1, 'if tile_size has less entries than image_size, ' \
                                 'len(tile_size) ' \
                                 'must be one shorter than len(image_size) ' \
                                 '(only dimension ' \
                                 'discrepancy of 1 allowed).'
            steps = compute_steps_for_sliding_window(image_size[1:], self.final_patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is'
                                   f' {image_size}, tile_size {self.final_patch_size}, '
                                   f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.final_patch_size)]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.final_patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(
                f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.final_patch_size}, '
                f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.final_patch_size)]]))
        return slicers

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        case_properties = []
        datas=[]
        segs=[]
        sls=[]
        min_len=10000
        for j, i in enumerate(selected_keys):
            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)
            # data, slicer_revert_padding = pad_nd_image(data, self.final_patch_size,
            #                                            'constant', {'value': 0}, True,
            #                                            None)
            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])
            datas.append(data)
            segs.append(seg)
            sls.append(slicers)
            if min_len>len(slicers):
                min_len=len(slicers)
        if min_len<320:
            min_len=320
        if len(sls[0])==min_len:
            sls[0]= self.extend_list_to_length(sls[0], len(sls[1]))
        else:
            sls[1] = self.extend_list_to_length(sls[1], len(sls[0]))

        # start_index = np.random.randint(0,len(sls[0])-min_len + 1)
        # se0=sls[0][start_index:start_index + min_len]
        # start_index = np.random.randint(0,len(sls[1])-min_len + 1)
        # se1=sls[1][start_index:start_index + min_len]

        self.current_data = datas
        self.current_segs = segs
        self.current_slicers = sls
        self.current_index = 0
        self.case_propertiess=case_properties
        self.selected_keyss=selected_keys
        # return {'data': datas, 'seg': segs, 'current_slicers':current_slicers,'properties': self.case_propertiess, 'keys': self.selected_keyss}

    @staticmethod
    def extend_list_to_length(short_list, target_length):
        """
        扩展较短的列表到与较长列表相同长度。
        :param short_list: 较短的列表
        :param target_length: 目标长度
        :return: 扩展后的列表
        """
        repeat_count = (target_length + len(short_list) - 1) // len(short_list)
        extended_list = (short_list * repeat_count)[:target_length]
        return extended_list

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.current_data is None or self.current_index >= len(self.current_slicers[0]):
                self.generate_train_batch()
                self.current_index = 0
            # 获取当前的patch
            sl0 = self.current_slicers[0][self.current_index]
            sl1 = self.current_slicers[1][self.current_index]

            data_all = np.zeros(self.data_shape, dtype=np.float32)
            seg_all = np.zeros(self.seg_shape, dtype=np.int16)
            data_all[0] = self.current_data[0][sl0][None]
            seg_all[0] = self.current_segs[0][sl0][None]
            data_all[1] = self.current_data[1][sl1][None]
            seg_all[1] = self.current_segs[1][sl1][None]
            self.current_index += 1
            return {'data': data_all, 'seg': seg_all, 'properties': self.case_propertiess, 'keys': self.selected_keyss}

if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnUNetDataset(folder, 0)  # this should not load the properties!
    dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)
