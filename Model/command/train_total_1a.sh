# python /media/bit301/data/yml/project/python310/p3/LightMUNet/nnunetv2/dataset_conversion/Datasets301_Aorta_p3_1a.py
# nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity #noet：修改size  尽可能的大
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMednext --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUNETRPP --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0  nnUNetTrainer
##################################### total ##############################################
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMednext
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSegMamba --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUXnet --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerNnformer --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUNETR --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSwinUNETR --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUxLSTMBot --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUNETR --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUMambaBot --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSegResNet --c
