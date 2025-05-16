# python /media/bit301/data/yml/project/python310/p3/LightMUNet/nnunetv2/dataset_conversion/Datasets301_Aorta_p3_1ab.py
# nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity #noet：修改size  尽可能的大
 # CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerDSNet0 --c #no-supervision
 # CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerDSNet4 #cross-attention1
  # CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerDSNet44 #cross-attention1
 # CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerDSNet3 #cross-concaten
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerDSNet6 #cross-concaten
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerDSNet66 #cross-concaten
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerDSNet666 #cross-concaten
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerDSNet6666 #cross-concaten
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerDSNet66666 #cross-concaten
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSVSUP
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerDBUNet
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 2d 0 -tr nnUNetTrainerI2UNet
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 2d 0 -tr nnUNetTrainerMTFormer
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 2d 0 -tr nnUNetTrainerMTLSegFormer

##################################### total ##############################################
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMednext --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSegMamba --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUXnet --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerNnformer --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUNETR --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSwinUNETR --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUxLSTMBot --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUNETR --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUMambaBot --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSegResNet --c
