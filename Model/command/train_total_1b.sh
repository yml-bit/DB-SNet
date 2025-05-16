# python /media/bit301/data/yml/project/python310/p3/LightMUNet/nnunetv2/dataset_conversion/Datasets301_Aorta_p3_1b.py
# nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity #noet：修改size  尽可能的大
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMednext --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUNETRPP --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainer 
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUXnet
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerNnformer 
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUNETR 
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSwinUNETR
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUxLSTMBot 
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUMambaBot
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSegResNet
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSegMamba 
python ./nnunetv2/inference/predict_from_raw_data_1b.py 0 nnUNetTrainerNnformer
python ./nnunetv2/inference/predict_from_raw_data_1b.py 0 nnUNetTrainerUNETRPP
python ./nnunetv2/inference/predict_from_raw_data_1b.py 0 nnUNetTrainerSwinUNETR
