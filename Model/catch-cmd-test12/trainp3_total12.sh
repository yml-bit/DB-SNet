#CUDA_VISIBLE_DEVICES=0 python ./nnunetv2/dataset_conversion/Datasets301_Aorta_p3_step12.py
#nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMaCNNC
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUxLSTMBot

CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMaCNN
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUMambaBot

CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMednext

CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainer
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSwinUNETR
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUNETR

CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSegMamba
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUXnet
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerNnformer