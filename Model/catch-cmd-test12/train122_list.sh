#chage DATASET name 
CUDA_VISIBLE_DEVICES=0 python ./nnunetv2/dataset_conversion/Datasets301_Aorta_p3_step1.py
nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerunet

CUDA_VISIBLE_DEVICES=0 python ./nnunetv2/inference/predict_train1.py
#CUDA_VISIBLE_DEVICES=0 python ./nnunetv2/Datasets301_Aorta_p3_step12.py
CUDA_VISIBLE_DEVICES=0 python ./nnunetv2/Datasets301_Aorta_p3_step122.py #roi

nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity
#chage modalites2= to 3
#chage nnUNetTrainerr to nnUNetTrainer
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMaCNNC
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMaCNNC
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMaCNN
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSegMamba
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMednext
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainer
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUxLSTMBot
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUMambaBot
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUXnet
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerNnformer
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSwinUNETR
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUNETR
