#chage DATASET name 
method1
CUDA_VISIBLE_DEVICES=2 python ./nnunetv2/dataset_conversion/Datasets301_Aorta_p3_step1.py
nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerunet

#predict_train1.py get predicted and put in DATASET1, and then renmae DATASET1 to DATASET12
CUDA_VISIBLE_DEVICES=2 python ./nnunetv2/inference/predict_train1.py
python /root/autodl-tmp/yml/project/python310/p3/pre_process.py
CUDA_VISIBLE_DEVICES=2 python ./nnunetv2/dataset_conversion/Datasets301_Aorta_p3_step12.py
nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity
CUDA_VISIBLE_DEVICES=2 python ./nnunetv2/dataset_conversion/Datasets301_Aorta_p3_step12c.py



或者method2
CUDA_VISIBLE_DEVICES=2 python ./nnunetv2/dataset_conversion/Datasets301_Aorta_p3_step2.py #1
CUDA_VISIBLE_DEVICES=2 python ./nnunetv2/dataset_conversion/Datasets301_Aorta_p3_step12.py
copy step1 mask to files
nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity
CUDA_VISIBLE_DEVICES=2 python ./nnunetv2/dataset_conversion/Datasets301_Aorta_p3_step12c.py
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
