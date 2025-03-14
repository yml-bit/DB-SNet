python /media/bit301/data/yml/project/python310/p3/LightMUNet/nnunetv2/dataset_conversion/Datasets301_Aorta_p3_1a.py
nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity #noet：修改size  尽可能的大
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMednext
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUNETRPP
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0  nnUNetTrainer
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSegMamba
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUXnet
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerNnformer
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSwinUNETR
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUxLSTMBot
#################################### inference##############################################
python ./nnunetv2/inference/predict_from_raw_data_1a.py 0 nnUNetTrainerMednext
python ./nnunetv2/inference/predict_from_raw_data_1a.py 0 nnUNetTrainerUNETRPP
python ./nnunetv2/inference/predict_from_raw_data_1a.py 0 nnUNetTrainerUNETR
python ./nnunetv2/inference/predict_from_raw_data_1a.py 0 nnUNetTrainerNnformer
python ./nnunetv2/inference/predict_from_raw_data_1a.py 0 nnUNetTrainerSwinUNETR
python ./nnunetv2/inference/predict_from_raw_data_1a.py 0 nnUNetTrainerSegMamba
python ./nnunetv2/inference/predict_from_raw_data_1a.py 0 nnUNetTrainer
python ./nnunetv2/inference/predict_from_raw_data_1a.py 0 nnUNetTrainerUxLSTMBot
python ./nnunetv2/inference/predict_from_raw_data_1a.py 0 nnUNetTrainerUXnet
python ./nnunetv2/inference/predict_from_raw_data_1a.py 0 nnUNetTrainerSegResNet
