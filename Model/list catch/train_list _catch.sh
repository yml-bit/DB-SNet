train list
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMaCNN
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMaCNNC
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSegMamba
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMednext
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainer
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUxLSTMBot
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUMambaBot
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUXnet
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerNnformer
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSwinUNETR
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUNETR


train_total_2a
python /media/bit301/data/yml/project/python310/p3/LightMUNet/nnunetv2/dataset_conversion/Datasets301_Aorta_p3.py
nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity  #noet：输入多模态需要修改default_preprocessor.py line 183
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMednext
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSegMamba
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUXnet
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerNnformer
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSwinUNETR
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUNETR 
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainer

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUMambaBot
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSegResNet
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUxLSTMBot
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerAortaNet1

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUxLSTMEnc
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUMambaEnc
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerAortaNet
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMaXlCNN
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMaCNN
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMaCNNC
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMaXlCNNC-v1
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMaXlCNNC-v1


train_total_2b
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerunet #利用2a 模型的输出作为其他模型的输入之一
python /media/bit301/data/yml/project/python310/p3/LightMUNet/nnunetv2/dataset_conversion/Datasets301_Aorta_p3_2b.py
nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity #noet：输入多模态需要修改default_preprocessor.py line 183
python /media/bit301/data/yml/project/python310/p3/LightMUNet/nnunetv2/dataset_conversion/Datasets301_Aorta_p3_revisedjson.py  #revis crop size:64x224x224
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerAortaNet1 #Mednext+two head
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMaCNN2 #mamba+Mednext+concaten
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMednext
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainer
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSegMamba
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUxLSTMBot
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUMambaBot
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUXnet
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerNnformer
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSwinUNETR
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerUNETR
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerSegResNet
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMaCNN1 #mamba+Mednext
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMaCNN2 #mamba+Mednext+concaten
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMaCNN3 #mamba+Mednext+concaten  96->128
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMaCNN4 #mamba+Mednext+concaten  96->128  +[-200 400]
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMaCNN
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMaCNN#C


