python /media/bit301/data/yml/project/python310/p3/LightMUNet/nnunetv2/dataset_conversion/Datasets301_Aorta_p3_1ab.py
nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity #noet：修改size  尽可能的大
nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMednext0 #make sure the “nnUNetTrainer2”change to "nnUNetTrainer"
nnUNetv2_train 301 3d_fullres 0 -tr nnUNetTrainerMednext3b  --c #make sure the “nnUNetTrainer3”change to "nnUNetTrainer"
 python ./nnunetv2/inference/predict_from_raw_data_1.py 0 nnUNetTrainerMednext3b ##make sure the “nnUNetTrainer3”change to "nnUNetTrainer"