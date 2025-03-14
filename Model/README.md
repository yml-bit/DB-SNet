Guidelines
1-install: details can be see[https://github.com/MIC-DKFZ/nnUNet]
2-data process:use test/data_process.py (1)resample and convert DICOM data to nii.gz file 
(2)make annotation and process,including mask Silhouette and multi-category mask merge 

3-training and inference: 
(1)set correct path：./nnunetv2/path.py # 
(2)data prepare：./nnunetv2/dataset_conversion/Datasets301_Aorta_p3_1ab.py for DB-SNet, Datasets301_Aorta_p3_1a.py for task 1, Datasets301_Aorta_p3_1b.py for task 2 #read the code carefully and figure out format of the output file
(3)data checking and training preparation:execute a command "nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity" under the path nnunet-v2 
(4)Model/nnunetv2/training/nnUNetTrainer(baseline model)/nnUNetTrainer2(two-task model)/nnUNetTrainer3(single branch tuning model)
note:making sure that the name of the executed task file is always nnUNetTrainer
(5) run:change the root to the model file and run bash DB-SNet-"./command/total_1ab.sh"    single-task model for task 1- "./command/total_1a.sh"     single-task model for task 2-"./command/total_1b.sh"
note:need to ensure that the path for different task
