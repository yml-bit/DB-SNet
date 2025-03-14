from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from pathlib import Path
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw,nnUNet_preprocessed
import random
import SimpleITK as sitk
import numpy as np

def make_out_dirs(dataset_id: int, task_name="Aorta"):
    dataset_name = f"Dataset{dataset_id:03d}_{task_name}"
    out_dir = Path(nnUNet_raw.replace('"', "")) / dataset_name
    train_img_dir = out_dir / "imagesTr"
    train_labels_dir = out_dir / "labelsTr"
    test_img_dir = out_dir / "imagesTs"
    test_labels_dir = out_dir / "labelsTs"
    os.makedirs(out_dir, exist_ok=True)

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)

    return out_dir, train_img_dir, train_labels_dir, test_img_dir,test_labels_dir

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def copy_files(src_data_folder: Path, train_dir: Path, labelTr_dir: Path, test_dir: Path,labelTs_dir: Path):
    """Copy files from the ACDC dataset to the nnUNet dataset folder. Returns the number of training cases."""

    patients_train = []#后续会自动划分训练集和测试集
    for root, dirs, files in os.walk(src_data_folder, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "0.nii.gz" in path:
                patients_train.append(path)
    # random.seed(0)
    # random.shuffle(patients_train)

    # src_data_folderTs="/home/wangtao/yml/data/p2_nii/external"  ##后续使用需要对其更换为测试及 p2_ss
    # patients_test=[]
    # for root, dirs, files in os.walk(src_data_folderTs, topdown=False):
    #     for file in files:
    #         path = os.path.join(root, file)
    #         if "0.nii.gz" in path:
    #             patients_test.append(path)

    num_cases = 0
    num_training_cases=0
    pathh=os.path.join(nnUNet_preprocessed,"Dataset301_Aorta")
    os.makedirs(pathh, exist_ok=True)
    train =os.path.join(nnUNet_preprocessed,"Dataset301_Aorta","train.json")
    trainObject = open(train, 'a', encoding='utf-8')
    trainObject.seek(0)#
    trainObject.truncate()#clear content

    num_test_cases = 0
    test = os.path.join(nnUNet_preprocessed, "Dataset301_Aorta", "test.json")
    testObject = open(test, 'a', encoding='utf-8')
    testObject.seek(0)  #
    testObject.truncate()  # clear content

    splits_file = os.path.join(nnUNet_preprocessed,"Dataset301_Aorta", "splits_final.json")
    splits = []
    train_keys=[]
    test_keys=[]
    # Copy training files and corresponding labels.
    for path_str in patients_train:#内部数据用于开发，选择一个外部数据作为验证集。
        file=Path(path_str)
        if file.name == "0.nii.gz":
            # We split the stem and append _0000 and _0001 to the patient part.
            # se0output = os.path.join(nnUNet_raw, path_str.split("p3/")[1])
            # os.makedirs(se0output.split("/0.nii.gz")[0])

            shutil.copy(file, train_dir / f"Aorta_{str(num_cases).zfill(4)}_0000.nii.gz")

            CTA_path=path_str.replace("0.nii.gz","1.nii.gz")
            shutil.copy(CTA_path, train_dir / f"Aorta_{str(num_cases).zfill(4)}_0001.nii.gz")

            label_path=path_str.replace("0.nii.gz","2.nii.gz")
            shutil.copy(label_path, labelTr_dir / f"Aorta_{str(num_cases).zfill(4)}.nii.gz")

            new_data = {f"Aorta_{str(num_cases).zfill(4)}": path_str}
            js = json.dumps(new_data, ensure_ascii=False)
            if "external" not in path_str:
                num_training_cases += 1
                trainObject.write(js +'\n')
                train_keys.append(f"Aorta_{str(num_cases).zfill(4)}")
            else:
                num_test_cases += 1
                testObject.write(js + '\n')
                test_keys.append(f"Aorta_{str(num_cases).zfill(4)}")
            num_cases += 1
        else:
            print(path_str)
    splits.append({})
    splits[-1]['train'] = train_keys#list(train_keys)
    splits[-1]['val'] = test_keys#list(test_keys)
    save_json(splits, splits_file)
    trainObject.close()
    testObject.close()
    return num_cases,num_test_cases


def convert_Aorta(src_data_folder: str, dataset_id=301):
    out_dir, train_img_dir, train_labels_dir, test_img_dir,test_labels_dir = make_out_dirs(dataset_id=dataset_id)
    num_cases,num_test_cases = copy_files(src_data_folder, train_img_dir, train_labels_dir, test_img_dir,test_labels_dir)

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "CT",
            1: "CTA",
        },
        labels={    #将病灶标签区间融合到label里面？
            "background": 0,
            "blood": 1,
            "calcified": 2,
            "noncalcified": 3,
            "fake_cavity ":4
        },
        file_ending=".nii.gz",
        num_training_cases=num_cases,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="The downloaded Aorta dataset dir. Should contain extracted 'training' and 'testing' folders.",
    )
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=301, help="nnU-Net Dataset ID, default: 301"
    )
    args = parser.parse_args()
    print("Converting...")
    # convert_Aorta(args.input_folder, args.dataset_id)
    input_folder="/media/bit301/data/yml/data/p3/pre/"  #define training and test datasets by self
    dataset_id=301
    convert_Aorta(input_folder, dataset_id)
    print("Done!")
