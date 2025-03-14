import torch
torch.multiprocessing.set_sharing_strategy('file_system')##训练python脚本中import torch后，加上下面这句。
from load_datasets_transforms_seg import data_loader, data_transforms, infer_post_transforms,remove_regions
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.transforms import AsDiscrete,SaveImaged
from monai.metrics import DiceMetric,MeanIoU,ConfusionMatrixMetric,HausdorffDistanceMetric,SurfaceDistanceMetric
from monai.data.meta_tensor import MetaTensor
import re

import os
import argparse
import yaml
import random
import numpy as np
import SimpleITK as sitk

def config():
    parser = argparse.ArgumentParser(description='overal test')
    ## Input data hyperparameters
    # parser.add_argument('--root', type=str, default='', required=True, help='Root folder of all your images and labels')
    parser.add_argument('--dataset', type=str, default='301', help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')
    parser.add_argument('--patch', type=int, default=(128,128,128), help='Batch size for subject input')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')#5 2
    parser.add_argument('--mode', type=str, default='overal_test', help='Training or testing mode')
    parser.add_argument('--batch_size', type=int, default='1', help='Batch size for subject input')
    parser.add_argument('--crop_sample', type=int, default='2', help='Number of cropped sub-volumes for each subject')
    parser.add_argument('--sw_batch_size', type=int, default=4, help='Sliding window batch size for inference')
    parser.add_argument('--overlap', type=float, default=0.5, help='Sub-volume overlapped percentage')

    ## Efficiency hyperparameters
    parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
    parser.add_argument('--cache_rate', type=float, default=0.2, help='Cache rate to cache your dataset into GPUs')#0.1
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')

    args = parser.parse_args()
    return args

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True / False

def taob_seg():
    args=config() #the first running should excute this code
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    models=["nnUNetTrainerSegResNet"]
    # models=["nnUNetTrainerUNETRPP", "nnUNetTrainerSwinUNETR",
    #           "nnUNetTrainerSegMamba", "nnUNetTrainerUXnet"]
    # models = ["nnUNetTrainerUxLSTMBot",
    #           "nnUNetTrainer", "nnUNetTrainerUMambaBot", "nnUNetTrainerNnformer", "nnUNetTrainerUNETR"]
    labels=["hnnk","lz","cq"]
    file_p="./seg_aob/"
    os.makedirs(file_p, exist_ok=True)
    flags=[1,2] #task1
    for flag in flags:
        for model in models:
            if flag == 1:
                pp = "p4testa/" + model
                ff="2.nii.gz"
                output_file = file_p + model + "_p4a.txt"  ## 创建一个文件来写入结果
            else:
                pp = "p4testb/" + model
                ff="3.nii.gz"
                output_file = file_p + model + "_p4b.txt"  ## 创建一个文件来写入结果
            for label in labels:
                labelsTs = "/media/bit301/data/yml/data/p4/external/"+label  # hnnk lz cq
                # out_test=labelsTs.replace("hnnk","unet/hnnk")
                labelsTs_list = []
                out_test_list=[]
                for root, dirs, files in os.walk(labelsTs, topdown=False):
                    for k in range(len(files)):
                        path = os.path.join(root, files[k])
                        if ff in path:
                            # path="/media/bit301/data/yml/data/p2_nii/external/cq/dis/dmzyyh/PA57/2.nii.gz"
                            labelsTs_list.append(path)
                            target_path = path.replace("p4", pp)  # test/MedNeXtx2
                            # target_path = path.replace("external", "Aortic_index").replace("2.nii.gz","22.nii.gz")
                            # flag=dis_dimention(path, target_path)
                            # flag=check_data(path, target_path)
                            out_test_list.append(target_path)

                val_files = [{"image": image_name, "label": label_name}
                             for image_name, label_name in
                             zip(labelsTs_list, out_test_list)]
                val_transforms = data_transforms(args)
                torch.cuda.empty_cache()
                ## Inference Pytorch Data Loader and Caching
                val_ds = CacheDataset(
                    data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
                val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)

                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                device=torch.device("cpu")
                patch=np.array(args.patch,dtype=int) #(96,96,48)
                out_classes=args.num_classes
                post_label = AsDiscrete(to_onehot=out_classes)
                # post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
                dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False)  # 去除背景项目
                IoU_metric=MeanIoU(include_background=False, reduction="none", get_not_nans=False)
                # HD95_metric=HausdorffDistanceMetric(include_background=False, percentile=0.95,reduction="none", get_not_nans=False)
                conf_matrix_metric = ConfusionMatrixMetric(include_background=False,metric_name="precision", reduction="none", get_not_nans=False)
                # dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
                # conf_matrix_metric=ConfusionMatrixMetric(include_background=True, reduction="none", get_not_nans=False)#percentile=95,
                # assd_Matrix=SurfaceDistanceMetric(include_background=False, reduction="none", symmetric=False,get_not_nans=False)#ASSD应该设置 symmetric=True  结果存在inf指
                for i, val_data in enumerate(val_loader):  # 读取数据不对可能会导致数据加载报错
                    roi_size = patch  # roi_size=(96, 96, 96)
                    a=val_data["image"]
                    # a[a>2]=3
                    # a=np.where(a>1,2,a)
                    b=val_data["label"]
                    # b[b>2]=3
                    del val_data
                    val_labels,val_outputs = (a.to(device), b.to(device))  # 512x512x370
                    val_labels_list = decollate_batch(val_labels)
                    val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                    val_output_list = decollate_batch(val_outputs)
                    val_output_convert = [post_label(val_output_tensor) for val_output_tensor in val_output_list]
                    dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                    dice = dice_metric.aggregate().cpu().detach().numpy()
                    IoU_metric(y_pred=val_output_convert, y=val_labels_convert)
                    iou =IoU_metric.aggregate().cpu().detach().numpy()
                    # HD95_metric(y_pred=val_output_convert, y=val_labels_convert)
                    # HD95 =HD95_metric.aggregate().cpu().detach().numpy()
                    conf_matrix_metric(y_pred=val_output_convert, y=val_labels_convert)
                    ppv=conf_matrix_metric.aggregate()[0].cpu().detach().numpy()
                    # assd_Matrix(y_pred=val_output_convert, y=val_labels_convert)
                    # assd = assd_Matrix.aggregate().cpu().detach().numpy()
                    if i % 50 == 0:
                        print('numbers:', i)
                # mean_dice_val = np.mean(dice_vals)
                # print("dice:",mean_dice_val)
                sub_mean_dice = np.nanmean(dice, axis=0)  # 列平均
                sub_std_dice = np.nanstd(dice, axis=0)  # 列标准差
                sub_mean_iou = np.nanmean(iou, axis=0)  # 列平均
                sub_std_iou = np.nanstd(iou, axis=0)  # 列标准差
                # sub_mean_hd= np.nanmean(HD95, axis=0)  # 列平均
                # sub_std_hd = np.nanstd(HD95, axis=0)  # 列标准差
                sub_mean_ppv = np.nanmean(ppv, axis=0)  # 列平均
                sub_std_ppv = np.nanstd(ppv, axis=0)  # 列标准差
                # sub_mean_assd = np.nanmean(assd, axis=0)  # 列平均
                # sub_std_assd = np.nanstd(assd, axis=0)  # 列标准差

                # print("sub_mean_dice:", sub_mean_dice)
                # print("sub_std_dice:", sub_std_dice)
                # print("sub_mean_ppv:", sub_mean_ppv)
                # print("sub_std_ppv:", sub_std_ppv)
                # print("sub_mean_iou:", sub_mean_iou)
                # print("sub_std_iou:", sub_std_iou)

                # total_mean_dice = np.mean(sub_mean_dice)  # 列平均
                # print("total_mean_dice:", total_mean_dice)
                outname=model+"    "+label
                with open(output_file, 'a') as file:
                    file.write(outname+"\n")
                    file.write(f"sub_mean_dice: {sub_mean_dice}\n")
                    file.write(f"sub_std_dice: {sub_std_dice}\n")

                    file.write(f"sub_mean_iou: {sub_mean_iou}\n")
                    file.write(f"sub_std_iou: {sub_std_iou}\n")

                    # file.write(f"sub_mean_hd: {sub_mean_hd}\n")
                    # file.write(f"sub_std_hd: {sub_std_hd}\n")

                    file.write(f"sub_mean_ppv: {sub_mean_ppv}\n")
                    file.write(f"sub_std_ppv: {sub_std_ppv}\n")
                    # file.write(f"sub_mean_assd: {sub_mean_assd}\n")
                    # file.write(f"sub_std_assd: {sub_std_assd}\n")
                    file.write("\n")

def tab_seg():
    args=config() #the first running should excute this code
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    models=["nnUNetTrainerMednext3"]
    # models=["nnUNetTrainerMednext0a","nnUNetTrainerMednext12","nnUNetTrainerMednext0","nnUNetTrainerMednext13"，"nnUNetTrainerMednext3b"
    # ,"nnUNetTrainerMednext4b"]
    # models=["nnUNetTrainerMednext12","nnUNetTrainerMednext0","nnUNetTrainerUNETR","nnUNetTrainerSwinUNETR",
    #         "nnUNetTrainerSegMamba","nnUNetTrainerUXnet","nnUNetTrainerUxLSTMBot",
    #         "nnUNetTrainer","nnUNetTrainerUMambaBot","nnUNetTrainerNnformer","nnUNetTrainerSegResNet"]
    # models=["nnUNetTrainerMaCNN2","nnUNetTrainerMaCNN3","nnUNetTrainerAortaNet1"]#"nnUNetTrainerMaCNN",  nnUNetTrainerUxLSTMBot
    labels=["hnnk","lz","cq"]
    # labels=["cq"]
    file_p="./seg_ab/"
    os.makedirs(file_p, exist_ok=True)
    flags=[1,2] #task1
    for flag in flags:
        for model in models:
            if flag == 1:
                pp = "p4test/a/" + model
                ff="2.nii.gz"
                output_file = file_p + model + "_p4a.txt"  ## 创建一个文件来写入结果
            else:
                pp = "p4test/b/" + model
                ff="3.nii.gz"
                output_file = file_p + model + "_p4b.txt"  ## 创建一个文件来写入结果
            for label in labels:
                labelsTs = "/media/bit301/data/yml/data/p4/external/"+label  # hnnk lz cq
                # out_test=labelsTs.replace("hnnk","unet/hnnk")
                labelsTs_list = []
                out_test_list=[]
                for root, dirs, files in os.walk(labelsTs, topdown=False):
                    for k in range(len(files)):
                        path = os.path.join(root, files[k])
                        if ff in path:
                            # path="/media/bit301/data/yml/data/p2_nii/external/cq/dis/dmzyyh/PA57/2.nii.gz"
                            labelsTs_list.append(path)
                            target_path = path.replace("p4", pp)  # test/MedNeXtx2
                            # target_path = path.replace("external", "Aortic_index").replace("2.nii.gz","22.nii.gz")
                            # flag=dis_dimention(path, target_path)
                            # flag=check_data(path, target_path)
                            out_test_list.append(target_path)

                val_files = [{"image": image_name, "label": label_name}
                             for image_name, label_name in
                             zip(labelsTs_list, out_test_list)]
                val_transforms = data_transforms(args)
                torch.cuda.empty_cache()
                ## Inference Pytorch Data Loader and Caching
                val_ds = CacheDataset(
                    data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
                val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)

                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                device=torch.device("cpu")
                patch=np.array(args.patch,dtype=int) #(96,96,48)
                out_classes=args.num_classes
                post_label = AsDiscrete(to_onehot=out_classes)
                # post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
                dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False)  # 去除背景项目
                IoU_metric=MeanIoU(include_background=False, reduction="none", get_not_nans=False)
                # HD95_metric=HausdorffDistanceMetric(include_background=False, percentile=0.95,reduction="none", get_not_nans=False)
                conf_matrix_metric = ConfusionMatrixMetric(include_background=False,metric_name="precision", reduction="none", get_not_nans=False)
                # dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
                # conf_matrix_metric=ConfusionMatrixMetric(include_background=True, reduction="none", get_not_nans=False)#percentile=95,
                # assd_Matrix=SurfaceDistanceMetric(include_background=False, reduction="none", symmetric=False,get_not_nans=False)#ASSD应该设置 symmetric=True  结果存在inf指
                for i, val_data in enumerate(val_loader):  # 读取数据不对可能会导致数据加载报错
                    roi_size = patch  # roi_size=(96, 96, 96)
                    a=val_data["image"]
                    # a[a>2]=3
                    # a=np.where(a>1,2,a)
                    b=val_data["label"]
                    # b[b>2]=3
                    del val_data
                    val_labels,val_outputs = (a.to(device), b.to(device))  # 512x512x370
                    val_labels_list = decollate_batch(val_labels)
                    val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                    val_output_list = decollate_batch(val_outputs)
                    val_output_convert = [post_label(val_output_tensor) for val_output_tensor in val_output_list]
                    dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                    dice = dice_metric.aggregate().cpu().detach().numpy()
                    IoU_metric(y_pred=val_output_convert, y=val_labels_convert)
                    iou =IoU_metric.aggregate().cpu().detach().numpy()

                    conf_matrix_metric(y_pred=val_output_convert, y=val_labels_convert)
                    ppv=conf_matrix_metric.aggregate()[0].cpu().detach().numpy()

                    # HD95_metric(y_pred=val_output_convert, y=val_labels_convert)
                    # HD95 =HD95_metric.aggregate().cpu().detach().numpy()
                    # assd_Matrix(y_pred=val_output_convert, y=val_labels_convert)
                    # assd = assd_Matrix.aggregate().cpu().detach().numpy()
                    if i % 50 == 0:
                        print('numbers:', i)
                # mean_dice_val = np.mean(dice_vals)
                # print("dice:",mean_dice_val)
                sub_mean_dice = np.nanmean(dice, axis=0)  # 列平均
                sub_std_dice = np.nanstd(dice, axis=0)  # 列标准差
                sub_mean_iou = np.nanmean(iou, axis=0)  # 列平均
                sub_std_iou = np.nanstd(iou, axis=0)  # 列标准差

                sub_mean_ppv = np.nanmean(ppv, axis=0)  # 列平均
                sub_std_ppv = np.nanstd(ppv, axis=0)  # 列标准差

                # sub_mean_hd= np.nanmean(HD95, axis=0)  # 列平均
                # sub_std_hd = np.nanstd(HD95, axis=0)  # 列标准差

                # sub_mean_assd = np.nanmean(assd, axis=0)  # 列平均
                # sub_std_assd = np.nanstd(assd, axis=0)  # 列标准差

                # print("sub_mean_dice:", sub_mean_dice)
                # print("sub_std_dice:", sub_std_dice)
                # print("sub_mean_ppv:", sub_mean_ppv)
                # print("sub_std_ppv:", sub_std_ppv)
                # print("sub_mean_iou:", sub_mean_iou)
                # print("sub_std_iou:", sub_std_iou)

                # total_mean_dice = np.mean(sub_mean_dice)  # 列平均
                # print("total_mean_dice:", total_mean_dice)
                outname=model+"    "+label
                with open(output_file, 'a') as file:
                    file.write(outname+"\n")
                    file.write(f"sub_mean_dice: {sub_mean_dice}\n")
                    file.write(f"sub_std_dice: {sub_std_dice}\n")

                    file.write(f"sub_mean_iou: {sub_mean_iou}\n")
                    file.write(f"sub_std_iou: {sub_std_iou}\n")

                    # file.write(f"sub_mean_hd: {sub_mean_hd}\n")
                    # file.write(f"sub_std_hd: {sub_std_hd}\n")

                    file.write(f"sub_mean_ppv: {sub_mean_ppv}\n")
                    file.write(f"sub_std_ppv: {sub_std_ppv}\n")
                    # file.write(f"sub_mean_assd: {sub_mean_assd}\n")
                    # file.write(f"sub_std_assd: {sub_std_assd}\n")
                    file.write("\n")

def calculate_and_append_row_averages(file_path):
    # 读取文件
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 初始化变量
    current_group = None
    output_lines = []

    # 处理每一行
    for line in lines:
        line = line.strip()  # 去除首尾空白字符
        if line.startswith('nnUNetTrainert'):
            # 提取分组标记（hnnk, lz, cq）
            current_group = line.split()[-1]
            output_lines.append(line)  # 保留原始行
        elif line.startswith('sub_mean_dice:') or line.startswith('sub_mean_iou:') or line.startswith('sub_mean_ppv:'):
            # 提取数据并计算平均值
            data = line.split(':')[1].strip()  # 提取数据部分
            data = data[1:-1]  # 去掉方括号
            values = [float(x) for x in data.split()]  # 转换为浮点数列表
            mean_value = np.mean(values)  # 计算平均值
            # 追加平均值到行中
            output_lines.append(f"{line}  # {current_group} mean: {mean_value:.4f}")
        else:
            output_lines.append(line)  # 保留其他行

    # 将结果写回文件
    with open(file_path, 'w') as file:
        for line in output_lines:
            file.write(line + '\n')

#对评测的指标计算平均值
def maob_seg():
    models=["nnUNetTrainerUxLSTMBot"]
    # models=["nnUNetTrainerMednext","nnUNetTrainerUNETRPP","nnUNetTrainerSegResNet","nnUNetTrainerSwinUNETR",
    #         "nnUNetTrainerSegMamba","nnUNetTrainerUXnet","nnUNetTrainerUxLSTMBot",
    #         "nnUNetTrainer","nnUNetTrainerUMambaBot","nnUNetTrainerNnformer","nnUNetTrainerUNETR"]
    file_p="./seg_aob/"
    os.makedirs(file_p, exist_ok=True)
    flags=[1,2] #task1
    for flag in flags:
        for model in models:
            if flag == 1:
                file_path = file_p + model + "_p4a.txt"  ## 创建一个文件来写入结果
            else:
                file_path = file_p + model + "_p4b.txt"  ## 创建一个文件来写入结果
            calculate_and_append_row_averages(file_path)

def mab_seg():
    models=["nnUNetTrainerMednext3"]
    # models = ["nnUNetTrainerMednext0a","nnUNetTrainerMednext12", "nnUNetTrainerMednext0", "nnUNetTrainerMednext13"，"nnUNetTrainerMednext3b"]
    file_p = "./seg_ab/"
    os.makedirs(file_p, exist_ok=True)
    flags = [1, 2]  # task1
    for flag in flags:
        for model in models:
            if flag == 1:
                pp = "p4test/a/" + model
                file_path = file_p + model + "_p4a.txt"  ## 创建一个文件来写入结果
            else:
                pp = "p4test/b/" + model
                file_path = file_p + model + "_p4b.txt"  ## 创建一个文件来写入结果

            calculate_and_append_row_averages(file_path)

if __name__ == '__main__':
    # t1catch_seg()
    #p3:原始数据，pp3:拉直数据，t1直接分割，t1total：先分割然后再整体分割，t2：针对拉直数据进行
    # taob_seg() #注意修改config 类别数，要不然要报错
    # maob_seg()
    tab_seg()
    mab_seg()
