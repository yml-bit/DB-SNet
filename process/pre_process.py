import numpy as np
import shutil
import random
import os
import SimpleITK as sitk
import cv2
from skimage import measure
import itk
from natsort import natsorted
import copy
from skimage.morphology import ball,binary_closing,disk,binary_dilation
from scipy.ndimage import binary_dilation as binary_dilationn
import datetime

############# 文件操作模块 #############
# 将各个子文件夹合并到一起
def mv_file():
    # path = "../../../data/diag_data/"  # CT_CTA disease
    catch = "../../../data/catch"
    if not os.path.isdir(catch):
        os.makedirs(catch)

    input = '../output/Cyc/1e34/'  ######
    output = '../output/diag1a_data/'  ######
    # output = '/media/yml/yml/data/make_choice/diag0a_data/'  ######
    if not os.path.isdir(input):
        os.makedirs(input)
    if not os.path.isdir(output):
        os.makedirs(output)
    path_list = []
    for root, dirs, files in os.walk(input, topdown=False):
        if "SE1" in root:
            path_list.append(root)
    path_list.sort()

    ii = 0
    for sub_path in path_list:
        input_files = os.listdir(sub_path)
        input_files.sort()
        input_files.sort(key=lambda x: (int(x.split('IM')[1])))
        sub_out = sub_path.replace(input, output)
        sub_out = sub_out.replace("SE1", "SE3")  #######
        if not os.path.isdir(sub_out):
            os.makedirs(sub_out)

        for j in range(len(input_files)):
            in_file_path = os.path.join(sub_path, input_files[j])
            out_file_path = os.path.join(sub_out, input_files[j])
            shutil.move(in_file_path, out_file_path)

        ii = ii + 1
        if ii % 10 == 0:
            print('numbers:', ii)

#批量移除文件
def remove_file():
    path = "../../../data/p2_nii/"  # CT_CTA disease
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        if "4.nii.gz" in files:
            path = os.path.join(root, '4.nii.gz')
            # path_list.append(path)
            os.remove(path)
    # path_list.sort()
    # for sub_path in path_list:
    #     aa = os.path.join(sub_path.split('SE0')[0], 'SE2')
    #     if os.path.isdir(aa):
    #         shutil.rmtree(aa)

def copy_and_paste():
    # path = "/media/bit301/backup/use/p3"  #
    path = "/media/bit301/data/yml/data/p3/"  # p33为备份数据
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "2.nii.gz" in path:
                path_list.append(path)
    # out="/media/bit301/data/yml/data/p3"
    ii = 0
    for se2output in path_list:
        out2 = se2output.replace("p3", "p3_backup_mask2")
        target_directory = os.path.dirname(out2)
        os.makedirs(target_directory, exist_ok=True)
        shutil.copy(se2output, out2)
        se3output=se2output.replace("2.nii.gz", "3.nii.gz")

        # read3 = sitk.ReadImage(se3output, sitk.sitkInt16)  #
        # img_array3 = sitk.GetArrayFromImage(read3)  # 假腔
        # img_array3[img_array3 ==4] =2  # have some caclified。只区分假腔血液，不管假腔的血栓
        # img_array3[img_array3 == 5] = 4
        # out3 = sitk.GetImageFromArray(img_array3.astype(np.int16))
        # sitk.WriteImage(out3, se3output)

        out3=out2.replace("2.nii.gz", "3.nii.gz")
        shutil.copy(se3output, out3)

        # 打印进度
        ii += 1
        if ii % 10 == 0:
            print(f'Processed {ii} files.')
    print("All files processed.")

def remove_small_points(img, threshold_point=20):  # 80
    img_label, num = measure.label(img, return_num=True,connectivity=2)  # 输出二值图像中所有的连通域
    props = measure.regionprops(img_label)  # 输出连通域的属性，包括面积等
    resMatrix = np.zeros(img_label.shape)
    dia = 6  # 60
    for i in range(1, len(props)):
        are = props[i].area
        if are > threshold_point:
            if are < 3400:
                dia = int(np.sqrt(are) / 2)
                tmp = (img_label == i + 1).astype(np.uint8)
                kernel = np.ones((dia, dia), np.uint8)
                tmp = cv2.dilate(tmp, kernel)  # [-1 1]
                # x, y = np.nonzero(tmp)
                # tmp[x[0] - dia:x[-1] + dia, y[0] - dia:y[-1] + dia] = 1  # resize
                resMatrix += tmp  # 组合所有符合条件的连通域
    resMatrix *= 1
    resMatrix[resMatrix > 1] = 1  #
    return resMatrix

# get real input list
def get_files_list():
    path = "/media/bit301/data/yml/data/p3/external/cq/"  # CT_CTA internal external
    files_list = "cq.txt"
    f1 = open(files_list, "w")  # 564
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        # if "xm" in root or "p1/dmzyyh" in root or "hnnk" in root:
        #     if '0.nii.gz' in files:
        #         path_list.append(root.split("p2_nii")[1])
        for file in files:
            path = os.path.join(root, file)
            # if "xm" in path and "0.nii.gz" in path:
            if "0.nii.gz" in path:
                path_list.append(path)
    path_list = natsorted(path_list)
    for j in range(len(path_list)):
        f1.writelines(path_list[j] + "\n")
    # ff.close()
    f1.close()  # 关

def crop(ncct_npy,seg_npy2,seg_npy3,preseg):
    output_size = [1000, 300, 256]  # 统计mask范围得到my=262 mx=217
    crop_y_down=106
    crop_y_up=406
    crop_x_down=128
    crop_x_up=384
    ncct_npy = ncct_npy[:, crop_y_down: crop_y_up, crop_x_down: crop_x_up]
    seg_npy2= seg_npy2[:, crop_y_down: crop_y_up, crop_x_down: crop_x_up]
    seg_npy3= seg_npy3[:, crop_y_down: crop_y_up, crop_x_down: crop_x_up]
    preseg= preseg[:, crop_y_down: crop_y_up, crop_x_down: crop_x_up]
    return ncct_npy,seg_npy2,seg_npy3,preseg

def crop_test_by_mask():#disscard
    path_list = []
    # inputs = input_dir  # 输入目录
    inputs = "/media/bit301/backup/use/p3"  # pre
    for root, dirs, files in os.walk(inputs, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "1.nii.gz" in path:
                path_list.append(path)

    ii = 0
    for path in path_list:
        read = sitk.ReadImage(path, sitk.sitkInt16)
        ncct = sitk.GetArrayFromImage(read)
        path2=path.replace("1.nii.gz", "2.nii.gz")
        read = sitk.ReadImage(path2, sitk.sitkInt16)
        mask2 = sitk.GetArrayFromImage(read)

        path3=path.replace("1.nii.gz", "3.nii.gz")
        read = sitk.ReadImage(path3, sitk.sitkInt16)
        mask3 = sitk.GetArrayFromImage(read)

        path=path.replace("p3", "p4")
        out_put=path.split("1.nii.gz")[0]
        if not os.path.isdir(out_put):
            os.makedirs(out_put)
        ncct,mask2,mask3,_=crop(ncct,mask2,mask3,mask3)#preseg
        ncct = sitk.GetImageFromArray(ncct.astype(np.int16))
        sitk.WriteImage(ncct, path)

        mask2 = sitk.GetImageFromArray(mask2.astype(np.int16))
        sitk.WriteImage(mask2, path.replace("1.nii.gz", "2.nii.gz"))

        mask3 = sitk.GetImageFromArray(mask3.astype(np.int16))
        sitk.WriteImage(mask3, path.replace("1.nii.gz", "3.nii.gz"))
        # 打印进度
        ii += 1
        if ii % 10 == 0:
            print(f'Processed {ii} files.')
    print("All files processed.")

if __name__ == '__main__':
    crop_test_by_mask()
    a = 1