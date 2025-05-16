import numpy as np
import SimpleITK as sitk
import itk
import copy
import os

#使用钙化先验对分割后的钙化进行后处理
def post_possess1():
    ij = 0
    path = "/media/bit301/data/yml/data/test2a/p3_2a/"
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "2.nii.gz" in path:#3.nii.gz
                path_list.append(path)

    # f = open("/media/bit301/data/yml/project/python39/p2/Aorta_net/data/test.txt")  # hnnk test.txt
    # for line in f.readlines():  # tile_step_size=0.75较好处理官腔错位问题
    #     path = line.split('\n')[0]
    for mask_path in path_list:
        path=mask_path.replace("p3_2a/","p3").replace("1.nii.gz", "0.nii.gz")
        img = sitk.ReadImage(path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
        img_array = sitk.GetArrayFromImage(img)
        mask_path1=mask_path
        # mask_path1=mask_path.replace("data/yml/data/p2_nii","use/p2")
        mask_read = sitk.ReadImage(mask_path1, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
        mask_array = sitk.GetArrayFromImage(mask_read)
        mask=copy.deepcopy(mask_array)
        #process
        mask[mask > 0]=1
        img_array1=img_array*mask#勾画区域
        img_array1=np.where(img_array1>130,1,0)
        inverted_mask=1-img_array1
        mask_array[mask_array==2]=1#让钙化也先划分为管腔
        mask_array=mask_array*inverted_mask#腾出钙化区域
        mask_array=mask_array+img_array1*2 #钙化为2
        if mask_array.max()>4:
            print(mask_path)
        label = sitk.GetImageFromArray(mask_array.astype(np.int16))
        sitk.WriteImage(label, mask_path)
        ij = ij + 1
        if ij % 10 == 0:
            print('numbers:', ij)
    print('finished:!')

def remove_small_volums(mask):
    mask1 = np.where(mask > 0, 1, 0)
    segmentation_sitk = sitk.GetImageFromArray(mask1)
    # 计算标签图像的连通分量
    connected_components_filter = sitk.ConnectedComponentImageFilter()
    labeled_image = connected_components_filter.Execute(segmentation_sitk)
    # 获取每个连通分量的大小
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(labeled_image)
    # 初始化一个空的数组来存储处理后的mask
    cleaned_segmentation = np.zeros_like(mask1)
    # 遍历每个连通分量，保留体积大于min_volume的连通区域
    for i in range(1, label_shape_filter.GetNumberOfLabels() + 1):
        if label_shape_filter.GetNumberOfPixels(i) >= 10000:
            binary_mask = sitk.Equal(labeled_image, i)
            binary_mask_array = sitk.GetArrayFromImage(binary_mask)
            cleaned_segmentation[binary_mask_array == 1] = 1
    # 返回处理后的mask
    cleaned_segmentation = cleaned_segmentation * mask
    return cleaned_segmentation.astype(np.int16)

#去除小碎片
def post_possess2():
    path = "/media/bit301/data/yml/data/test1/p3t_1b/nnUNetTrainerMaCNN4/"
    # path="/media/bit301/data/yml/data/test1/p3t_1b/nnUNetTrainerMaCNN4pp/external/cq/dis/dml/PA1"
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "2.nii.gz" in path:#3.nii.gz
                path_list.append(path)

    ij=0
    for mask_path in path_list:
        img = sitk.ReadImage(mask_path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
        img_array = sitk.GetArrayFromImage(img)
        mask_array = remove_small_volums(img_array)
        seg = sitk.GetImageFromArray(mask_array.astype(np.int16))
        out_path = mask_path.replace("/nnUNetTrainerMaCNN4/", "/nnUNetTrainerMaCNN4pp/")  # "33.nii.gz"
        folder_path = os.path.dirname(out_path)
        os.makedirs(folder_path, exist_ok=True)
        sitk.WriteImage(seg, out_path)
        ij = ij + 1
        if ij % 10 == 0:
            print('numbers:', ij)
    print('finished:!')

if __name__ == '__main__':
    # radiologist_score_file()
    # reshape_mask()
    # cq_statistic()
    # lz_statistic()
    # post_possess1()
    post_possess2()
    # remove_case()
    a = 1
