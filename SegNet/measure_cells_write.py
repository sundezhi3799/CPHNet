import matplotlib.pyplot as plt
import cv2,os
import numpy as np
from skimage import measure, morphology, color, io
import skimage.feature
import pandas as pd
import glob
# 1. 读取DNA mask和AGP mask
# 指定要遍历的根目录
root_dir= "D:\data\cell_images\Substructures1220"

def write_head(file):
    result=[]
    result.append('group_dir')
    result.append('hole')
    result.append('cell')

    result.append("nucleus_area")
    result.append("nucleus_perimeter")
    result.append("nucleus_shape_factor")
    result.append("nucleus_aspect_ratio")

    result.append("cell_area")
    result.append("cell_perimeter")
    result.append("cell_shape_factor")
    result.append("cell_aspect_ratio")
    for i in ['dna', 'ag', 'er', 'mit', 'rna', 'pm']:
        result.append(i + '_' + 'glcm_contrast')
        result.append(i + '_' + 'glcm_dissimilarity')
        result.append(i + '_' + 'glcm_homogeneity')
        result.append(i + '_' + 'glcm_energy')
        result.append(i + '_' + 'glcm_correlation')
        result.append(i + '_' + 'glcm_asm')
        result.append(i + '_' + 'granularity')
        result.append(i + '_' + 'intensity')
    file.write(', '.join(result)+'\n')


# 遍历根目录下的所有子目录（group_dir）
for group_dir in os.listdir(root_dir):
    group_dir_path = os.path.join(root_dir, group_dir)
    # 确保当前路径是一个目录而不是文件
    if os.path.isdir(group_dir_path):
        print(f"Group Directory: {group_dir}")
        if os.path.exists(os.path.join(root_dir, group_dir + '_cell_measures.csv')):
            continue
        with open(os.path.join(root_dir, group_dir + '_cell_measures.csv'), 'a') as file:
            write_head(file)
            # 遍历当前子目录中的图像文件
            image_files = os.listdir(group_dir_path) # 按您的需求更改文件扩展名
            prefixes=list(set(map(lambda x:x.split('ag')[0].split('dap')[0].split('dna')[0].split('er')[0].split('mit')[0].split('rna')[0].split('pm')[0],image_files)))
            head=0
            for prefix in prefixes:
                result=[]
                # print(f"cell: {prefix}")
                lst=prefix.split('_')
                hole=lst[0]
                cell_number=lst[2]


                dna_mask = cv2.imread(os.path.join(root_dir, group_dir, prefix + 'dna_mask.png'), cv2.IMREAD_GRAYSCALE)
                dap_mask = cv2.imread(os.path.join(root_dir, group_dir, prefix + 'dap_mask.png'), cv2.IMREAD_GRAYSCALE)
                # 设置阈值
                threshold_value = 128

                # 使用阈值处理将图像转换为二进制图像
                _, dna_mask_binary = cv2.threshold(dna_mask, threshold_value, 255, cv2.THRESH_BINARY)
                _, dap_mask_binary = cv2.threshold(dap_mask, threshold_value, 255, cv2.THRESH_BINARY)


                # 2. 计算细胞核的特征
                # 使用DNA mask进行计算，例如：面积、周长、形状因子、轮廓、长宽比
                dna_labels = measure.label(dna_mask_binary)
                dna_regions = measure.regionprops(dna_labels)

                # 3. 计算细胞质的特征
                # 使用AGP mask进行计算，类似于细胞核的特征
                dap_labels = measure.label(dap_mask_binary)
                dap_regions = measure.regionprops(dap_labels)
                if not dna_regions or not dap_regions:
                    continue
                # max_area=0
                # sec_area=0
                # region=None
                # for reg in dna_regions:
                #     if reg.area>max_area:
                #         max_area=reg.area
                #         region=reg
                #     elif reg.area>sec_area:
                #         sec_area=reg.area
                # if max_area>40*sec_area:
                region=dna_regions[0]
                nucleus_area = region.area
                nucleus_perimeter = region.perimeter
                nucleus_shape_factor = 4 * np.pi * nucleus_area / (nucleus_perimeter ** 2)
                nucleus_contour = region.convex_image
                nucleus_aspect_ratio = region.major_axis_length / region.minor_axis_length
                #     # print("细胞亚结构 %s 特征：" % 'dna')
                #     # print("nucleus_area：", nucleus_area)
                #     # print("nucleus_perimeter：", nucleus_perimeter)
                #     # print("nucleus_shape_factor：", nucleus_shape_factor)
                #     # print("nucleus_aspect_ratio：", nucleus_aspect_ratio)
                #
                # else:
                #     continue

                # max_area = 0
                # sec_area = 0
                # region=None
                # for reg in dap_regions:
                #     # print(reg.area)
                #     if reg.area > max_area:
                #         max_area = reg.area
                #         region = reg
                #     elif reg.area > sec_area:
                #         sec_area = reg.area
                # # print(max_area,sec_area)
                #
                # if max_area > 40 * sec_area:
                region=dap_regions[0]
                cell_area = region.area
                cell_perimeter = region.perimeter
                cell_shape_factor = 4 * np.pi * cell_area / (cell_perimeter ** 2)
                cell_contour = region.convex_image
                cell_aspect_ratio = region.major_axis_length / region.minor_axis_length
                #         # print("细胞亚结构 %s 特征：" % 'dap')
                #         # print("cell_area：", cell_area)
                #         # print("cell_perimeter：", cell_perimeter)
                #         # print("cell_shape_factor：", cell_shape_factor)
                #         # print("cell_aspect_ratio：", cell_aspect_ratio)
                #
                # else:
                #     continue

                result.append(group_dir)
                result.append(hole)
                result.append(cell_number)

                result.append(nucleus_area)
                result.append(nucleus_perimeter)
                result.append(nucleus_shape_factor)
                result.append(nucleus_aspect_ratio)

                result.append(cell_area)
                result.append(cell_perimeter)
                result.append(cell_shape_factor)
                result.append(cell_aspect_ratio)
                # 4. 计算细胞亚结构的特征
                # 读取细胞亚结构图像
                for i in ['dna','ag','er','mit','rna','pm']:
                    substructure_image = cv2.imread(os.path.join(root_dir, group_dir, prefix + i + '.png'), cv2.IMREAD_GRAYSCALE)
                    # 示例：计算纹理特征，可以使用各种纹理分析方法，如GLCM
                    # 计算 GLCM
                    glcm = skimage.feature.graycomatrix(substructure_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

                    contrast = skimage.feature.graycoprops(glcm, 'contrast')[0][0]
                    dissimilarity = skimage.feature.graycoprops(glcm, 'dissimilarity')[0][0]
                    homogeneity = skimage.feature.graycoprops(glcm, 'homogeneity')[0][0]
                    energy = skimage.feature.graycoprops(glcm, 'energy')[0][0]
                    correlation = skimage.feature.graycoprops(glcm, 'correlation')[0][0]
                    asm=skimage.feature.graycoprops(glcm, 'ASM')[0][0]
                    # 示例：计算粒度，可以使用颗粒度分析方法
                    grain_size = len(measure.regionprops(measure.label(substructure_image)))

                    # 示例：计算强度，可以计算平均亮度或颜色强度等
                    mean_intensity = np.mean(substructure_image)

                    # print("细胞亚结构 %s 特征："%i)
                    # print("纹理特征（对比度）：", contrast)
                    # print("粒度：", grain_size)
                    # print("强度（平均亮度）：", mean_intensity)

                    result.append(contrast)
                    result.append(dissimilarity)
                    result.append(homogeneity)
                    result.append(energy)
                    result.append(correlation)
                    result.append(asm)
                    result.append(grain_size)
                    result.append(mean_intensity)
                file.write(', '.join(list(map(str,result)))+'\n')


