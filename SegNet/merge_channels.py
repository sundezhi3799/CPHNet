import os
import cv2
import glob
from data_path import *

# 源图像目录和目标目录
for source_dir in paths[-1:]:
    # source_dir = 'D:\data\cell_images\\20230613\\20230613-CELL PAINTING-A549T__2023-06-13T14_12_14-Measurement 2\output'
    print(len(paths))
    print(paths.index(source_dir),source_dir)
    target_dir = source_dir+'_merged'

    # 创建目标目录
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 获取所有图像文件
    image_files = glob.glob(os.path.join(source_dir, '*.tiff'))  # 你可以根据实际的图像格式来修改文件扩展名

    # 遍历图像文件
    for image_file in image_files:
        # 获取文件名和扩展名
        file_name = os.path.basename(image_file)
        base_name, ext = os.path.splitext(file_name)

        # 获取前缀和后缀
        prefix, suffix = base_name.split('-')[0], base_name.split('-')[1]

        # 根据后缀选择通道
        if 'ch1' in suffix:
            channel = 0
        elif 'ch3' in suffix:
            channel = 1
        elif 'ch5' in suffix:
            channel = 2
        else:
            # 如果后缀不包含ch1、ch3、ch5，则跳过该图像
            continue

        # 读取图像
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        img=img/img.max()*255

        # 创建一个3通道的黑白图像
        if channel == 0:
            merged_img = cv2.merge((img, img, img))
        else:
            # 对于通道1和通道2，将它们合并到3通道图像的相应通道上
            merged_img = cv2.imread(os.path.join(target_dir, prefix + '.png'))

            if merged_img is None:
                merged_img = cv2.merge((img, img, img))
            else:
                merged_img[:, :, channel] = img

        # 保存合并后的图像到目标目录
        cv2.imwrite(os.path.join(target_dir, prefix + '.png'), merged_img)

    print('图像合并完成，并保存到目标目录:', target_dir)
