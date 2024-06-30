import pandas as pd
import shutil
import os

# 设置CSV文件路径和目标目录路径
# csv_file_path ="D:\PycharmProject\pytorch-cifar100-master\outputs\hypoxia_scores_screening_A549_by_both.csv"# CSV文件路径
# target_directory = "D:\文章\\20230712-cell painting\\20231102-细胞画像素材\\figure5\cells_scoring_0"  # 目标目录路径
csv_file_path ="D:\文章\\20230712-cell painting\\20231102-细胞画像素材\\figure5\hypoxia_scores_1106_by_both.csv"# CSV文件路径
target_directory = "D:\文章\\20230712-cell painting\\20231102-细胞画像素材\\figure5\\1106_scoring_1"  # 目标目录路径

# 读取CSV文件
df = pd.read_csv(csv_file_path)

# 确保目标目录存在
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# 遍历CSV中的行
for index, row in df.iterrows():
    # 检查缺氧分数是否为1
    if row['score'] == 1:
        # image_path = 'D:\data\cell_images\Substructures1220-A549'+row['cell'].replace('data/ours/screening/A549','').replace('Normal','NH')+'_pm.png'  # 获取图像路径
        image_path = 'D:\data\cell_images\Substructures1'+row['cell'].replace('data/ours/test_binary/1','').replace('NH','C')+'_pm.png'  # 获取图像路径
        if os.path.exists(image_path):  # 确保图像文件存在
            # 构建目标路径
            target_path = os.path.join(target_directory, os.path.basename(image_path))
            # 复制图像文件到目标目录
            shutil.copy(image_path, target_path)
        else:
            print(f"File not found: {image_path}")

print("Image extraction completed.")
