import os
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# 设置预测和ground_truth目录的路径
prediction_dir = 'D:\data\cell_images\man\DAP_DNA\\both\\test\CP'
ground_truth_dir = 'D:\data\cell_images\man\DAP_DNA\\both\\test\GT'

# 获取两个目录中的所有文件名
prediction_files = os.listdir(prediction_dir)
ground_truth_files = os.listdir(ground_truth_dir)

# 初始化存储精确度、召回率和F1分数的列表
precisions_dna = []
precisions_agp = []
recalls_dna = []
recalls_agp = []
f1_scores_dna = []
f1_scores_agp = []
# 遍历预测目录中的每个文件
for prediction_file in prediction_files:
    # 构建ground_truth文件路径

    ground_truth_file = os.path.join(ground_truth_dir, prediction_file)

    # 读取预测掩码和ground_truth掩码
    prediction_mask = cv2.imread(os.path.join(prediction_dir, prediction_file), cv2.IMREAD_GRAYSCALE)
    ground_truth_mask = cv2.imread(ground_truth_file, cv2.IMREAD_GRAYSCALE)

    # 将二进制掩码转换为二进制数组
    prediction_binary = (prediction_mask > 0).astype(np.uint8)
    ground_truth_binary = (ground_truth_mask > 0).astype(np.uint8)

    # 计算精确度、召回率和F1分数
    precision = precision_score(ground_truth_binary.flatten(), prediction_binary.flatten())
    recall = recall_score(ground_truth_binary.flatten(), prediction_binary.flatten())
    f1 = f1_score(ground_truth_binary.flatten(), prediction_binary.flatten())

    # 将得分添加到列表中
    if 'ch1' in prediction_file:
        precisions_dna.append(precision)
        recalls_dna.append(recall)
        f1_scores_dna.append(f1)
    if 'ch3' in prediction_file:
        precisions_agp.append(precision)
        recalls_agp.append(recall)
        f1_scores_agp.append(f1)

# 计算平均精确度、召回率和F1分数
average_precision_dna = np.mean(precisions_dna)
average_precision_agp = np.mean(precisions_agp)
average_recall_dna = np.mean(recalls_dna)
average_recall_agp = np.mean(recalls_agp)
average_f1_dna = np.mean(f1_scores_dna)
average_f1_agp = np.mean(f1_scores_agp)

# 输出平均得分
lst = [['dna Average Precision', average_precision_dna], ['dap Average Precision', average_precision_agp],
       ['dna Average Recall', average_recall_dna], ['dap Average Recall', average_recall_agp],
       ['dna Average F1 score', average_f1_dna], ['dap Average F1 score', average_f1_agp]]
import pandas as pd

a=pd.DataFrame(lst)
a.to_csv(os.path.join(prediction_dir,'perfomances_CP.csv'),index=None,header=None)
print(lst)
