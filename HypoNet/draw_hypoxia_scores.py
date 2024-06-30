import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 使用Seaborn库增强美观性
# 读取CSV文件
type='A549'
# type='HPMEC'

# df = pd.read_csv(f'outputs/hypoxia_scores_screening_A549_by_both_vgg16.csv')
# df = pd.read_csv(f'outputs/hypoxia_scores_{type}_by_both_hyponet.csv')
df = pd.read_csv(f'outputs/hypoxia_scores_screening_{type}_by_both.csv')

# 提取第一列中/之前的字符串作为分组依据
df['Group'] = df['cell'].apply(lambda x: x.split('__')[0].split('-')[-1])
# df['Group'] = df['cell'].apply(lambda x: x.split('__')[0].split('-')[-2])


# 按照分组计算score的平均值和方差
grouped_data = df.groupby('Group')['score'].agg(['mean','std', 'sem']).reset_index()
grouped_data = grouped_data.sort_values(by='mean')
# 使用Seaborn设置美观的风格
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 44})
# 绘制条形图
plt.figure(figsize=(21, 18))
ax = sns.barplot(x='Group', y='mean', data=grouped_data, errorbar='se', palette='viridis_r')

# 添加误差线
for bar, err in zip(ax.patches, grouped_data['sem']):
    plt.errorbar(bar.get_x() + bar.get_width()/2, bar.get_height(), yerr=err, color='black', linewidth=1.5,capsize=15)

# 设置图表标题和标签
# plt.xlabel('Oxygen Concentration',fontdict={'size':48,'fontname':'Arial'})
plt.xlabel('Agents',fontdict={'size':48,'fontname':'Arial'},color='black')
plt.ylabel('Average Hypoxia Score',fontdict={'size':48,'fontname':'Arial'},color='black')
# plt.title(f'Average Hypoxia Score for {type}',fontdict={'size':28,'fontname':'Arial'})
# 设置组别标签的字体
plt.xticks(fontname='Arial', fontsize=24, rotation=45,color='black')  # 修改字体家族、字号、旋转角度和对齐方式
plt.yticks(fontname='Arial', fontsize=24, rotation=0,color='black')  # 修改字体家族、字号、旋转角度和对齐方式
# 旋转x轴标签，使其更易读
# plt.xticks(rotation=45, ha='right')

# 显示图表
# plt.show()
# plt.savefig(f'outputs/hypoxia_scores_screening_A549_by_both_vgg16.pdf')
# plt.savefig(f'outputs/Hypoxia_scores_{type}_by_both_hyponet.pdf')
plt.savefig(f'outputs/Hypoxia_scores_screening_{type}_by_both_hyponet.pdf')