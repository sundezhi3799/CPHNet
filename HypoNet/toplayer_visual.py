import pandas as pd
from sklearn.manifold import TSNE
import os
from matplotlib import pyplot as plt

features_out_dir = 'features/1106_both'
# all_features_df=pd.read_csv(os.path.join(features_out_dir, 'all_resnet_features.csv'))
all_features_df=pd.read_csv(os.path.join(features_out_dir, '1106_resnet_features.csv'))

# Plot t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
tsne_result = tsne.fit_transform(all_features_df.iloc[:, :-1])
names = all_features_df['cell']
# Create a color map for each directory
unique_dirs = list(set([os.path.dirname(name) for name in names]))
unique_dirs.sort()
dir_to_color = {dir_name: i for i, dir_name in enumerate(unique_dirs)}
colors = [dir_to_color[os.path.dirname(name)] for name in names]

# Scatter plot
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, cmap='viridis')
plt.title('t-SNE Visualization of HypoNet Features',fontdict={'size':18,'fontname':'Arial'})
plt.xlabel('t-SNE Dimension 1',fontdict={'size':18,'fontname':'Arial'})
plt.ylabel('t-SNE Dimension 2',fontdict={'size':18,'fontname':'Arial'})
# Create legend
legend_labels = list(map(lambda x: x.split('__')[0].split('9-')[-1].split('-')[0].replace('C','N'), list(unique_dirs)))
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title='classes', loc='upper right')
plt.savefig('1106_both_tsne_plot.pdf')
# plt.show()