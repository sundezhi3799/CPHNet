import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def extract_data_from_event_file(event_file):
    """ 从TensorBoard事件文件中提取标量数据 """
    ea = event_accumulator.EventAccumulator(event_file,
        size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    return {tag: ea.Scalars(tag) for tag in ea.Tags()["scalars"]}

def plot_data(data_dict, tag, label):
    """ 绘制特定标签的数据 """
    steps = [point.step for point in data_dict[tag]]
    values = [point.value for point in data_dict[tag]]
    plt.plot(steps, values, label=label,linewidth=3)

def main():
    root_dir = "D:\文章\\20230712-cell painting\\20231102-细胞画像素材\\figure5\\runs_comparing" # 设置根目录路径
    for tag in [ 'Test/Accuracy', 'Test/Precision', 'Test/Recall', 'Test/F1 Score', 'Test/AUC','Train/loss','Test/Average loss' ]:
        plt.figure(figsize=(10,8))
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            subdir_path=os.path.join(subdir_path,os.listdir(subdir_path)[0])
            for file in os.listdir(subdir_path):
                if "events.out.tfevents" in file:
                    event_file = os.path.join(subdir_path, file)
                    data = extract_data_from_event_file(event_file)
                    if tag in data:
                        plot_data(data, tag, label=subdir)
        tag_name = tag.replace('/', '-').replace('Test','Val')
        plt.xticks(fontname='Arial', fontsize=18, rotation=0)  # 修改字体家族、字号、旋转角度和对齐方式
        plt.yticks(fontname='Arial', fontsize=18, rotation=0)
        # plt.xlabel("Steps")
        plt.ylabel(tag_name.capitalize(),fontdict={'size':24,'fontname':'Arial'})
        # plt.title(f"{tag.capitalize()} over Time")
        plt.legend(fontsize='x-large')
        # plt.show()

        plt.savefig(f'D:\\文章\\20230712-cell painting\\20231102-细胞画像素材\\figure5\\{tag_name}.pdf')

if __name__ == "__main__":
    main()
