import os
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def fine2coarse(x):
    ranges = [(0, 4), (5, 10), (11, 23), (24, 31), (32, 37), (38, 47), (48, 51)]
    for coarse_id, (start, end) in enumerate(ranges):
        if start <= x <= end:
            return coarse_id
    raise ValueError(f"Value {x} does not fall into any defined range.")


def draw_plot(draw_data, save_path, status):
    plt.figure(figsize=(16, 12))
    sns.countplot(x="class_id", data=draw_data, saturation=0.75)
    counts = draw_data["class_id"].value_counts()
    counts_sort = counts.sort_index(ascending=True)
    plt.title("Distribution of videos " + status)
    for index, value in counts_sort.items():
        plt.text(index, value, value, ha="center", va="bottom")
    plt.savefig(os.path.join(save_path, status + ".png"))


if __name__ == "__main__":

    train_txt = "/data/xueruoyao/ActionAnalysis_dataset/MA-52/annotations/train_val_list_videos.txt"
    # val_txt = "/data/xueruoyao/ActionAnalysis_dataset/MA-52/annotations/val_list_videos.txt"
    train_aug_txt = "/data/xueruoyao/ActionAnalysis_dataset/MA-52/annotations/train_val_list_videos_aug.txt"

    save_path = "data_distribution"
    os.makedirs(save_path, exist_ok=True)

    train_data = pd.read_csv(
        train_txt, sep=" ", header=None, names=["video_name", "class_id"]
    )
    # val_data = pd.read_csv(
    #     val_txt, sep=" ", header=None, names=["video_name", "class_id"]
    # )
    train_aug_data = pd.read_csv(
        train_aug_txt, sep=" ", header=None, names=["video_name", "class_id"]
    )

    coarse_train_data = copy.deepcopy(train_data)
    # coarse_val_data = copy.deepcopy(val_data)
    coarse_train_aug_data = copy.deepcopy(train_aug_data)

    for i in range(52):
        coarse_train_data["class_id"].replace(i, fine2coarse(i), inplace=True)
        # coarse_val_data["class_id"].replace(i, fine2coarse(i), inplace=True)
        coarse_train_aug_data["class_id"].replace(i, fine2coarse(i), inplace=True)

    draw_plot(train_data, save_path, "fine_train_val")
    # draw_plot(val_data, save_path, "fine_val")
    draw_plot(train_aug_data, save_path, "fine_train_val_aug")

    draw_plot(coarse_train_data, save_path, "coarse_train_val")
    # draw_plot(coarse_val_data, save_path, "coarse_val")
    draw_plot(coarse_train_aug_data, save_path, "coarse_train_val_aug")
