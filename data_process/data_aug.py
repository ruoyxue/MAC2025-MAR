import math
import pandas as pd


train_txt = "/data/xueruoyao/ActionAnalysis_dataset/MA-52/annotations/train_val_list_videos.txt"
train_aug_txt = "/data/xueruoyao/ActionAnalysis_dataset/MA-52/annotations/train_val_list_videos_aug.txt"

train_data = pd.read_csv(
    train_txt, sep=" ", header=None, names=["video_name", "class_id"]
)

class_counts = train_data["class_id"].value_counts()
underrepresented_classes = class_counts[class_counts < 200]

if not underrepresented_classes.empty:
    repeat_times = (200 // underrepresented_classes).apply(math.log2).astype(int) + 1

    data_to_add = []
    for class_id, n_repeats in repeat_times.items():
        class_subset = train_data[train_data.class_id == class_id]
        data_to_add.extend([class_subset] * n_repeats)

    final_data = pd.concat([train_data] + data_to_add, ignore_index=True)
else:
    final_data = train_data


final_data.to_csv(train_aug_txt, sep=" ", index=False, header=False)

print(f"Original dataset size: {len(train_data)}")
print(f"Augmented dataset size: {len(final_data)}")
print(f"Result saved to: {train_aug_txt}")
