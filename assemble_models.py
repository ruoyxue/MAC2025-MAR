import csv
import pickle
import numpy as np


def parse_test_pickle(pickle_path):
    """Load prediction results from pickle."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    probs = [item["pred_score"].detach().cpu().numpy() for item in data]
    preds = [item["pred_label"].detach().cpu().numpy() for item in data]
    gts = [item["gt_label"].detach().cpu().numpy() for item in data]

    return np.array(probs), np.array(preds), np.array(gts)


def fine2coarse(label):
    if label <= 4:
        return 0
    elif label <= 10:
        return 1
    elif label <= 23:
        return 2
    elif label <= 31:
        return 3
    elif label <= 37:
        return 4
    elif label <= 47:
        return 5
    else:
        return 6


def sort_instance_keys(val_labels):
    return sorted(
        list(val_labels.keys()),
        key=lambda x: (
            int(x.split("_")[0]),  # 第一部分数字 (0003 -> 3)
            int(x.split("_")[1]),  # 第二部分数字 (01 -> 1)
            int(x.split("_")[2].split(".")[0]),  # 第三部分数字 (0005 -> 5)
        ),
    )


def sort_test_subset_instance_keys(val_labels):
    return sorted(
        list(val_labels.keys()),
        key=lambda x: (int(x[4:8]),),  # 提取数字部分 (0305 -> 305)
    )


def save_predictions_to_csv(file_names, fused_probs, save_path):
    fused_preds = np.argsort(fused_probs, axis=1)[:, -5:][:, ::-1]

    fused_coarse_probs = np.zeros((fused_probs.shape[0], 7))  # N x 7
    fused_coarse_probs[:, 0] = np.sum(fused_probs[:, 0:5], axis=1)
    fused_coarse_probs[:, 1] = np.sum(fused_probs[:, 5:11], axis=1)
    fused_coarse_probs[:, 2] = np.sum(fused_probs[:, 11:24], axis=1)
    fused_coarse_probs[:, 3] = np.sum(fused_probs[:, 24:32], axis=1)
    fused_coarse_probs[:, 4] = np.sum(fused_probs[:, 32:38], axis=1)
    fused_coarse_probs[:, 5] = np.sum(fused_probs[:, 38:48], axis=1)
    fused_coarse_probs[:, 6] = np.sum(fused_probs[:, 48:], axis=1)

    fused_coarse = np.argsort(fused_coarse_probs, axis=1)[:, -5:][:, ::-1]

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["vid"]
            + [f"action_pred_{i+1}" for i in range(5)]
            + [f"body_pred_{i+1}" for i in range(5)]
        )
        for name, actions, bodies in zip(file_names, fused_preds, fused_coarse):
            writer.writerow([name] + list(actions) + list(bodies))


if __name__ == "__main__":

    with open(
        "/data/xueruoyao/ActionAnalysis_dataset/MA-52/test_subset_instances/all_instance.pickle",
        "rb",
    ) as f:
        val_labels = pickle.load(f)
    video_names = sort_test_subset_instance_keys(val_labels)

    result_paths = {
        "videomae2-base_baseline_clip10_tta": "/home/xueruoyao/MAC2025/MAC-2025_MAR/work_dirs_test_subset/videomae2-base_baseline_clip10_tta/result_merged.pkl",
        "videomae2-base_droppath0.05_clip10_tta": "/home/xueruoyao/MAC2025/MAC-2025_MAR/work_dirs_test_subset/videomae2-base_droppath0.05_clip10_tta/result_merged.pkl",
        "videomae2-base_multi-supervision_clip10_tta": "/home/xueruoyao/MAC2025/MAC-2025_MAR/work_dirs_test_subset/videomae2-base_multi-supervision_clip10_tta/result_merged.pkl",
        "videomae2-base_residual_clip10_tta": "/home/xueruoyao/MAC2025/MAC-2025_MAR/work_dirs_test_subset/videomae2-base_residual_clip10_tta/result_merged.pkl",
    }

    parsed_data = [parse_test_pickle(p) for p in result_paths.values()]
    gts_list = [data[2] for data in parsed_data]
    assert all(
        np.array_equal(gts_list[0], g) for g in gts_list[1:]
    ), "GT mismatch across pickles"

    probs_s = [data[0] for data in parsed_data]
    best_weights = np.array([0.346892, 0.205148, 0.279666, 0.168294], dtype=np.float32)
    fused_probs = sum(w * p for w, p in zip(best_weights, probs_s))

    save_predictions_to_csv(video_names, fused_probs, "prediction.csv")
