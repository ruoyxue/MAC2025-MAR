import os
import copy
import pickle


no_tta_path = "/home/xueruoyao/MAC2025/MAC-2025_MAR/work_dirs_test_subset/videomae2-base_baseline_clip10/result.pkl"
tta_path = "/home/xueruoyao/MAC2025/MAC-2025_MAR/work_dirs_test_subset/videomae2-base_baseline_clip10_tta/result.pkl"
save_path = "/home/xueruoyao/MAC2025/MAC-2025_MAR/work_dirs_test_subset/videomae2-base_baseline_clip10_tta/result_merged.pkl"


def load_results(file_path):
    with open(file_path, "rb") as file:
        results = pickle.load(file)
    return results


if __name__ == "__main__":

    no_tta_results = load_results(no_tta_path)
    tta_results = load_results(tta_path)

    assert len(no_tta_results) == len(tta_results), "Results length mismatch"

    save_pickle = []
    for no_tta_item, tta_item in zip(no_tta_results, tta_results):
        no_tta_scores = no_tta_item["pred_score"]
        tta_scores = tta_item["pred_score"]

        avg_score = (no_tta_scores + tta_scores) / 2

        save_item = copy.deepcopy(no_tta_item)
        save_item["pred_score"] = avg_score
        save_pickle.append(save_item)

    with open(save_path, "wb") as file:
        pickle.dump(save_pickle, file)
