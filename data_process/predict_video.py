import os
import glob
import pickle
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from multiprocessing import Pool


def extract_bounding_box(results):
    bbox_list = []
    for result in results:
        bboxes = result.boxes.xyxy.cpu().numpy()
        if bboxes.shape[0] > 0:
            bbox_list.append(bboxes[0])
    return np.round(np.min(bbox_list, axis=0)).tolist() if bbox_list else []


def predict(video_list, index, device):
    all_bbox_dict = {}
    for video_path in tqdm(video_list, desc=f"Chunk {index+1}"):
        results = model.predict(
            source=video_path,
            stream=True,
            device=device,
            classes=0,
            iou=0.45,
            conf=0.5,
            save=False,
            save_txt=False,
            verbose=False,
        )

        video_name = os.path.basename(video_path)
        all_bbox_dict[video_name] = extract_bounding_box(results)

    with open(os.path.join(pickle_dir, f"{index}.pickle"), "wb") as fr:
        pickle.dump(all_bbox_dict, fr)


if __name__ == "__main__":

    model = YOLO("/data/xueruoyao/ActionAnalysis_dataset/MA-52/yolov8m.pt")

    cls_set = "train"
    video_folder = f"/data/xueruoyao/ActionAnalysis_dataset/MA-52/{cls_set}/*.mp4"
    pickle_dir = f"/data/xueruoyao/ActionAnalysis_dataset/MA-52/{cls_set}_instances"
    os.makedirs(pickle_dir, exist_ok=True)

    all_video_list = glob.glob(video_folder)
    chunk_size = max(len(all_video_list) // 16, 1)
    chunked_video_list = [
        all_video_list[i : i + chunk_size]
        for i in range(0, len(all_video_list), chunk_size)
    ]

    processes = 16
    with Pool(processes) as pool:
        for i, video_list in enumerate(chunked_video_list):
            device = i // 16
            pool.apply_async(predict, (video_list, i, device))
        pool.close()
        pool.join()

    # merge all pickle files
    all_bbox_dict = {}
    for pickle_file in os.listdir(pickle_dir):
        if not pickle_file.endswith(".pickle") or pickle_file == "all_instance.pickle":
            continue
        pickle_file_path = os.path.join(pickle_dir, pickle_file)
        with open(pickle_file_path, "rb") as fr:
            bbox_dict = pickle.load(fr)
            for key, value in bbox_dict.items():
                value = np.round(value).astype(int).tolist()
                value[0] -= 10
                value[1] -= 10
                value[2] += 10
                value[3] += 10
                bbox_dict[key] = value
            all_bbox_dict.update(bbox_dict)

    with open(os.path.join(pickle_dir, "all_instance.pickle"), "wb") as fr:
        pickle.dump(all_bbox_dict, fr)

    for pickle_file in os.listdir(pickle_dir):
        if pickle_file != "all_instance.pickle":
            os.remove(os.path.join(pickle_dir, pickle_file))
