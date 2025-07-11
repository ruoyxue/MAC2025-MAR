import os
import cv2
import pickle
from tqdm import tqdm
from shapely.geometry import polygon


def verify_bbox(pickle_path):
    with open(pickle_path, "rb") as fr:
        temp = pickle.load(fr)

    invalid_bboxes = {
        video_name: bbox for video_name, bbox in temp.items() if len(bbox) != 4
    }
    for video_name, bbox in invalid_bboxes.items():
        raise ValueError(f"Invalid bounding box for {video_name}: {bbox}")

    areas = [
        polygon.Polygon(
            [
                (bbox[0], bbox[1]),
                (bbox[2], bbox[1]),
                (bbox[2], bbox[3]),
                (bbox[0], bbox[3]),
            ]
        ).area
        for _, bbox in temp.items()
        if len(bbox) == 4
    ]

    if areas:
        print(f"minimum area: {min(areas)}")
    else:
        raise ValueError("No valid bounding boxes found.")


cls_set = "test"
video_path = f"/data/xueruoyao/ActionAnalysis_dataset/MA-52/{cls_set}"
pickle_path = f"/data/xueruoyao/ActionAnalysis_dataset/MA-52/{cls_set}_instances/all_instance.pickle"
vis_image_save_root = (
    f"/data/xueruoyao/ActionAnalysis_dataset/MA-52/{cls_set}_instances/visualize"
)
os.makedirs(vis_image_save_root, exist_ok=True)

verify_bbox(pickle_path)

with open(pickle_path, "rb") as fr:
    temp = pickle.load(fr)

for path_video in tqdm(os.listdir(video_path)):
    video_name = os.path.basename(path_video)
    cap = cv2.VideoCapture(os.path.join(video_path, path_video))
    ret, frame = cap.read()
    bbox = temp[video_name]

    frame = cv2.rectangle(
        frame,
        [int(bbox[0]), int(bbox[1])],
        [int(bbox[2]), int(bbox[3])],
        (0, 255, 0),
        3,
    )

    cv2.imwrite(os.path.join(vis_image_save_root, f"{video_name}.jpg"), frame)
    cap.release()
