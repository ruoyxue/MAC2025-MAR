import cv2
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_video_frames(
    video_folder: Path, output_filename: str = "frame_count_histogram.png"
):
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    frame_counts = [
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for video_path in tqdm(video_folder.iterdir(), desc="Processing videos")
        if video_path.is_file()
        and video_path.suffix.lower() in video_extensions
        and (cap := cv2.VideoCapture(str(video_path))).isOpened()
    ]

    plt.figure(figsize=(10, 6))
    plt.hist(frame_counts, bins=30, edgecolor="black", color="skyblue")
    plt.xlabel("Frame Count")
    plt.ylabel("Number of Videos")
    plt.title(f'Histogram of Video Frame Counts in "{video_folder.name}"')
    plt.grid(axis="y", alpha=0.75)
    plt.savefig(output_filename)
    plt.close()
    print(f"Histogram saved to {output_filename}")


if __name__ == "__main__":

    video_folder_path = Path("/data/xueruoyao/ActionAnalysis_dataset/MA-52/test_subset")
    analyze_video_frames(video_folder_path, "frame_count_histogram_test_subset.png")
