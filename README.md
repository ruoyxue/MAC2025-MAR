# Official Codes for The MAC 2025 Grand Challenge Track 1

## 1. Data preparation

- Downlaod Track1 dataset in data or other floder as following:

```
-data
  |-annotations
  |-train
  |-train_val (merge train and val)
  |-val
  |-test
```

- You need to prepare the virtual environment as follows:

```
conda create --name mar
conda activate mar
pip install -r requirements.txt
```

### 1.1 Balance data
Those videos less than 200 are copied several times to mitigate the severe data imbalance, which is a commonly used trick.

```
python data_process/data_aug.py
```

### 1.2 Instance Detection

**Pretrained people detector** is employed to locate the interviewed person. Specifically, the bounding box of person instance is detected by YOLOv8m. All the bounding boxes are saved in pickle format.

```
python data_process/predict_video.py
```

## 2. Training

- Before training your model following our tutorial, please make sure that **the path of instance** is right in line 68-73 of *mmaction/datasets/video_dataset.py*.

- Make sure **the path of dataset** in config file. 

```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='0,1,2,3' bash tools/dist_train.sh configs_train_val/videomae2-base_baseline.py 4 --work-dir <your_work_dir_path>
```

## 3. Testing Inference

With corresponding configuration, you can inference model forward and save the results in pickle format.

```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='0,1' bash tools/dist_test.sh configs_train_val/videomae2-base_baseline.py MAC-2025_MAR/checkpoints/videomae2-base_baseline/pth 2 work_dirs_test_subset/videomae2-base_baseline_clip10

(optional: TTA reference)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='0,1' bash tools/dist_test.sh configs_train_val/videomae2-base_baseline.py MAC-2025_MAR/checkpoints/videomae2-base_baseline/pth 2 work_dirs_test_subset/videomae2-base_baseline_clip10_tta

(optinal: merge TTA results)
python tta.py
```

## 4. Submission

More important, model performance could be improved further by weighting the different predictions via Model ensembling, a simple yet useful trick.

```
python assemble_models.py
```

Our final submission is **prediction.csv** and our checkpoints can be downloaded from https://huggingface.co/ruoyxue/MAC2025_MAR/tree/main.
