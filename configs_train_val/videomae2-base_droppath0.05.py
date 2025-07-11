_base_ = ["../configs/_base_/default_runtime.py"]
load_from = r"/data/xueruoyao/ActionAnalysis_dataset/MA-52/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth"

# model settings
model = dict(
    type="Recognizer3D",
    backbone=dict(
        type="VisionTransformerRes",
        img_size=224,
        drop_path_rate=0.05,
        patch_size=16,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=16,
        norm_cfg=dict(type="LN", eps=1e-6),
    ),
    cls_head=dict(
        type="TimeSformerHead",
        num_classes=52,
        in_channels=768,
        average_clips="prob",
        loss_cls=dict(type="CoarseFocalLoss", use_f1_loss=True),
    ),
    data_preprocessor=dict(
        type="ActionDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape="NCTHW",
    ),
)

# dataset settings
dataset_type = "VideoDataset"
data_root = "/data/xueruoyao/ActionAnalysis_dataset/MA-52/train_val"
data_root_val = "/data/xueruoyao/ActionAnalysis_dataset/MA-52/val"
data_root_test = "/data/xueruoyao/ActionAnalysis_dataset/MA-52/test_subset/"
ann_file_train = "/data/xueruoyao/ActionAnalysis_dataset/MA-52/annotations/train_val_list_videos_aug.txt"
ann_file_val = "/data/xueruoyao/ActionAnalysis_dataset/MA-52/annotations/val_list_videos.txt"
ann_file_test = "/data/xueruoyao/ActionAnalysis_dataset/MA-52/annotations/test_subset_list_videos.txt"
file_client_args = dict(io_backend="disk")
train_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(type="UniformSample", clip_len=16, num_clips=1, test_mode=False),
    dict(type="DecordDecodeCrop", train=True, scale=(224, 448)),
    dict(type="ColorJitter"),
    dict(type="RandomErasing", max_area_ratio=0.2),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]
val_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(type="UniformSample", clip_len=16, num_clips=4, test_mode=True),
    dict(type="DecordDecodeCrop", train=False, scale=(224, 448)),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]
test_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(type="UniformSample", clip_len=16, num_clips=10, test_mode=True),
    dict(type="DecordDecodeCrop", train=False, scale=(224, 448)),
    # dict(type='Flip', flip_ratio=1.0),    # uncomment this when applying TTA.
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_test),
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

val_evaluator = dict(
    type="AccMetric", metric_list=("f1_mean", "top_k_accuracy", "mean_class_accuracy")
)
test_evaluator = val_evaluator

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=5, val_begin=1, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=5e-4, betas=(0.9, 0.999), weight_decay=0.05),
    accumulative_counts=32,
    constructor="LearningRateDecayOptimizerConstructor",
    paramwise_cfg=dict(
        decay_rate=0.9,
        decay_type="layer_wise",
        num_layers=12,
        backbone=dict(lr_mult=0.1),
    ),
    clip_grad=dict(max_norm=40, norm_type=2),
)
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=15,
        eta_min=0,
        by_epoch=True,
        begin=5,
        end=20,
        convert_to_iter_based=True,
    ),
]

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=5), logger=dict(interval=100)
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=64)
