ann_file_test = '/data/xueruoyao/ActionAnalysis_dataset/MA-52/annotations/test_subset_list_videos.txt'
ann_file_train = '/data/xueruoyao/ActionAnalysis_dataset/MA-52/annotations/train_val_list_videos_aug.txt'
ann_file_val = '/data/xueruoyao/ActionAnalysis_dataset/MA-52/annotations/val_list_videos.txt'
auto_scale_lr = dict(base_batch_size=64, enable=True)
data_root = '/data/xueruoyao/ActionAnalysis_dataset/MA-52/train_val'
data_root_test = '/data/xueruoyao/ActionAnalysis_dataset/MA-52/test_subset/'
data_root_val = '/data/xueruoyao/ActionAnalysis_dataset/MA-52/val'
dataset_type = 'VideoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=5, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
launcher = 'pytorch'
load_from = '/home/xueruoyao/MAC2025/MAR/work_dirs_train_val/videomae2-base_droppath0.05_continue/best_acc_f1_mean_epoch_4.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        depth=12,
        drop_path_rate=0.05,
        embed_dims=768,
        img_size=224,
        mlp_ratio=4,
        norm_cfg=dict(eps=1e-06, type='LN'),
        num_frames=16,
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        type='VisionTransformer'),
    cls_head=dict(
        average_clips='prob',
        in_channels=768,
        loss_cls=dict(type='CoarseFocalLoss'),
        num_classes=52,
        type='TimeSformerHead'),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    type='Recognizer3D')
optim_wrapper = dict(
    accumulative_counts=32,
    clip_grad=dict(max_norm=40, norm_type=2),
    constructor='LearningRateDecayOptimizerConstructor',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        backbone=dict(lr_mult=0.1),
        decay_rate=0.9,
        decay_type='layer_wise',
        num_layers=12))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=1,
        start_factor=0.1,
        type='LinearLR'),
    dict(
        T_max=4,
        begin=1,
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        eta_min=0,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '/data/xueruoyao/ActionAnalysis_dataset/MA-52/annotations/test_subset_list_videos.txt',
        data_prefix=dict(
            video='/data/xueruoyao/ActionAnalysis_dataset/MA-52/test_subset/'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=16,
                frame_interval=4,
                num_clips=10,
                test_mode=True,
                type='SampleFrames'),
            dict(scale=(
                224,
                448,
            ), train=False, type='DecordDecodeCrop'),
            dict(flip_ratio=1.0, type='Flip'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        metric_list=(
            'f1_mean',
            'top_k_accuracy',
            'mean_class_accuracy',
        ),
        type='AccMetric'),
    dict(
        out_file_path=
        '/home/xueruoyao/MAC2025/MAR/work_dirs_test_subset_continue/videomae2-base_droppath0.05_clip10_tta/result.pkl',
        type='DumpResults'),
]
test_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=16,
        frame_interval=4,
        num_clips=10,
        test_mode=True,
        type='SampleFrames'),
    dict(scale=(
        224,
        448,
    ), train=False, type='DecordDecodeCrop'),
    dict(flip_ratio=1.0, type='Flip'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=5, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file=
        '/data/xueruoyao/ActionAnalysis_dataset/MA-52/annotations/train_val_list_videos_aug.txt',
        data_prefix=dict(
            video='/data/xueruoyao/ActionAnalysis_dataset/MA-52/train_val'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=16,
                frame_interval=4,
                num_clips=1,
                type='SampleFrames'),
            dict(scale=(
                224,
                448,
            ), train=True, type='DecordDecodeCrop'),
            dict(type='ColorJitter'),
            dict(max_area_ratio=0.2, type='RandomErasing'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=16, frame_interval=4, num_clips=1, type='SampleFrames'),
    dict(scale=(
        224,
        448,
    ), train=True, type='DecordDecodeCrop'),
    dict(type='ColorJitter'),
    dict(max_area_ratio=0.2, type='RandomErasing'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '/data/xueruoyao/ActionAnalysis_dataset/MA-52/annotations/val_list_videos.txt',
        data_prefix=dict(
            video='/data/xueruoyao/ActionAnalysis_dataset/MA-52/val'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=16,
                frame_interval=4,
                num_clips=4,
                test_mode=True,
                type='SampleFrames'),
            dict(scale=(
                224,
                448,
            ), train=False, type='DecordDecodeCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    metric_list=(
        'f1_mean',
        'top_k_accuracy',
        'mean_class_accuracy',
    ),
    type='AccMetric')
val_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=16,
        frame_interval=4,
        num_clips=4,
        test_mode=True,
        type='SampleFrames'),
    dict(scale=(
        224,
        448,
    ), train=False, type='DecordDecodeCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/home/xueruoyao/MAC2025/MAR/work_dirs_test_subset_continue/videomae2-base_droppath0.05_clip10_tta'
