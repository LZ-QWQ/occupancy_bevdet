# Copyright (c) Phigent Robotics. All rights reserved.

# epoch2_ema ===> mIoU of 6019 samples: 44.76
# epoch4_ema ===> mIoU of 6019 samples: 45.5

work_dir = "/data/work_dirs/bevdet-occ-intenimage_B_custom_decay-4d-stereo-672x1408-24e-labelsmooth_0.0001-load-CBGS"
find_unused_parameters = False

_base_ = ["../../_base_/datasets/nus-3d.py", "../../_base_/default_runtime.py"]
# # For nuScenes we usually do 10-class detection
# class_names = [
#     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
#     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
# ]
# Actually we have 17 clesses for nuScenes
class_names = [
    "void",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
    "free",
]

data_config = {
    "cams": ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
    "Ncams": 6,
    "input_size": (672, 1408),
    "src_size": (900, 1600),
    # Augmentation
    "resize": (-0.06, 0.11),
    "rot": (-5.4, 5.4),
    "flip": True,
    "crop_h": (0.0, 0.0),
    # "resize_test": 0.00, # deprecated in TTA
}

# Model
grid_config = {
    "x": [-40, 40, 0.4],
    "y": [-40, 40, 0.4],
    "z": [-1, 5.4, 0.4],
    "depth": [1.0, 45.0, 0.5],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 32

multi_adj_frame_id_cfg = (1, 1 + 1, 1)

pretrained = "https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_b_fpn_3x_coco.pth"
model = dict(
    type="BEVStereo4DOCC",
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    img_backbone=dict(
        _delete_=True,
        type="InternImage",  # ......这玩意忘了用那个特殊的优化器，这一版使用了！
        core_op="DCNv3",
        channels=112,
        depths=[4, 4, 21, 4],
        groups=[7, 14, 28, 56],
        mlp_ratio=4.0,
        drop_path_rate=0.4,
        norm_layer="LN",
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=True,
        with_cp=True,
        out_indices=(2, 3),
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
        return_stereo_feat=True,  # 看起来要返回第一个stage的特征
    ),
    img_neck=dict(
        type="FPN_LSS",
        in_channels=448 + 896,
        out_channels=512,
        # with_cp=False,
        extra_upsample=None,
        input_feature_index=(0, 1),
        scale_factor=2,
    ),
    img_view_transformer=dict(
        type="LSSViewTransformerBEVStereo",
        grid_config=grid_config,
        input_size=data_config["input_size"],
        in_channels=512,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=False,
        loss_depth_weight=0.05,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96, stereo=True, bias=5.0),
        downsample=16,
    ),
    img_bev_encoder_backbone=dict(
        type="CustomResNet3D",
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg)) + 1),
        num_layer=[1, 2, 4],
        with_cp=False,
        num_channels=[numC_Trans, numC_Trans * 2, numC_Trans * 4],
        stride=[1, 2, 2],
        backbone_output_ids=[0, 1, 2],
    ),
    img_bev_encoder_neck=dict(type="LSSFPN3D", in_channels=numC_Trans * 7, out_channels=numC_Trans),
    pre_process=dict(
        type="CustomResNet3D",
        numC_input=numC_Trans,
        with_cp=False,
        num_layer=[
            1,
        ],
        num_channels=[
            numC_Trans,
        ],
        stride=[
            1,
        ],
        backbone_output_ids=[
            0,
        ],
    ),
    # loss_occ=dict(
    #     type='CrossEntropyLoss',
    #     use_sigmoid=False,
    #     loss_weight=1.0),
    loss_occ=dict(
        type="CrossEntropyLossLableSmoothing",
        use_sigmoid=False,
        loss_weight=1.0,
        label_smoothing=0.0001,
        class_weight=[
            1.02,
            1.02,
            1.02,
            1.02,
            1.02,
            1.02,
            1.02,
            1.02,
            1.02,
            1.02,
            1.02,
            1.02,
            1.02,
            1.02,
            1.02,
            1.02,
            1.02,
            0.66,
        ],  # others ...... free
    ),
    use_mask=True,
)

# Data
dataset_type = "NuScenesDatasetOccpancy"
data_root = "data/nuscenes-test/"
file_client_args = dict(backend="disk")

bda_aug_conf = dict(rot_lim=(-0.0, 0.0), scale_lim=(1.0, 1.0), flip_dx_ratio=0.5, flip_dy_ratio=0.5)

train_pipeline = [
    dict(type="PrepareImageInputs", is_train=True, data_config=data_config, sequential=True),
    dict(type="LoadOccGTFromFile"),
    dict(type="LoadAnnotationsBEVDepth", bda_aug_conf=bda_aug_conf, classes=class_names, is_train=True),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5, file_client_args=file_client_args),
    dict(type="PointToMultiViewDepth", downsample=1, grid_config=grid_config),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="Collect3D", keys=["img_inputs", "gt_depth", "voxel_semantics", "mask_lidar", "mask_camera"]),
]

# TTA !!!!!!
test_pipeline = [
    dict(
        type="OccTTA_TestPipline",  # PrepareImageInputs 和 LoadAnnotationsBEVDepth 以TTA的形式嵌入
        data_config=data_config,
        flip_view_num=2,
        scale_view_list=[-0.05, 0, 0.08],
        flip_bev_xy=[(False, False), (True, True)],  # (True, False), (False, True)
        transforms=[
            dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
            dict(type="Collect3D", keys=["img_inputs"]),  # 'points',
        ],
    )
]

# test_pipeline = [
#     dict(type="PrepareImageInputs", data_config=data_config, sequential=True),
#     dict(type="LoadAnnotationsBEVDepth", bda_aug_conf=bda_aug_conf, classes=class_names, is_train=False),
#     dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5, file_client_args=file_client_args),
#     dict(
#         type="MultiScaleFlipAug3D",
#         img_scale=(1333, 800),
#         pts_scale_ratio=1,
#         flip=False,
#         transforms=[
#             dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
#             dict(type="Collect3D", keys=["points", "img_inputs"]),
#         ],
#     ),
# ]

input_modality = dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype="bevdet4d",
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(pipeline=test_pipeline, ann_file=data_root + "bevdetv2-nuscenes_infos_test_2.pkl")

data = dict(
    samples_per_gpu=4,  # with 8 A100
    workers_per_gpu=8,
    train=dict(
        type="CBGSDatasetOcc",
        dataset=dict(
            data_root=data_root,
            ann_file=data_root + "bevdetv2-nuscenes_infos_train.pkl",
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d="LiDAR",
        ),
    ),
    val=test_data_config,
    test=test_data_config,
)

# for key in ['val', 'train', 'test']:
#     data[key].update(share_data_config)
for key in ["val", "test"]:
    data[key].update(share_data_config)
data["train"]["dataset"].update(share_data_config)

# Optimizer
optimizer = dict(
    type="AdamW",
    lr=2e-4,
    weight_decay=1e-2,
    constructor="CustomLayerDecayOptimizerConstructor",
    paramwise_cfg=dict(num_layers=33, layer_decay_rate=1.0, depths=[4, 4, 21, 4]),
)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[13, 21],
)
runner = dict(type="EpochBasedRunner", max_epochs=24)

custom_hooks = [
    dict(
        type="MEGVIIEMAHook",
        init_updates=10560,
        priority="NORMAL",
    ),
    dict(
        type="SyncbnControlHook",
        syncbn_start_epoch=0,
    ),
]

auto_resume = True  # 非常尴尬，因为失误只能从resume了
# load_from = "./ckpts/bevdet-InternImageB-LS_0.0001-epoch_19_ema.pth"
# backbone_init_weight_after_load = True  # backbone用别的加载
# fp16 = dict(loss_scale='dynamic')

evaluation = dict(interval=2, pipeline=test_pipeline)  # eval_pipeline , 这个地方真的需要pipline吗  # 貌似有bug，玩个屁....
