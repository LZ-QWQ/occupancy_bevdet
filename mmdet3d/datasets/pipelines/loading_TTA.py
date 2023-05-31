# Copyright (c) OpenMMLab. All rights reserved.
# This file using for test time augmentation for occupancy prediction task
# mainly modified from training augmentation

import os

import mmcv
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes
from ..builder import PIPELINES
import warnings
from copy import deepcopy

from .compose import Compose

from typing import List, Tuple


@PIPELINES.register_module()
class OccTTA_TestPipline(object):
    """
    用于实现TTA，该pipline里嵌套了图像处理增强和BEV特征增强的pipline。
    注意!!!用这玩意速度真的非常慢

    flip_view_num: 六视角图像自由翻转多少次（6视角各自随机，可能会被seed给固定
    scale_view_list: 六视角图像缩放
    flip_bev_xy: BEV图像增强（暂时只支持x,y,xy三种翻转呢）
    """

    def __init__(
        self,
        transforms,
        data_config,
        flip_view_num: int = 2,
        scale_view_list: List[float] = [-0.05, 0, 0.08],
        flip_bev_xy: List[Tuple[bool, bool]] = [(False, False), (True, True)],  # (True, False), (False, True)
    ):
        self.prepare_inputs = PrepareImageInputs_TTA(data_config, sequential=True)
        self.load_bevdepth = LoadAnnotationsBEVDepth_TTA()
        self.transforms = Compose(transforms)

        self.flip_view_num = flip_view_num
        self.scale_view_list = scale_view_list
        self.flip_bev_xy = flip_bev_xy

    def __call__(self, results):
        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            dict: The result dict contains the data that is augmented with
                different scales and flips.
        """
        aug_data = []

        for scale in self.scale_view_list:
            for flip_xy in self.flip_bev_xy:  # (flip_dx, flip_dy)
                for _ in range(self.flip_view_num):
                    _results = deepcopy(results)
                    _results = self.prepare_inputs(_results, scale=scale)
                    _results = self.load_bevdepth(_results, *flip_xy)

                    data = self.transforms(_results)
                    data["flip_xy"] = flip_xy
                    aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        # print(aug_data_dict.keys())
        return aug_data_dict


def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


class PrepareImageInputs_TTA(object):
    def __init__(
        self,
        data_config,
        sequential=False,
    ):
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential

    def get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims, crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        cam_names = self.data_config["cams"]
        return cam_names

    def sample_augmentation(self, H, W, random_flip=True, scale=None):
        fH, fW = self.data_config["input_size"]
        resize = float(fW) / float(W)
        if scale is not None:
            resize += scale
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(self.data_config["crop_h"])) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = np.random.choice([False, True]) if random_flip else False
        rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor_transforms(self, cam_info, cam_name):
        w, x, y, z = cam_info["cams"][cam_name]["sensor2ego_rotation"]
        # sweep sensor to sweep ego
        sensor2ego_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = torch.Tensor(cam_info["cams"][cam_name]["sensor2ego_translation"])
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran
        # sweep ego to global
        w, x, y, z = cam_info["cams"][cam_name]["ego2global_rotation"]
        ego2global_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
        ego2global_tran = torch.Tensor(cam_info["cams"][cam_name]["ego2global_translation"])
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def get_inputs(self, results, random_flip=True, scale=None):
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results["cam_names"] = cam_names
        canvas = []
        for cam_name in cam_names:
            cam_data = results["curr"]["cams"][cam_name]
            filename = cam_data["data_path"]
            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data["cam_intrinsic"])

            sensor2ego, ego2global = self.get_sensor_transforms(results["curr"], cam_name)
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(H=img.height, W=img.width, random_flip=random_flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = self.img_transform(
                img, post_rot, post_tran, resize=resize, resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate
            )

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert "adjacent" in results
                for adj_info in results["adjacent"]:
                    filename_adj = adj_info["cams"][cam_name]["data_path"]
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(img_adjacent, resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            for adj_info in results["adjacent"]:
                post_trans.extend(post_trans[: len(cam_names)])
                post_rots.extend(post_rots[: len(cam_names)])
                intrins.extend(intrins[: len(cam_names)])

                # align
                for cam_name in cam_names:
                    sensor2ego, ego2global = self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)

        imgs = torch.stack(imgs)

        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        results["canvas"] = canvas
        return (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans)

    def __call__(self, results, scale=None):  # random_flip=True
        # 不知道咋写好，先把random_flip强制为True
        results["img_inputs"] = self.get_inputs(results, random_flip=True, scale=scale)
        return results


class LoadAnnotationsBEVDepth_TTA(object):
    def bev_transform(self, rotate_angle, scale_ratio, flip_dx, flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0], [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        return rot_mat

    def __call__(self, results, flip_dx, flip_dy):
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        bda_rot = self.bev_transform(0.0, 1.0, flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot

        imgs, rots, trans, intrins = results["img_inputs"][:4]
        post_rots, post_trans = results["img_inputs"][4:]
        results["img_inputs"] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot)
        if "voxel_semantics" in results:
            if flip_dx:
                results["voxel_semantics"] = results["voxel_semantics"][::-1, ...].copy()
                results["mask_lidar"] = results["mask_lidar"][::-1, ...].copy()
                results["mask_camera"] = results["mask_camera"][::-1, ...].copy()
            if flip_dy:
                results["voxel_semantics"] = results["voxel_semantics"][:, ::-1, ...].copy()
                results["mask_lidar"] = results["mask_lidar"][:, ::-1, ...].copy()
                results["mask_camera"] = results["mask_camera"][:, ::-1, ...].copy()
        return results
