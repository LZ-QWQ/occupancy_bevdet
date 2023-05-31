# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from .builder import DATASETS

@DATASETS.register_module()
class CBGSDataset(object):
    """A wrapper of class sampled dataset with ann_file path. Implementation of
    paper `Class-balanced Grouping and Sampling for Point Cloud 3D Object
    Detection <https://arxiv.org/abs/1908.09492.>`_.

    Balance the number of scenes under different classes.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be class sampled.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.sample_indices = self._get_sample_indices()
        # self.dataset.data_infos = self.data_infos
        if hasattr(self.dataset, 'flag'):
            self.flag = np.array(
                [self.dataset.flag[ind] for ind in self.sample_indices],
                dtype=np.uint8)

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx in range(len(self.dataset)):
            sample_cat_ids = self.dataset.get_cat_ids(idx)
            for cat_id in sample_cat_ids:
                class_sample_idxs[cat_id].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []

        frac = 1.0 / len(self.CLASSES)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds,
                                               int(len(cls_inds) *
                                                   ratio)).tolist()
        return sample_indices

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        ori_idx = self.sample_indices[idx]
        return self.dataset[ori_idx]

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.sample_indices)

@DATASETS.register_module()
class CBGSDatasetOcc(object):
    """A wrapper of class sampled dataset with ann_file path. Implementation of
    paper `Class-balanced Grouping and Sampling for Point Cloud 3D Object
    Detection <https://arxiv.org/abs/1908.09492.>`_.

    Balance the number of scenes under different classes.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be class sampled.
    """

    def __init__(self, dataset):
        print("\nCBGSDatasetOcc Starts\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.dataset = dataset
        self.CLASSES = self.dataset.CLASSES
        print("Classes num:", self.CLASSES)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.sample_indices = self._get_sample_indices()
        # self.dataset.data_infos = self.data_infos
        if hasattr(self.dataset, 'flag'):
            self.flag = np.array(
                [self.dataset.flag[ind] for ind in self.sample_indices],
                dtype=np.uint8)

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        # class_sample_idxs[dict] -key:类别id -value:[list]存入的是 “存在这个类别id的样本的idx” 
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        del class_sample_idxs[17]

        ## Collect Sample and Compute Distribution
        target_size = len(self.dataset) * 2 # 数据增强后的目标样本总数
        print("Target Data Size:", target_size)

        for idx in range(len(self.dataset)): # 500 有为 len(self.dataset)
            if idx % 1000 == 0: 
                print("- Sample", idx, "/", len(self.dataset))
            
            sample_cat_ids = self.dataset.get_cat_ids(idx)
            for cat_id in sample_cat_ids:
                if cat_id != 17:  # 如果在这里 del17 就不会爆内存了
                    class_sample_idxs[cat_id].append(idx)
        print("Class Sample Num:", [len(class_sample_idxs[idx]) for idx in class_sample_idxs.keys()])

        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        print("Duplicated Samples:", duplicated_samples)
            
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }
        print("Class Distribution:", class_distribution.values())

        frac = 1.0 / (len(self.CLASSES) - 1)

        ## Sample
        sample_indices = []
        for cls_inds in list(class_sample_idxs.values()):
            sample_num = int(frac * target_size) + 1
            sample_indices += np.random.choice(cls_inds, sample_num).tolist()
        print("Sample Indices Len:", len(sample_indices))

        ## Recompute Distribution
        # print("\n[After CBGS]")
        # class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        # del class_sample_idxs[17]

        # for idx in range(len(sample_indices)):
        #     if idx % 1000 == 0: 
        #         print("- Recompute", idx, "/", len(sample_indices))
            
        #     sample_cat_ids = self.dataset.get_cat_ids(sample_indices[idx])
        #     for cat_id in sample_cat_ids:
        #         if cat_id != 17: # 不存入标签为17的节点就不会爆内存了
        #             class_sample_idxs[cat_id].append(sample_indices[idx])
        # print("Class Sample Num:", [len(class_sample_idxs[idx]) for idx in class_sample_idxs.keys()])
        
        # duplicated_samples = sum(
        #     [len(v) for _, v in class_sample_idxs.items()])
        # print("Duplicated Samples:", duplicated_samples)
            
        # class_distribution = {
        #     k: len(v) / duplicated_samples
        #     for k, v in class_sample_idxs.items()
        # }
        # print("Class Distribution:", class_distribution.values())
        # exit()
        return sample_indices

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        ori_idx = self.sample_indices[idx]
        return self.dataset[ori_idx]

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.sample_indices)