from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    MapLabelValued,
    EnsureTyped,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import numpy as np
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob



# ATTENTION! map_trans是函数内部的局部变量，如果train/val_transforms定义在外部的话无法调用函数内部的变量！
def get_transforms(stage='liver', mode='train'):
    if stage == 'liver':
        # 阶段一：把 2(肿瘤) 变成 1(肝脏)，实现 0 vs 1
        map_trans = MapLabelValued(
            keys=["label"], orig_labels=[0, 1, 2], target_labels=[0, 1, 1]
        )
    elif stage == 'tumor':
        # 阶段二：把 1(肝脏) 变成 0(背景)，实现 0 vs 2
        map_trans = MapLabelValued(
            keys=["label"], orig_labels=[0, 1, 2], target_labels=[0, 0, 1]
        )
    else:
        raise ValueError(f"Stage 必须是 'liver' 或 'tumor'，你传了: {stage}")
    
    # ------------
    if mode == 'train':
        return Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),    # 增加通道维度
        map_trans,
        Orientationd(keys=["image", "label"], axcodes="RAS"),      # 统一方向, RAS (Right, Anterior, Superior) 是国际标准方向
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),       # 重采样 (Resampling); 肝脏小肿瘤建议(1.0, 1.0, 2.0)，否则太粗糙
        ScaleIntensityRanged(         # CT值窗口化 (Windowing) + 归一化
            keys=["image"],
            a_min=-175,     # 器官的特征窗口
            a_max=250,
            b_min=0.0,      # 把剩下的值压缩到 0 到 1 之间
            b_max=1.0,
            clip=True,      # 超出范围的直接截断
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),      # 切掉黑边
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(64, 64, 64),      # 防止显卡吃不下图，切出来的块大小是 64x64x64 像素
            pos=1,       # 正负样本均衡。保证切出来的块里，有一半是包含标签（脾脏/肝脏）的，一半是背景。
            neg=1,
            num_samples=4,       # 每张 CT 图切出 4 个这样的小块送入 batch
            image_key="image",
            image_threshold=0,
        ),
        EnsureTyped(keys=["image", "label"]),
        # user can also add other random transforms
        # RandAffined(
        #     keys=['image', 'label'],
        #     mode=('bilinear', 'nearest'),
        #     prob=1.0, spatial_size=(96, 96, 96),
        #     rotate_range=(0, 0, np.pi/15),
        #     scale_range=(0.1, 0.1, 0.1)),
    ]
)

    elif mode == 'val':
        return Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        map_trans,
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        EnsureTyped(keys=["image", "label"]),
    ]
)    #验证的时候，需要对整个肝脏进行评估，不能只切一小块看。通常在验证阶段，使用滑动窗口推断（Sliding Window Inference），而不是随机切块。


def get_inference_trans(spatial_size=(96, 96, 32), intensity_range=(-175, 250)):
    """
    专门用于推理的预处理。
    注意：没有 Label 相关操作，没有随机增强。
    """
    keys = ["image"]
    return Compose ([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        # 注意：推理时也需要重采样，否则模型看不懂
        Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0), mode="bilinear"), 
        ScaleIntensityRanged(
            keys=keys, 
            a_min=intensity_range[0], 
            a_max=intensity_range[1], 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        CropForegroundd(keys=keys, source_key="image", allow_smaller=True),
        EnsureTyped(keys=keys),
    ])