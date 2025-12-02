import os
import glob
import pandas as pd
import torch
import numpy as np
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
from monai.transforms import LoadImage, EnsureChannelFirst, Compose, AsDiscrete, RemoveSmallObjects
from monai.data import decollate_batch
from tqdm import tqdm

from configs.config import config


def evaluate(pred_dir, gt_dir, output_csv="evaluation_results.csv"):
    """
    离线评估脚本
    pred_dir: 存放预测结果 (.nii.gz) 的文件夹
    gt_dir: 存放真实标签 (Ground Truth) 的文件夹
    """

    print(f"预测目录: {pred_dir}")
    print(f"真值目录: {gt_dir}")

    # 1. 匹配文件
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.nii*")))     # 假设 inference 输出的文件名里包含 'volume-X'，我们通过 X 来匹配
    
    results = []

    # 2. 定义指标计算器
    dice_metric = DiceMetric(include_background=False, reduction="mean")     # include_background=False 表示只算前景 (肝脏/肿瘤)
    iou_metric = MeanIoU(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")    # percentile=95 即 HD95

    # 3. 数据加载器 (不需 Dataset/DataLoader，直接读取)
    loader = LoadImage(image_only=True)
    ensure_channel = EnsureChannelFirst()
    to_onehot = AsDiscrete(to_onehot=3)     # 转换为 One-Hot (0, 1, 2)

    for path in tqdm(pred_files, desc="Evaluating"):
        filename = os.path.basename(path)

        '''
        文件名匹配逻辑:
        假设 pred 是 "volume-5_cascade_seg.nii.gz"
        从中提取 "5"，然后去 GT 目录找 "segmentation-5.nii"
        '''
        case_id = filename.split('_')[0].replace('volume-', '')     # 提取 ID "5"
        gt_filename = f"segmentation-{case_id}.nii"
        gt_path = os.path.join(gt_dir, gt_filename)

        if not os.path.exists(gt_path):
            print(f"警告: 找不到对应的 GT 文件: {gt_path}，跳过。")
            continue

    # 4. 读取数据
    y_pred = loader(path)
    y_gt = loader(gt_path)

    # 确保都有通道维度 [C, D, H, W]
    y_pred = ensure_channel(y_pred)
    y_gt = ensure_channel(y_gt)

    # 放入 Batch 维度 [1, C, D, H, W] 以适配 Metric
    y_pred = y_pred.unsqueeze(0)     # 一个张量维度操作函数，核心作用是在指定位置插入一个长度为 1 的新维度，用于扩展张量的维度数
    y_gt = y_gt.unsqueeze(0)

    print(y_pred_oh.shape)

    # 5. 格式转换
    # 此时 y_pred 和 y_gt 里面的值是 0, 1, 2，需要把它变成 One-Hot [1, 3, D, H, W] 才能同时算肝脏和肿瘤
    y_pred_oh = to_onehot(y_pred)
    y_gt_oh = to_onehot(y_gt)

    print(y_pred_oh.shape)

    # 6. 计算指标
    dice_metric(y_pred=y_pred_oh, y=y_gt_oh)
    iou_metric(y_pred=y_pred_oh, y=y_gt_oh)
    hd95_metric(y_pred=y_pred_oh, y=y_gt_oh)    # HD95（如果某张图里没有肿瘤，HD95 可能会报错或返回 NaN）

    # 获取当前样本的数值
    # aggregate 返回的是 [3] (背景, 肝脏, 肿瘤)，通常只关心 肝脏(idx=1) 和 肿瘤(idx=2)
    dice_score = dice_metric.aggregate()     # aggregate() 对批量计算的指标结果进行 “聚合统计”
    iou_score = iou_metric.aggregate()
    hd95_score = hd95_metric.aggregate()

    # 重置状态
    dice_metric.reset()
    iou_metric.reset()
    hd95_metric.reset()

    # 记录结果
    res = {
        "Case": filename,
        # 肝脏指标
        "Liver_Dice": dice_score[1].item(),
        "Liver_IoU": iou_score[1].item(),
        "Liver_HD95": hd95_score[1].item(),
        # 肿瘤指标
        "Tumor_Dice": dice_score[2].item(),
        "Tumor_IoU": iou_score[2].item(),
        "Tumor_HD95": hd95_score[2].item(),
    }
    results.append(res)

    # 7. 保存结果
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print("最终评估报告")
    print("="*30)
    print(f"样本数: {len(df)}")
    print(f"肝脏平均 Dice: {df['Liver_Dice'].mean():.4f}")
    print(f"肿瘤平均 Dice: {df['Tumor_Dice'].mean():.4f}")
    print(f"肝脏平均 HD95: {df['Liver_HD95'].mean():.4f}")
    print(f"肿瘤平均 HD95: {df['Tumor_HD95'].mean():.4f}")
    print("="*30)
    print(f"详细结果已保存至: {output_csv}")


if __name__ == "__main__":
    PRED_DIR = "./output/preds" 
    GT_DIR = "./data/LiTS(train_test)/test_mask" 
    
    evaluate(PRED_DIR, GT_DIR)