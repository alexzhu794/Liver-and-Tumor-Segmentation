'''
这个脚本需要它能够：
1. 读取一张（或多张）新的 CT。
2. 加载训练好的模型（肝脏 或 肿瘤）。
3. 预测。
4. 还原坐标 (Invert)。
5. 保存为 .nii.gz 文件。
'''

import os
import glob
import torch
import numpy as np
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader, decollate_batch
from monai.transforms import (
    AsDiscreted,
    Compose,
    Invertd,
    SaveImaged,
)
from monai.utils import first

from configs.config import config
from src.transforms import get_inference_trans
from src.networks import get_net


def get_bbox(mask, margin=5):
    """
    核心辅助函数：根据肝脏 Mask 计算包围盒 (Bounding Box)。
    mask: Shape [C, D, H, W] or [D, H, W]
    margin: 在包围盒周围多留一点边界，防止切得太死
    """
    if isinstance(mask, torch.Tensor):    # 内置判断函数: isinstance(对象, 类型/类型元组)
        mask = mask.cpu().numpy()

        # 找到所有值 > 0 (即肝脏) 的索引
        # mask 可能是 [1, D, H, W]，我们取 [0] 变成 3D
        any_liver = np.where(mask[0] > 0)

    if len(any_liver[0]) == 0:
        return None    # 没找到肝脏

    # 计算最小和最大坐标
    z_min, z_max = np.min(any_liver[0]), np.max(any_liver[0])
    y_min, y_max = np.min(any_liver[1]), np.max(any_liver[1])
    x_min, x_max = np.min(any_liver[2]), np.max(any_liver[2])    

    # 加上 margin，同时防止越界
    d, h, w = mask.shape[1:]
    z_min = max(0, z_min - margin)
    z_max = min(d, z_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(h, y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(w, x_max + margin)

    return (z_min, z_max, y_min, y_max, x_min, x_max)    


def run_inference(input_dir, output_dir, liver_model_path, tumor_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"启动级联推理 | 设备: {device}")

    # 1. 准备数据加载器
    # 使用基础的推理 transforms (读取 -> 统一方向 -> 统一间距 -> 归一化)
    images = sorted(glob.glob(os.path.join(input_dir, "volume-*.nii")))
    if not images: images = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))   # 兼容

    data_dicts = [{"image": image} for image in images]
    transforms = get_inference_trans(spatial_size=config.PATCH_SIZE, intensity_range=config.INTENSITY_RANGE)
    ds = Dataset(data=data_dicts, transform=transforms)
    loader = DataLoader(ds, batch_size=1, num_workers=0)

    # 2. 加载两个模型
    print("正在加载双模型...")
    liver_model = get_net(device)
    liver_model.load_state_dict(torch.load(liver_model_path, map_location=device))
    liver_model.eval()

    tumor_model = get_net(device)
    tumor_model.load_state_dict(torch.load(tumor_model_path, map_location=device))
    tumor_model.eval()

    # 3. 定义最终的还原 (Invert) 逻辑
    # 我们只在最后一步把最终结果还原回原始空间
    inverter = Invertd(
        keys="pred",
        transform=transforms,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        nearest_interp=False,
        to_tensor=True,
    )

    saver = SaveImaged(
        keys="pred",
        meta_keys="pred_meta_dict",
        output_dir=output_dir,
        output_postfix="cascade_seg",
        resample=False,
        output_dtype=torch.uint8
    )

    os.makedirs(output_dir, exist_ok=True)

    # 4. 级联循环
    with torch.no_grad():       # 只要不是在 loss.backward()（反向传播）的时候，凡是预测、验证、测试，必须加这句.不需要反思“为什么错了”, 否则现存会爆炸
        for i, data in enumerate(loader):
            inputs = data["image"].to(device)    # 全图 CT
            filename = data["image_meta_dict"]["filename_or_obj"][0]
            print(f"[{i+1}/{len(loader)}] 处理中: {os.path.basename(filename)}")

            # 阶段一：全图预测肝脏
            # 使用滑动窗口，因为整图可能很大
            liver_logits = sliding_window_inference(inputs, config.PATCH_SIZE, 4, liver_model)
            # 转为 0/1 Mask
            liver_mask = torch.argmax(liver_logits, dim=1, keepdim=True)

            # 阶段二：根据肝脏 Mask 裁剪 (Crop)
            bbox = get_bbox(liver_mask, margin=10)
            
            # 创建一个全黑的肿瘤 Mask，大小和原图一样
            final_tumor_mask = torch.zeros_like(liver_mask)

            if bbox is None:
                print(f"警告: 未检测到肝脏，跳过肿瘤检测。")
            else:
                z1, z2, y1, y2, x1, x2 = bbox
                # 关键：只把肝脏区域切出来 (ROI)
                liver_roi = inputs[:, :, z1:z2, y1:y2, x1:x2]
                
                print(f"肝脏区域 ROI 大小: {tuple(liver_roi.shape)}")

                # 阶段三：在 ROI 里预测肿瘤
                # 即使是切出来的 ROI，也可能比 patch_size 大，所以依然用 sliding_window
                tumor_logits_roi = sliding_window_inference(liver_roi, config.PATCH_SIZE, 4, tumor_model)
                tumor_mask_roi = torch.argmax(tumor_logits_roi, dim=1, keepdim=True)

                # 阶段四：把 ROI 的结果贴回大图 (Paste)
                final_tumor_mask[:, :, z1:z2, y1:y2, x1:x2] = tumor_mask_roi
                
                # 可选：用肝脏 Mask 再次过滤，确保肿瘤一定在肝脏内（消除边缘误判）
                final_tumor_mask = final_tumor_mask * liver_mask

            # 阶段五：合并结果(这里的策略取决于想要什么样的输出)
            # 策略 A: 0=背景, 1=肝脏, 2=肿瘤 (标准 LiTS 格式)
            # 目前的 liver_mask 里肝脏是 1，final_tumor_mask 里肿瘤是 1
            combined_mask = liver_mask.clone()
            combined_mask[final_tumor_mask == 1] = 2 # 把肿瘤的位置盖在肝脏上，赋值为 2
            
            # 放入字典，准备 Invert
            data["pred"] = combined_mask

            # 阶段六：还原坐标并保存
            # 解包
            data_list = decollate_batch(data)
            for d in data_list:
                d = inverter(d)   # 变回原始分辨率、方向
                d = AsDiscreted(keys="pred", argmax=False)(d)   # 已经是整数了，不需要 argmax，只需格式整理
                saver(d)   # 保存

    print(f"级联推理完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    # 示例调用：
    # python inference.py 
    
    # 实际项目中建议用 argparse，这里为了演示直接写死路径
    # 请确保您已经训练好了两个模型
    INPUT_DIR = config.data_root + "/LiTS(train_test)/test_CT"    # 假设您有测试数据
    OUTPUT_DIR = "D:/工程项目/肝脏和肿瘤分割/output/preds"
    LIVER_MODEL = "./output/models/best_model_liver.pth"
    TUMOR_MODEL = "./output/models/best_model_tumor.pth"
    
    if os.path.exists(LIVER_MODEL) and os.path.exists(TUMOR_MODEL):
        run_inference(INPUT_DIR, OUTPUT_DIR, LIVER_MODEL, TUMOR_MODEL)
    else:
        print("请先训练好 liver 和 tumor 两个模型再运行此脚本！")