# LiTS-Cascade-Seg: 3D 肝脏与肿瘤级联分割系统

[](https://monai.io/)
[](https://pytorch.org/)
[](https://www.google.com/search?q=)

> 基于 PyTorch 和 U-Net 的肝脏肿瘤自动分割系统。利用深度学习技术，构建鲁棒性 Coarse-to-Fine 由粗到精的级联框架，帮助医生从 3D CT 影像中快速、高精度地识别病灶区域。

## 项目简介 (Introduction)

肝脏肿瘤分割是医学图像分析中的难点任务。主要挑战在于肿瘤与肝脏组织的对比度低，且存在极端的**类别不平衡**（肿瘤通常仅占总体积的 1% 不到）。

本项目实现了一套**两阶段级联策略**来解决上述问题：

1.  **阶段一 (Coarse):** 在完整的 3D CT 影像中定位并分割肝脏。
2.  **阶段二 (Fine):** 提取肝脏感兴趣区域 (ROI)，在该限定区域内精细分割肿瘤。

通过缩小搜索空间，该方法显著减少了背景区域（如胃、脾等）的假阳性误判，并大幅提升了微小病灶的分割精度。

## 核心特性 (Key Features)

  * **级联架构 (Cascade Architecture):** 实现了“定位-裁剪-精修”的流水线，有效解决小目标检测难题。
  * **MONAI 深度集成:** 利用 MONAI 框架实现高性能的 3D 数据加载、预处理（窗口化、重采样）及网络构建。
  * **滑动窗口推理 (Sliding Window):** 支持在有限显存（如 12GB）下对大尺寸 3D 影像（如 512x512x100+）进行高分辨率推理。
  * **鲁棒的数据流水线:**
      * 训练时采用基于 Patch 的动态平衡采样 (Balanced Patch Sampling)。
      * 自适应的 HU 值窗口化 (Windowing) 处理。
      * **坐标空间还原 (Invert):** 确保输出的 Mask 能与原始 CT 的空间元数据（Spacing/Direction）完美对齐。
  * **形态学后处理:** 集成连通域分析算法，自动去除微小噪点 (Small Connected Components)。


## 技术细节 (Methodology Details)

### 1\. 预处理 (Preprocessing)

  * **强度处理:** 软组织窗口化 (-175, 250 HU) 以突显肝脏结构。
  * **空间处理:** 统一重采样至各向同性体素间距 (1.0 x 1.0 x 2.0 mm)。
  * **采样策略:** 训练时使用随机 3D 切块 (96x96x32)，并保证前景/背景 1:1 采样。

### 2\. 级联推理逻辑 (Cascade Inference Logic)

1.  **全局搜索:** 肝脏模型在全图范围内预测肝脏掩码。
2.  **ROI 提取:** 根据肝脏掩码动态计算 3D 包围盒 (Bounding Box)，并增加安全边距。
3.  **局部精修:** 肿瘤模型仅在裁剪出的 ROI 区域内进行预测。
4.  **融合与清洗:** 将肿瘤预测结果贴回原图坐标，并移除小于 30 体素的孤立噪点。

## 技术栈 (Tech Stack)
1. **核心框架：** PyTorch
2. **医学AI框架：** MONAI (Medical Open Network for AI)
3. **数据读取库：** SimpleITK 和 NiBabel （用于读取 .nii.gz 或 .mha 格式的医学影像）
4. **环境：** Python，Jupyter Notebook/Lab，venv

## 快速开始 (Getting Started)

### 1\. 安装环境

```bash
git clone https://github.com/alexzhu794/Liver-and-Tumor-Segmentation.git
pip install -r requirements.txt
```

### 2\. 数据准备

使用 [LiTS (Liver Tumor Segmentation Challenge)](https://competitions.codalab.org/competitions/17094) 数据集。


### 3\. 模型训练

**步骤 1: 训练肝脏模型 (Stage 1)**

```bash
python train_liver.py
# 模型将保存至 output/models/best_model_liver.pth
```

**步骤 2: 训练肿瘤模型 (Stage 2)**

```bash
python train_tumor.py
# 模型将保存至 output/models/best_model_tumor.pth
```

*提示: 您可以在 `configs/config.py` 中调整 `BATCH_SIZE`, `LR`, `PATCH_SIZE` 等超参数。*

### 4\. 级联推理 (Cascade Inference)

对新数据进行端到端推理。该脚本会自动定位肝脏、裁剪 ROI、检测肿瘤并还原坐标。

```bash
python inference.py
```


### 5\. 性能评估 (Evaluation)

计算预测结果的 Dice, IoU, HD95 等指标。

```bash
python evaluate.py
```


## 致谢 (Acknowledgements)

  * [MONAI Framework](https://monai.io/)
  * [LiTS Challenge Dataset](https://competitions.codalab.org/competitions/17094)