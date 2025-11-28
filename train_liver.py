import os
import torch
from tqdm import tqdm 

from monai.utils import first, set_determinism
from monai.handlers.utils import from_engine
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import shutil

from src.dataset import get_loaders1, get_loaders2
from src.networks import get_net
from configs.config import config


def main():
    stage = 'liver'
    set_determinism(seed=config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'will use {device}')

    # 1. 准备数据
    train_loader, val_loader = get_loaders1(
        data_root = config.data_root, stage = stage
    )

    # 2. 准备模型
    model = get_net(device)

    # 3. 准备Loss和优化器
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    max_epochs = config.MAX_EPOCHS
    val_interval = config.VAL_INTERVAL
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    # 定义验证时的后处理
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])   # 预测结果：先取最大概率类别(argmax)，再转成 One-Hot
    post_label = Compose([AsDiscrete(to_onehot=2)])     # 真实标签：直接转成 One-Hot

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # 4. 训练循环
    for epoch in range(max_epochs):
        print(f"epoch {epoch + 1}/{max_epochs}")

        # --- Train ---
        model.train()
        epoch_loss = 0
        step = 0
        progress_bar = tqdm(train_loader, desc=f"Training ({stage})", unit="batch")

        for batch_data in progress_bar:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # --- Validation ---
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )

                    val_outputs = sliding_window_inference(
                        val_inputs, 
                        roi_size = config.PATCH_SIZE,
                        sw_batch_size = 1,    # 显存小的保险起见设为1，4090可以设4  
                        model
                    )


                    # 后处理与指标计算
                    # 1. 解包 (Batch -> List of samples)
                    al_outputs_list = decollate_batch(val_outputs)
                    val_labels_list = decollate_batch(val_labels)

                    # 2. 对每个样本做后处理 (转 One-Hot)
                    val_outputs = [post_pred(i) for i in val_outputs_list]
                    val_labels = [post_label(i) for i in val_labels_list]

                    # 3. 计算 Dice
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # 聚合结果
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                print(f"Validation Dice: {metric:.4f}")

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    save_path = os.path.join(config.MODEL_DIR, f"best_model_{stage}.pth")
                    torch.save(model.state_dict(), save_path)
                    print(f"新的最高分！模型已保存至: {save_path}")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )


if __name__ == "__main__":
    main()
