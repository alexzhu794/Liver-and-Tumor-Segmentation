from monai.networks.nets import UNet
from monai.networks.layers import Norm

def get_net():
    '''这里使用标准的 3D U-Net'''
    model = UNet(
        spatial_dims=3,          # 这是一个 3D 网络
        in_channels=1,           # 输入通道数 (CT是灰度图，所以是1)
        out_channels=2,          # 输出通道数 (输出 2 层：[0]是背景概率，[1]是前景概率)；MONAI 的 DiceLoss 配合 Softmax 使用时，通常输出两个通道效果更稳定
        channels=(16, 32, 64, 128, 256),   # 网络宽度(每层通道数）。越深越宽（通道数越多），为了提取高级特征
        strides=(2, 2, 2, 2),    # 下采样步长（每次下采样缩小的倍数）。每次长宽高都缩小一半，一共缩小 2^4=16 倍
        num_res_units=2,         # 残差单元（每个层级里残差单元的数量）。防止网络太深梯度消失（ResNet的核心思想）
        norm=Norm.BATCH,         # 归一化。让训练更稳定，收敛更快（使用 Batch Normalization）
    ).to(device)

    return model