import os
import glob
from sklearn.model_selection import train_test_split
from monai.data import CacheDataset, DataLoader, Dataset
from src.transforms import get_transforms

# image_dir = os.path.join(data_root, 'LiTS(train_test)', 'train_CT')
# label_dir = os.path.join(data_root, 'LiTS(train_test)', 'train_mask')

def get_loaders1(data_root, stage='liver', batch_size=2, val_ratio=0.2, cache=True):     # 'liver'是在没有告知的情况下默认的
    """
    【原始数据集未划分，只使用train数据划分】
    准备数据的核心函数

    Args:
        _dir (str): 数据目录
        stage (str): 'liver' 或 'tumor'
        batch_size (int): 显存大就调大，显存小就调小
        val_ratio (float): 验证集比例，0.2 表示 20% 做验证
        cache (bool): 是否使用 CacheDataset (吃内存但快)
    """

    # 1. 搜寻所有文件
    images = sorted(glob.glob(os.path.join(data_root, 'LiTS(train_test)', 'train_CT', "volume-*.nii")))
    labels = sorted(glob.glob(os.path.join(data_root, 'LiTS(train_test)', 'train_mask', "segmentation-*.nii")))

    if len(images) == 0:
        raise ValueError("没找到 .nii 文件，请检查路径")

    # 2. 组装成字典列表 [{'image': path1, 'label': path1}, ...]
    dicts = [
        {"image:": img_name, "label": lbl_name}
        for img_name, lbl_name in zip(images, labels)
    ]

    # 3. 划分训练集和验证集
    train_fl, val_fl = train_test_split(dicts, test_size = val_ratio, random_state=42)
    print(f"数据准备完成: 总数 {len(dicts)} | 训练集 {len(train_fl)} | 验证集 {len(val_fl)}")

    # 4. 准备 Transforms
    train_transform = get_transforms(stage=stage, mode='train')
    val_transform = get_transforms(stage=stage, mode='val')

    # 5. 定义 Dataset
    if cache:
        # 生产环境/显存够大用这个
        # cache_rate=1.0 表示全存，如果爆内存可以改成 0.5 或 0.1
        train_ds = CacheDataset(data=train_fl, transform=train_transform, cache_rate=0.2, num_workers=4)
        val_ds = CacheDataset(data=val_fl, transform=val_transform, cache_rate=0.2, num_workers=4)
    else:
        # 调试用，或者内存实在不够用这个
        train_ds = Dataset(data=train_fl, transform=train_transform)
        val_ds = Dataset(data=val_fl, transform=val_transform)

    # 6. 定义 DataLoader
    # Windows下 num_workers 建议设为 0 或 2，设太大容易报错
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader



def get_loaders2(image_dir, label_dir, stage='liver', batch_size=2, cache=True):
    '''【原始数据集已划分】'''
    train_images = sorted(glob.glob(os.path.join(data_root, 'LiTS(train_test)', 'train_CT', "volume-*.nii")))
    train_labels = sorted(glob.glob(os.path.join(data_root, 'LiTS(train_test)', 'train_mask', "segmentation-*.nii")))
    
    # 检查训练集是否为空
    if len(train_images) == 0:
        raise ValueError("没找到 .nii 文件！")
    if len(train_images) != len(train_labels):
        raise ValueError(f"训练集图片数量({len(train_images)})与标签数量({len(train_labels)})不匹配！")

    # 组装训练字典
    train_fl = [
        {"image": img, "label": lbl} 
        for img, lbl in zip(train_images, train_labels)
    ]

    # 2. 读取测试集
    val_images = sorted(glob.glob(os.path.join(data_root, 'LiTS(train_test)', 'test_CT', "volume-*.nii")))
    val_labels = sorted(glob.glob(os.path.join(data_root, 'LiTS(train_test)', 'test_mask', "segmentation-*.nii")))

    # 检查测试集
    if len(val_images) == 0:
        print("警告: test_images 文件夹为空")
    
    # 组装验证字典
    val_fl = [
        {"image": img, "label": lbl} 
        for img, lbl in zip(val_images, val_labels)
    ]

    print(f"数据准备完成: 训练集 {len(train_files)} 例 | 验证集(来自test文件夹) {len(val_files)} 例")

    # 3. 获取 Transforms
    train_transform = get_transforms(stage=stage, mode='train')
    val_transform = get_transforms(stage=stage, mode='val')

    # 4. 定义 Dataset
    if cache:
        train_ds = CacheDataset(data=train_fl, transform=train_transform, cache_rate=0.2, num_workers=4)
        val_ds = CacheDataset(data=val_fl, transform=val_transform, cache_rate=0.2, num_workers=4)
    else:
        train_ds = Dataset(data=train_fl, transform=train_transform)
        val_ds = Dataset(data=val_fl, transform=val_transform)

    # 5. 定义 DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader