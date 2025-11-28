# 1.路径设置
data_root = 'D:/工程项目/肝脏和肿瘤分割/data'
MODEL_DIR = 'D:/工程项目/肝脏和肿瘤分割/output/models'
LOG_DIR = 'D:/工程项目/肝脏和肿瘤分割/output/logs'

INPUT_DIR = config.data_root + "/LiTS(train_test)/test_CT"  
OUTPUT_DIR = "./output/preds"
LIVER_MODEL = "./output/models/best_model_liver.pth"
TUMOR_MODEL = "./output/models/best_model_tumor.pth"
    

# 2.训练超参数
SEED = 42           # 随机种子
MAX_EPOCHS = 100    # 训练轮数
VAL_INTERVAL = 2    # 验证频率
BATCH_SIZE = 2      # 批次大小 (显存不够就改这里)
LR = 1e-4           # 学习率
NUM_WORKERS = 2     # 数据加载线程数

# 3.数据预处理参数
INTENSITY_RANGE = (-175, 250)    # CT值窗口范围 (Min, Max) - 对应肝脏/腹部窗口
PATCH_SIZE = (96, 96, 32)     # 训练时切块大小 (Depth, Height, Width)
NUM_SAMPLES = 4    # 切块采样数 (每个CT切几个块)