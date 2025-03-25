import os
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

# 定义标准化参数
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
    mean=IMAGE_NET_MEAN,
    std=IMAGE_NET_STD
)

class AVADataset(Dataset):
    def __init__(self, path_to_csv, images_path, if_train):
        """
        初始化数据集。

        参数：
            path_to_csv (str): CSV 文件的路径，包含图片 ID 和评分信息。
            images_path (str): 图片存储的根路径。
            if_train (bool): 是否为训练集，用于决定数据增强。
        """
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.if_train = if_train

        # 定义数据转换和增强
        if if_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        """
        返回数据集的大小。
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        获取指定索引的数据。

        参数：
            idx (int): 数据索引。

        返回：
            tuple: 包含图片张量、归一化的评分向量和图片 ID。
        """
        row = self.df.iloc[idx]

        # 提取评分列（假设评分列为 score2 到 score11）
        scores_names = [f'score{i}' for i in range(2, 12)]
        y = np.array([row[k] for k in scores_names], dtype=np.float32)
        p = y / y.sum()  # 归一化评分

        # 获取图片 ID 并转换为整数类型
        image_id = int(row['image_id'])

        # 构建图片路径
        image_path = os.path.join(self.images_path, f'{image_id}.jpg')

        # 加载图片
        image = default_loader(image_path)

        # 调整图片大小
        image = image.resize((224, 224))

        # 应用转换
        x = self.transform(image)

        return x, p, image_id
