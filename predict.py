import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import os
from config import get_config
from models import build_model
import warnings

# 禁用警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FakeArgs:
    """模拟训练脚本的参数结构"""

    def __init__(self, config_path):
        self.cfg = config_path
        self.opts = None
        self.data_path = ""  # 修复关键缺失参数
        self.tag = "predict"
        self.eval = True
        self.pretrained = ''
        self.amp = False
        self.output = 'output'
        self.resume = ''
        self.batch_size = 1  # 添加可能需要的参数
        self.num_workers = 0


def load_model(config_path, ckpt_path):
    """安全加载模型"""
    try:
        # 验证文件存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model checkpoint {ckpt_path} not found")

        # 创建模拟参数
        args = FakeArgs(config_path)

        # 获取配置
        config = get_config(args)

        # 构建模型
        print(f"Building {config.MODEL.TYPE} model...")
        model = build_model(config)

        # 加载权重
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "state_dict" in checkpoint:  # 兼容不同保存格式
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")


def preprocess_image(img_path):
    """与训练一致的预处理"""
    try:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = Image.open(open(img_path, 'rb')).convert('RGB')
        return transform(img).unsqueeze(0)  # 添加batch维度
    except Exception as e:
        raise ValueError(f"Image processing error: {str(e)}")


def calculate_score(pred_dist):
    """与训练一致的分数计算"""
    weights = torch.linspace(1, 10, 10).to(device)
    return torch.sum(pred_dist * weights, dim=1).item()


def predict(image_path):
    """端到端预测流程"""
    try:
        # 配置路径（需用户修改！）
        CONFIG_FILE = "configs/dat_base.yaml"  # 模型配置文件
        MODEL_CKPT = 'model_weights.pth'  # 模型权重文件

        # 加载模型
        model = load_model(CONFIG_FILE, MODEL_CKPT)

        # 处理图像
        input_tensor = preprocess_image(image_path)

        # 推理预测
        with torch.no_grad():
            pred_dist, _, _ = model(input_tensor.to(device))

        # 计算分数
        score = calculate_score(pred_dist)
        return round(score, 4)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    # 命令行界面
    parser = argparse.ArgumentParser(
        description="图像质量评估 (0-10分)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("image_path",
                        type=str,
                        help="输入图像路径")
    args = parser.parse_args()

    try:
        # 执行预测
        score = predict(args.image_path)

        # 打印美观结果
        print("\n" + "=" * 40)
        print(f"  Predicted Quality Score: {score}/10")
        print("=" * 40)
        print("质量等级说明:")
        print("  🟢 9-10: 极佳质量")
        print("  🟡 7-8:  良好质量")
        print("  🟠 5-6:  一般质量")
        print("  🔴 3-4:  较差质量")
        print("  ⚫ 1-2:  无法接受")
        print("=" * 40)

    except Exception as e:
        print("\n❌ 错误发生:")
        print(f"  {str(e)}")
        print("\n故障排查建议:")
        print("1. 检查图像路径是否正确（支持jpg/png格式）")
        print("2. 确认模型文件存在且路径正确")
        print("3. 检查配置文件是否与训练时一致")
        print("4. 尝试重新安装依赖库：torch, torchvision, pillow")