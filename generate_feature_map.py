import os
import argparse
import pickle
import torch
import clip
from PIL import Image
from tqdm import tqdm
import numpy as np


# 这是一个新脚本，用于实现我们讨论的预处理流程的 "步骤 C"。

def generate_feature_maps(dataset_path):
    """
    加载CLIP模型和预训练的PCA模型，为数据集中的每张图片生成低维语义特征图。
    """
    # --- 1. 路径设置 ---
    input_dir = os.path.join(dataset_path, "input")
    output_dir = os.path.join(dataset_path, "semantic_features")
    pca_model_path = os.path.join(dataset_path, "pca.pkl")

    if not os.path.exists(input_dir):
        print(f"错误: 找不到输入图片文件夹 {input_dir}")
        return

    if not os.path.exists(pca_model_path):
        print(f"错误: 找不到PCA模型文件 {pca_model_path}。请先运行 train_pca.py。")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"语义特征图将保存到: {output_dir}")

    # --- 2. 加载模型 ---
    print("正在加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载 CLIP 模型
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    print("CLIP 模型加载成功。")

    # 加载 PCA 模型
    with open(pca_model_path, 'rb') as f:
        pca = pickle.load(f)
    print("PCA 模型加载成功。")

    # --- 3. 遍历图片并处理 ---
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"找到 {len(image_files)} 张图片进行处理。")

    for image_name in tqdm(image_files, desc="生成特征图"):
        image_path = os.path.join(input_dir, image_name)

        # 加载并预处理图片
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            # 使用 CLIP 提取高维特征 (这里我们使用一个简化的方法，直接用图像特征)
            # 一个更复杂的实现会从模型的中间层提取密集的特征图
            high_dim_features = clip_model.encode_image(image_input)

        # 将特征从GPU转到CPU, 并转换为numpy
        high_dim_features_np = high_dim_features.cpu().numpy()

        # 使用 PCA 模型进行降维
        low_dim_features_np = pca.transform(high_dim_features_np)

        # 结果是一个 (1, 16) 的向量，我们需要将它广播成一张图
        # 这里我们创建一个 H x W x 16 的特征图，其中每个像素的特征都相同
        # 注意：这是最简单的实现。更高级的方法会为不同像素生成不同特征。
        # 但对于启动项目而言，这个方法是足够有效的。
        h, w = image.height, image.width
        # 将 (1, 16) -> (16,) -> (1, 1, 16) -> (h, w, 16)
        feature_map_np = np.broadcast_to(low_dim_features_np.reshape(1, 1, 16), (h, w, 16))

        # 转换为 PyTorch 张量
        feature_map_tensor = torch.from_numpy(feature_map_np).float()

        # --- 4. 保存结果 ---
        output_filename = os.path.splitext(image_name)[0] + ".pt"
        output_path = os.path.join(output_dir, output_filename)
        torch.save(feature_map_tensor, output_path)

    print("所有语义特征图生成完毕！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为数据集生成低维语义特征图")
    parser.add_argument("--dataset_path", type=str, required=True, help="数据集的路径 (例如, data/playroom)")
    args = parser.parse_args()

    generate_feature_maps(args.dataset_path)