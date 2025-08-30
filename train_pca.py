import os
import numpy as np
import pickle
from sklearn.decomposition import PCA
from tqdm import tqdm
import argparse

# 新增文件，压缩clip feature

def train_pca(dataset_path):
    features_dir = os.path.join(dataset_path, "sam_clip_features")
    if not os.path.exists(features_dir):
        print(f"错误: 找不到特征文件夹 {features_dir}")
        return

    # 1. 查找所有特征文件
    all_feature_files = [os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith("_features.npy")]
    print(f"找到 {len(all_feature_files)} 个特征文件。")

    if not all_feature_files:
        print("错误: 未找到任何特征文件。请先运行预处理脚本。")
        return

    # 2. 加载所有特征并堆叠
    all_features = []
    print("正在加载所有特征...")
    for f_path in tqdm(all_feature_files, desc="加载特征文件"):
        features = np.load(f_path)
        if features.ndim == 2 and features.shape[0] > 0:
            all_features.append(features)

    stacked_features = np.concatenate(all_features, axis=0)
    print(f"所有特征加载完毕。总特征数量: {stacked_features.shape[0]}, 维度: {stacked_features.shape[1]}")

    # 3. 训练PCA模型
    print("开始训练PCA模型 (目标维度: 16)...")
    pca = PCA(n_components=16)
    pca.fit(stacked_features)
    print("PCA模型训练完成。")

    # 4. 保存模型
    save_path = os.path.join(dataset_path, "pca.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(pca, f)
    print(f"PCA模型已成功保存至: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练PCA模型用于CLIP特征降维")
    parser.add_argument("--dataset_path", type=str, required=True, help="数据集的路径 (例如, data/playroom)")
    args = parser.parse_args()

    train_pca(args.dataset_path)