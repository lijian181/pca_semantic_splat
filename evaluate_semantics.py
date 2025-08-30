import os
import argparse
import pickle
import torch
import clip
import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import matplotlib.cm as cm


def evaluate_semantics(ply_path, pca_path, query, output_path, threshold):
    """
    加载3DGS模型，根据文本查询和相似度阈值，筛选出匹配的高斯点并保存为新的PLY文件。
    """
    # --- 1. 加载模型 ---
    print("正在加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    print("CLIP 模型加载成功。")
    if not os.path.exists(pca_path):
        print(f"错误: 找不到PCA模型文件 {pca_path}。")
        return
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    print("PCA 模型加载成功。")

    # --- 2. 加载PLY文件 ---
    print(f"正在从 '{ply_path}' 加载原始点云...")
    if not os.path.exists(ply_path):
        print(f"错误: 找不到训练好的PLY文件 {ply_path}。")
        return
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    semantic_feature_names = [name for name in vertices.data.dtype.names if name.startswith('f_semantic_')]
    if not semantic_feature_names:
        print("错误: 此PLY文件中未找到'f_semantic_'开头的语义特征。")
        return
    semantic_features = np.vstack([vertices[name] for name in semantic_feature_names]).T
    semantic_features_torch = torch.from_numpy(semantic_features).to(device)

    # --- 3. 处理文本查询 ---
    print(f"正在处理文本查询: '{query}'")
    with torch.no_grad():
        text_token = clip.tokenize([query]).to(device)
        text_features_high_dim = clip_model.encode_text(text_token).float()
    text_features_high_dim_np = text_features_high_dim.cpu().numpy()
    text_features_low_dim_np = pca.transform(text_features_high_dim_np)
    text_features_torch = torch.from_numpy(text_features_low_dim_np).to(device)

    # --- 4. 计算相似度  ---
    print("正在计算语义相似度...")
    semantic_features_torch = torch.nn.functional.normalize(semantic_features_torch, p=2, dim=1)
    text_features_torch = torch.nn.functional.normalize(text_features_torch, p=2, dim=1)
    similarities = (semantic_features_torch @ text_features_torch.T).squeeze()
    similarities_cpu = similarities.cpu().numpy()
    print(f"相似度计算完成。分数范围: [{similarities_cpu.min():.4f}, {similarities_cpu.max():.4f}]")

    # --- 5. 应用阈值进行筛选 ---
    print(f"正在应用相似度阈值: {threshold}...")
    mask = similarities_cpu > threshold
    num_passed = np.sum(mask)

    if num_passed == 0:
        print(f"警告: 在当前阈值 {threshold} 下，没有点通过筛选。尝试降低阈值。")
        print("程序退出。")
        return

    print(f"{num_passed} / {len(vertices)} 个点通过筛选。")

    # 使用掩码筛选出符合条件的顶点
    filtered_vertices = vertices[mask]

    # --- 6. [新] 保存筛选后的稀疏点云 ---
    print("正在保存筛选后的点云...")

    new_element = PlyElement.describe(filtered_vertices, 'vertex')
    new_ply = PlyData([new_element], text=False)
    new_ply.write(output_path)
    print(f"筛选完成！结果已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通过阈值筛选测试3DGS模型的CLIP语义命中情况")
    parser.add_argument("--ply_path", type=str, required=True, help="训练好的模型.ply文件路径")
    parser.add_argument("--pca_path", type=str, required=True, help="训练好的pca.pkl文件路径")
    parser.add_argument("--query", type=str, required=True, help="要查询的文本描述")
    parser.add_argument("--output_path", type=str, required=True, help="输出的筛选结果.ply文件路径")

    parser.add_argument("--threshold", type=float, default=0.25, help="用于筛选的余弦相似度阈值 (0到1之间)")
    args = parser.parse_args()

    evaluate_semantics(args.ply_path, args.pca_path, args.query, args.output_path, args.threshold)