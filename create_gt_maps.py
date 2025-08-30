# import os
# import argparse
# import pickle
# import torch
# import numpy as np
# from tqdm import tqdm
# from PIL import Image
#
# # generate_feature_map.py效果不理想，使用此版本
#
# def create_gt_maps(dataset_path):
#     """
#     使用预先计算的SAM掩码和CLIP特征，为每张图片生成高质量的、分区域的语义特征图。
#     核心计算部分使用PyTorch以提高健壮性。
#     """
#     print("开始生成高质量语义真值图...")
#     input_img_dir = os.path.join(dataset_path, "images")
#     masks_dir = os.path.join(dataset_path, "sam_clip_features")
#     features_dir = os.path.join(dataset_path, "sam_clip_features")
#     pca_model_path = os.path.join(dataset_path, "pca.pkl")
#     output_dir = os.path.join(dataset_path, "semantic_features")
#
#     for p in [input_img_dir, masks_dir, features_dir, pca_model_path]:
#         if not os.path.exists(p):
#             print(f"错误: 找不到必需的文件或文件夹: {p}")
#             return
#
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"输出文件夹已确认: {output_dir}")
#
#     with open(pca_model_path, 'rb') as f:
#         pca = pickle.load(f)
#     print("PCA 模型加载成功。")
#
#     image_names = [f for f in os.listdir(input_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#
#     for image_name in tqdm(image_names, desc="正在生成真值图"):
#         base_name = os.path.splitext(image_name)[0]
#
#         mask_path = os.path.join(masks_dir, f"{base_name}_masks.npy")
#         feature_path = os.path.join(features_dir, f"{base_name}_features.npy")
#
#         if not os.path.exists(mask_path) or not os.path.exists(feature_path):
#             continue
#
#         masks = np.load(mask_path)
#         high_dim_features = np.load(feature_path)
#
#         if masks.shape[0] != high_dim_features.shape[0]:
#             continue
#
#         low_dim_features = pca.transform(high_dim_features)
#
#         h, w = masks.shape[1], masks.shape[2]
#
#         masks_torch = torch.from_numpy(masks)
#         low_dim_features_torch = torch.from_numpy(low_dim_features)
#         feature_map_tensor = torch.zeros((h, w, 16), dtype=torch.float32)
#
#         mask_areas = np.sum(masks, axis=(1, 2))
#         sorted_indices = np.argsort(mask_areas)
#
#         for i in sorted_indices:
#             # +++ 修改: 将mask显式转换为布尔类型以消除警告 +++
#             mask = masks_torch[i].bool()
#             # +++ 修改结束 +++
#             feature = low_dim_features_torch[i]
#             feature_map_tensor[mask] = feature
#
#         output_pt_path = os.path.join(output_dir, f"{base_name}.pt")
#         torch.save(feature_map_tensor, output_pt_path)
#
#     print("\n所有高质量语义真值图生成完毕！")
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="根据SAM掩码和CLIP特征创建高质量的语义真值图。")
#     parser.add_argument("--dataset_path", type=str, required=True, help="数据集的根路径 (例如, data/playroom)")
#     args = parser.parse_args()
#
#     create_gt_maps(args.dataset_path)

import os
import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import DBSCAN


def create_gt_maps(dataset_path):
    """
    使用预先计算的SAM掩码和CLIP特征，通过特征聚类提纯，
    为每张图片生成高质量的、分区域的语义特征图。
    """
    # --- 1. 路径和模型加载 (与之前类似) ---
    print("开始生成高质量语义真值图 (v2 - 带聚类功能)...")
    input_img_dir = os.path.join(dataset_path, "images")
    masks_dir = os.path.join(dataset_path, "sam_clip_features")
    features_dir = os.path.join(dataset_path, "sam_clip_features")
    pca_model_path = os.path.join(dataset_path, "pca.pkl")
    output_dir = os.path.join(dataset_path, "semantic_features")

    for p in [input_img_dir, masks_dir, features_dir, pca_model_path]:
        if not os.path.exists(p):
            print(f"错误: 找不到必需的文件或文件夹: {p}")
            return

    os.makedirs(output_dir, exist_ok=True)
    print(f"输出文件夹已确认: {output_dir}")

    with open(pca_model_path, 'rb') as f:
        pca = pickle.load(f)
    print("PCA 模型加载成功。")

    image_names = [f for f in os.listdir(input_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_name in tqdm(image_names, desc="正在生成真值图"):
        base_name = os.path.splitext(image_name)[0]

        mask_path = os.path.join(masks_dir, f"{base_name}_masks.npy")
        feature_path = os.path.join(features_dir, f"{base_name}_features.npy")

        if not os.path.exists(mask_path) or not os.path.exists(feature_path):
            continue

        masks = np.load(mask_path)
        high_dim_features = np.load(feature_path)

        if masks.shape[0] != high_dim_features.shape[0] or masks.shape[0] == 0:
            continue

        norm_features = high_dim_features / np.linalg.norm(high_dim_features, axis=1, keepdims=True)

        clustering = DBSCAN(eps=0.4, min_samples=2, metric='cosine').fit(norm_features)
        cluster_labels = clustering.labels_

        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # -1代表噪声点，我们不处理

        # --- 3. 为每个簇计算原型特征 ---
        prototype_features = {}
        for k in unique_labels:
            cluster_mask = (cluster_labels == k)
            cluster_features = high_dim_features[cluster_mask]
            # 计算原型（平均特征）
            prototype_features[k] = np.mean(cluster_features, axis=0)

        # --- 4. 使用原型特征“绘制”真值图 ---
        h, w = masks.shape[1], masks.shape[2]
        feature_map = np.zeros((h, w, 16), dtype=np.float32)

        # 遍历所有计算出的原型
        for k in unique_labels:
            # 获取该簇对应的所有mask
            cluster_mask_indices = np.where(cluster_labels == k)[0]

            # 将该簇的所有mask合并成一个大的mask
            object_mask = np.any(masks[cluster_mask_indices], axis=0)

            # 对原型进行PCA降维
            prototype_high_dim = prototype_features[k].reshape(1, -1)
            prototype_low_dim = pca.transform(prototype_high_dim).flatten()

            # 将降维后的原型特征绘制到合并后的mask区域
            feature_map[object_mask] = prototype_low_dim

        # --- 5. 保存结果 ---
        feature_map_tensor = torch.from_numpy(feature_map).float()
        output_pt_path = os.path.join(output_dir, f"{base_name}.pt")
        torch.save(feature_map_tensor, output_pt_path)

    print("\n所有高质量语义真值图（经聚类提纯）生成完毕！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据SAM掩码和CLIP特征创建高质量的语义真值图。")
    parser.add_argument("--dataset_path", type=str, required=True, help="数据集的根路径 (例如, data/playroom)")
    args = parser.parse_args()

    create_gt_maps(args.dataset_path)