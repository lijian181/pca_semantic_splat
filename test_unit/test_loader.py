import sys
import torch
import argparse
from scene import sceneLoadTypeCallbacks
from PIL import Image
import numpy as np


import matplotlib.pyplot as plt


def test_data_loader(dataset_path):
    print("--- 开始测试数据加载器 ---")

    # 1. 调用您修改后的场景加载函数
    print(f"1. 尝试从 {dataset_path} 加载场景...")
    try:

        scene_info = sceneLoadTypeCallbacks["Colmap"](dataset_path, images="images", depths="", eval=False,
                                                      train_test_exp=False)
        print("   场景加载函数运行成功！")
    except Exception as e:
        print(f"   场景加载失败！错误: {e}")
        return

    # 2. 检查返回的数据结构
    print("2. 检查相机数据结构...")
    assert len(scene_info.train_cameras) > 0, "测试失败: 未能加载任何训练相机！"

    first_cam = scene_info.train_cameras[0]
    print(f"   成功获取第一个相机: {first_cam.image_name}")

    assert hasattr(first_cam, 'gt_feature_map'), "测试失败: CameraInfo对象中不存在 'gt_feature_map' 属性!"
    print("   属性 'gt_feature_map' 存在。")

    assert isinstance(first_cam.gt_feature_map,
                      torch.Tensor), f"测试失败: gt_feature_map 不是一个PyTorch张量, 而是 {type(first_cam.gt_feature_map)}"
    print(f"   属性类型正确 (torch.Tensor)。")

    expected_shape = (first_cam.height, first_cam.width, 16)
    assert first_cam.gt_feature_map.shape == expected_shape, f"测试失败: 特征图形状不正确! 应该是 {expected_shape}，但现在是 {first_cam.gt_feature_map.shape}"
    print(f"   特征图形状正确: {first_cam.gt_feature_map.shape}。")

    assert torch.sum(first_cam.gt_feature_map) > 0, "测试失败: 特征图是全零的，说明mask或feature未能正确填充!"
    print("   特征图包含非零数据，填充成功。")

    print("\n--- ✅ 逻辑检查通过！正在进行可视化... ---")

    # 3. 可视化第一个相机的真值图
    try:
        # 加载原始图片
        original_image = Image.open(first_cam.image_path)

        # 将特征图转换为可显示的图像
        # 我们只取前3个通道，并将其数值归一化到0-1范围以便显示
        feature_map_vis = first_cam.gt_feature_map[:, :, :3].numpy()
        min_vals = feature_map_vis.min(axis=(0, 1), keepdims=True)
        max_vals = feature_map_vis.max(axis=(0, 1), keepdims=True)
        feature_map_vis = (feature_map_vis - min_vals) / (max_vals - min_vals + 1e-6)

        # 创建一个2栏的图像进行对比
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].imshow(original_image)
        axes[0].set_title("原始图片 (Original Image)")
        axes[0].axis('off')

        axes[1].imshow(feature_map_vis)
        axes[1].set_title("真值特征图可视化 (Feature Map Visualization)")
        axes[1].axis('off')

        plt.suptitle("数据加载器测试结果 (请检查右图是否有合理的颜色分区)")
        plt.show()

    except Exception as e:
        print(f"可视化失败！错误: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试数据加载器和特征图生成")
    parser.add_argument("--source_path", "-s", type=str, required=True,
                        help="数据集的路径 (例如, data/playroom 或 output/playroom)")
    args = parser.parse_args()

    test_data_loader(args.source_path)