# test_renderer.py 的最终版本

import torch
from scene import Scene
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render_for_test  # 导入新的测试函数
import matplotlib.pyplot as plt
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel


def test_feature_renderer(dataset_args):
    print("--- 开始测试语义特征渲染器 ---")

    # 1. 加载场景和预训练的高斯模型
    print("1. 加载场景和模型...")
    gaussians = GaussianModel(dataset_args.sh_degree)
    scene = Scene(dataset_args, gaussians, shuffle=False)
    view: Camera = scene.getTrainCameras()[0]
    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    print(f"   加载成功！将使用相机 '{view.image_name}' 进行测试。")

    # 2. 调用方法，为高斯点初始化随机的16维特征
    print("2. 初始化16维语义特征...")
    # 假设高斯模型中有一个方法可以初始化这些特征
    gaussians.initialize_features_for_finetuning(feature_dim=16)
    assert hasattr(gaussians, 'features') and gaussians.features is not None, "测试失败: 模型中未找到 'features' 属性!"
    print(f"   特征初始化成功，特征维度: {gaussians.features.shape[1]}。")

    # 3. 调用 render_for_test 函数
    print("3. 调用 render_for_test 函数...")
    try:
        # 在纯渲染测试中，我们不计算梯度
        with torch.no_grad():
            output = render_for_test(view, gaussians, background)

        rendered_color_map = output["color"]
        rendered_feature_map = output["feature"]
        print("   render_for_test 函数运行成功！")
    except Exception as e:
        print(f"   函数运行失败！错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 检查输出结果的有效性
    print("4. 检查输出...")
    expected_color_shape = (3, view.image_height, view.image_width)
    assert rendered_color_map.shape == expected_color_shape, f"测试失败: 颜色图形状不正确! 应该是 {expected_color_shape}，但现在是 {rendered_color_map.shape}"
    print(f"   颜色图形状正确: {rendered_color_map.shape}。")

    expected_feature_shape = (16, view.image_height, view.image_width)
    assert rendered_feature_map.shape == expected_feature_shape, f"测试失败: 特征图形状不正确! 应该是 {expected_feature_shape}，但现在是 {rendered_feature_map.shape}"
    print(f"   特征图形状正确: {rendered_feature_map.shape}。")

    assert torch.sum(torch.abs(rendered_feature_map)) > 0, "测试失败: 渲染的特征图是全零的!"
    print("   特征图包含非零数据。")

    print("\n--- ✅ 逻辑检查通过！正在进行可视化... ---")

    # 5. 可视化渲染结果
    try:
        feature_map_vis = rendered_feature_map[:3, :, :].detach().cpu().numpy().transpose(1, 2, 0)
        min_vals, max_vals = feature_map_vis.min(), feature_map_vis.max()
        feature_map_vis = (feature_map_vis - min_vals) / (max_vals - min_vals + 1e-6)

        color_map_vis = rendered_color_map.detach().cpu().numpy().transpose(1, 2, 0)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].imshow(color_map_vis)
        axes[0].set_title("Rendered Color")
        axes[0].axis('off')

        axes[1].imshow(feature_map_vis)
        axes[1].set_title("Rendered Feature Map")
        axes[1].axis('off')

        plt.suptitle("渲染器单元测试结果\n(请检查两张图是否都有合理的场景结构)")
        plt.show()

    except Exception as e:
        print(f"可视化失败！错误: {e}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script for feature renderer.")
    lp = ModelParams(parser)

    args = get_combined_args(parser)
    dataset_args = lp.extract(args)

    test_feature_renderer(dataset_args)