import torch
from scene.gaussian_model import GaussianModel


def test():
    print("--- 开始测试 GaussianModel ---")

    class MockTrainingArgs:
        def __init__(self):
            self.position_lr_init = 0.00016
            self.feature_lr = 0.0025
            self.opacity_lr = 0.05
            self.scaling_lr = 0.005
            self.rotation_lr = 0.001
            self.percent_dense = 0.01


            self.position_lr_final = 0.0000016
            self.position_lr_delay_mult = 0.01
            self.position_lr_max_steps = 30000
            self.exposure_lr_init = 0.0  # 即使未使用，也最好提供一个占位符
            self.exposure_lr_final = 0.0
            self.exposure_lr_delay_steps = 0
            self.exposure_lr_delay_mult = 0.0
            self.iterations = 30000

    training_args = MockTrainingArgs()

    # 1. 初始化高斯模型
    print("1. 初始化空的 GaussianModel...")
    gaussians = GaussianModel(sh_degree=3)
    print("   完成。")

    # 2. 加载您的预训练模型

    model_path = r"E:\pre_paper\LexiSplat\output\playroom\point_cloud\iteration_30000\point_cloud.ply"
    print(f"2. 从 {model_path} 加载预训练模型...")
    try:
        gaussians.load_ply(model_path)
        num_points = gaussians.get_xyz.shape[0]
        print(f"   加载成功！模型中包含 {num_points} 个高斯点。")
    except Exception as e:
        print(f"   加载失败！错误: {e}")
        return

    # 3. 调用我们的新方法：初始化语义特征
    print("3. 初始化语义特征...")
    gaussians.initialize_features_for_finetuning(feature_dim=16)
    assert hasattr(gaussians, '_features') and gaussians._features is not None, "   测试失败: _features 属性不存在或为None!"
    assert gaussians._features.shape == (
    num_points, 16), f"   测试失败: 特征形状不正确! 应该是 ({num_points}, 16)，但现在是 {gaussians._features.shape}"
    print("   特征初始化成功，形状正确。")

    # 4. 调用我们的新方法：冻结参数
    print("4. 冻结几何与外观参数...")
    gaussians.freeze_geometry_and_appearance()
    assert not gaussians.get_xyz.requires_grad, "   测试失败: XYZ 没有被冻结!"
    assert not gaussians._features_dc.requires_grad, "   测试失败: DC 特征没有被冻结!"
    assert gaussians._features.requires_grad, "   测试失败: 新的语义特征没有保持可训练状态!"
    print("   参数冻结成功。")

    # 5. 测试优化器设置
    print("5. 测试训练设置 (training_setup)...")
    try:
        gaussians.training_setup(training_args)
        feature_group_found = any(g['name'] == 'features' for g in gaussians.optimizer.param_groups)
        assert feature_group_found, "   测试失败: 'features' 参数组未添加到优化器!"
        print("   优化器设置成功，语义特征已加入优化列表。")
    except Exception as e:
        print(f"   训练设置失败！错误: {e}")
        return

    print("\n--- ✅ 所有测试通过！ `gaussian_model.py` 已准备就绪。 ---")


if __name__ == "__main__":
    test()