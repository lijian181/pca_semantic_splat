#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
        # 注意：标准render不渲染特征
        include_feature=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    shs = pc.get_features
    scales = pc.get_scaling
    rotations = pc.get_rotation

    colors_precomp = None
    if override_color is not None:
        colors_precomp = override_color
    elif pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    # Rasterize
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)

    rendered_image = rendered_image.clamp(0, 1)

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }



def render_all(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0):
    """
    Render the scene, simultaneously producing both the color image and the semantic feature map.

    Background tensor (bg_color) must be on GPU!
    """

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,

        antialiasing=True,  
        include_feature=True  
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 准备所有需要传递给光栅化器的数据
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # 颜色特征 (球谐函数)
    shs = pc.get_features

    # 语义特征
    semantic_features = pc.get_semantic_features

    # 调用我们修改后的底层光栅化器
    # 它现在会返回三个值：颜色图，特征图，半径
    rendered_image, rendered_features, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        language_feature_precomp=semantic_features,  # 传入16维特征
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)

    rendered_image = rendered_image.clamp(0, 1)

    # 将所有渲染结果打包成一个字典返回
    return {
        "render": rendered_image,
        "feature": rendered_features,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
# +++ MODIFICATION END +++


# # gaussian_renderer/__init__.py
#
#
# def render_features(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
#                     override_color=None):
#     """
#     Render the semantic features of the scene.
#     """
#     print("\n>>> DEBUG: 正在执行被我们修正过的 render_features 函数！ <<<\n")
#
#     # Create zero tensor. We will use it to make fine-grained controls over memory allocation
#     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass
#
#     # Set up rasterization configuration
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
#
#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         sh_degree=0,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         antialiasing=pipe.antialiasing,
#         debug=pipe.debug
#     )
#
#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)
#
#     means3D = pc.get_xyz
#     means2D = screenspace_points
#     opacity = pc.get_opacity

#
#     shs = pc.features
#     colors_precomp = None
#
#     # =================================================================
#
#     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
#     # scaling / rotation quaternions.
#     scales = None
#     rotations = None
#     cov3D_precomp = None
#     if pipe.compute_cov3D_python:
#         cov3D_precomp = pc.get_covariance(scaling_modifier)
#     else:
#         scales = pc.get_scaling
#         rotations = pc.get_rotation
#
#     # Rasterize visible Gaussians to image, this will produce rendered_image and radii
#     rendered_image, radii, depth_image = rasterizer(
#         means3D=means3D,
#         means2D=means2D,
#         shs=shs,  # << --- 确保这里传入的是我们的16维特征
#         colors_precomp=colors_precomp,
#         opacities=opacity,
#         scales=scales,
#         rotations=rotations,
#         cov3D_precomp=cov3D_precomp)
#
#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     # They will be excluded from gradient propagation.
#     return {"render": rendered_image,
#             "viewspace_points": screenspace_points,
#             "visibility_filter": radii > 0,
#             "radiadii": radii,
#             "depth": depth_image}
#
#
# def render_for_test(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor):
#     """
#     一个专门用于测试的渲染函数，它会同时渲染颜色和我们新增的语义特征。
#     """
#
#     # 引入核心的 autograd.Function
#     from diff_gaussian_rasterization import GaussianRasterizationSettings, _RasterizeGaussians
#
#     # 设置光栅化配置
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
#
#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=1.0,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         sh_degree=0,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         debug=False,
#         antialiasing=True,
#         include_feature=True
#     )
#
#     # 准备数据
#     means3D = pc.get_xyz
#     means2D = torch.zeros_like(means3D, requires_grad=True, device="cuda") + 0
#     colors_precomp = pc.get_features_dc.squeeze(1)
#     language_feature_precomp = pc.features
#     opacities = pc.get_opacity
#     scales = pc.get_scaling
#     rotations = pc.get_rotation
#     cov3Ds_precomp = None
#
#     # 我们直接调用底层的 torch.autograd.Function，并使用正确的参数名
#     rendered_color, rendered_feature, _ = _RasterizeGaussians.apply(
#         means3D,
#         means2D,
#         None,  # sh (我们不使用球谐函数，所以是None)
#         colors_precomp,
#         language_feature_precomp, # 传入16维特征
#         opacities,
#         scales,
#         rotations,
#         cov3Ds_precomp,
#         raster_settings
#     )
#
#     # 返回渲染结果
#     return {
#         "color": rendered_color,
#         "feature": rendered_feature
#         }