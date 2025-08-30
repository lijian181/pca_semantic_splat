import os
import random
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
from dataclasses import dataclass, field
from typing import Tuple, Type
import torchvision
from torch import nn

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"


# =================================================================================
# OpenCLIPNetwork 类和其配置 (来自您的代码，保持不变)
# =================================================================================
@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"  # 建议使用官方名称，或确保您的本地路径正确
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)


class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),  # <--- 添加这一行，它会将PIL Image转换为[0,1]范围的Tensor
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,
            pretrained=self.config.clip_model_pretrained,
            precision="fp16",  # 使用fp16可以加速
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims
        # ... (其余部分与您的代码相同，这里为了简洁省略，实际使用时请保留完整)

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    # =================================================================================
    # 【最终修正版】请用这个版本完整替换您代码中的 encode_image 函数
    # =================================================================================
    def encode_image(self, pil_image):
        """
        接收一个Pillow图像，返回其CLIP特征向量。
        """
        # 1. self.process 会将PIL Image转换为归一化后的Tensor (此时为 float32)
        processed_tensor = self.process(pil_image)

        # 2. 增加batch维度
        processed_tensor = processed_tensor.unsqueeze(0)

        # 3. 从模型参数中获取正确的device
        target_device = next(self.model.parameters()).device

        # 4. 【核心修正】不再自动推断dtype，而是直接强制转换为 float16
        processed_tensor = processed_tensor.to(device=target_device, dtype=torch.float16)

        # 5. (可选的调试代码) 加上这行可以验证两种类型是否已经一致
        # print(f"Input Tensor Dtype: {processed_tensor.dtype}, Model Weight Dtype: {next(self.model.parameters()).dtype}")

        # 6. 返回编码后的图像特征
        return self.model.encode_image(processed_tensor)


# =================================================================================
# 辅助函数 (来自您的代码，保持不变)
# =================================================================================
def get_seg_img(mask, image):
    image = image.copy()
    # 使用布尔索引，确保对所有通道应用mask
    image[~mask['segmentation']] = 0
    x, y, w, h = np.int32(mask['bbox'])
    seg_img = image[y:y + h, x:x + w, :]
    return seg_img


def pad_img(img):
    h, w, _ = img.shape
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        pad[:, (h - w) // 2:(h - w) // 2 + w, :] = img
    else:
        pad[(w - h) // 2:(w - h) // 2 + h, :, :] = img
    return pad


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # 改为False保证完全可复现


# =================================================================================
# 【核心修改】全新的、简化的处理函数
# =================================================================================
def create_simplified(image_paths, save_folder, mask_generator, clip_model):
    """
    简化的处理流程：对每张图片进行一次SAM分割，保存所有mask及其对应的CLIP特征。
    """
    print("--- 开始执行简化版预处理流程 ---")

    # 将SAM模型移至GPU
    mask_generator.predictor.model.to('cuda')

    for img_path in tqdm(image_paths, desc="处理图片中"):
        image_np_bgr = cv2.imread(img_path)
        if image_np_bgr is None:
            print(f"警告：无法读取图片 {img_path}，已跳过。")
            continue
        image_np_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)

        # 1. 使用SAM生成所有Mask
        # --- 修改开始 ---
        # 原始的generate函数可能返回了一个列表的列表，例如 [[mask1, mask2], [mask3], ...]
        list_of_mask_lists = mask_generator.generate(image_np_rgb)

        # 我们将这个嵌套的列表“压平”成一个单一的mask列表
        masks_result = [mask for sublist in list_of_mask_lists for mask in sublist]
        # --- 修改结束 ---

        if not masks_result:
            print(f"警告：图片 {os.path.basename(img_path)} 未检测到任何mask，将跳过。")
            continue

        # 2. 准备用于保存的Mask数组 (N, H, W)
        all_masks_np = np.stack([mask['segmentation'] for mask in masks_result], axis=0).astype(np.uint8)

        # 3. 为每个Mask提取CLIP特征
        clip_features_list = []
        for mask in masks_result:
            seg_img_tile = get_seg_img(mask, image_np_rgb)
            padded_tile = pad_img(seg_img_tile)

            # PIL Image转换和归一化更适合 torchvision transforms
            tile_pil = Image.fromarray(padded_tile)

            with torch.no_grad():
                feature = clip_model.encode_image(tile_pil)
                feature /= feature.norm(dim=-1, keepdim=True)

            clip_features_list.append(feature.squeeze(0).cpu())

        all_features_tensor = torch.stack(clip_features_list, dim=0)

        # 4. 保存结果
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path_base = os.path.join(save_folder, base_name)

        np.save(save_path_base + '_masks.npy', all_masks_np)
        np.save(save_path_base + '_features.npy', all_features_tensor.to(torch.float32).numpy())

    # 将SAM模型移回CPU以节省显存
    mask_generator.predictor.model.to('cpu')
    print("--- 处理完成！---")


# =================================================================================
# 【核心修改】修改后的主执行逻辑
# =================================================================================
if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser(description="Preprocess images with SAM and CLIP to generate masks and features.")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help="Path to the dataset directory (e.g., data/playroom).")
    parser.add_argument('--sam_ckpt_path', type=str, default="sam_vit_h_4b8939.pth",
                        help="Path to the SAM checkpoint file.")
    # 添加一个OpenCLIP权重的参数
    parser.add_argument('--clip_ckpt_path', type=str, default="laion2b_s34b_b88k",
                        help="Path or official name for the OpenCLIP checkpoint.")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)

    # 路径设置
    img_folder = os.path.join(args.dataset_path, 'images')
    if not os.path.isdir(img_folder):
        raise FileNotFoundError(f"错误：找不到图片文件夹 '{img_folder}'")

    save_folder = os.path.join(args.dataset_path, 'sam_clip_features')
    os.makedirs(save_folder, exist_ok=True)

    # 获取图片列表
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_names = sorted([f for f in os.listdir(img_folder) if os.path.splitext(f)[1].lower() in image_extensions])
    image_paths = [os.path.join(img_folder, fname) for fname in image_names]

    # 初始化模型
    print("正在初始化模型...")
    # 动态设置CLIP权重路径
    clip_config = OpenCLIPNetworkConfig(clip_model_pretrained=args.clip_ckpt_path)
    clip_model = OpenCLIPNetwork(clip_config)

    sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt_path)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        box_nms_thresh=0.7,
        min_mask_region_area=100,
    )
    print("模型初始化完成。")

    # === 调用新的、简化的主函数 ===
    create_simplified(image_paths, save_folder, mask_generator, clip_model)