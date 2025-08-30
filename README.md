# PCA-Splat: 通过PCA降维CLIP特征为3D高斯溅射注入语义场

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## 简介

本项目是基于 **3D Gaussian Splatting** 的一个开源实现，旨在为一个已经重建好的3D高斯场景**高效地注入一个开放词汇的语义场**。

作为 `3dgs` 的基础框架，功能上以及效果上相对较差，可作为一个简单的框架使用。该方法的核心思想是，利用强大的预训练视觉-语言模型(**CLIP**)作为语义知识的来源，通过主成分分析(**PCA**)构建一个稳定且紧凑的语义监督空间，最终以微调的方式为场景中的每个三维高斯基元赋予可学习的语义特征。

## 核心特性

* **开放词汇查询**: 能够响应任意文本描述，而不仅限于预设的类别。
* **后期处理**: 无需重新训练3DGS的几何和颜色，直接在已有的模型上进行语义微调。
* **低额外开销**: 仅为每个点增加一个16维的浮点特征，对模型体积影响较小。
* **方法简洁**: 整个流程不涉及复杂的网络结构或损失函数，易于理解和实现。

## 方法论

我们的技术管线分为两个主要阶段：

### 1. 监督信号生成

我们首先为训练视图生成像素级的真值(Ground Truth)语义图。

1.  使用 **Segment Anything Model (SAM)** 对每张训练图像进行过分割，得到高质量的物体掩码。
2.  对每个掩码，使用 **CLIP图像编码器** 提取其对应的512维高维特征向量。
3.  收集整个数据集中所有掩码的CLIP特征，使用 **主成分分析(PCA)** 学习一个从512维到16维的投影。
4.  将所有掩码的特征投影到这个16维空间，最终为每张训练图生成一张16通道的真值语义图。

### 2. 语义微调

1.  为原始3D高斯模型中的每个点，增加一个随机初始化的16维可学习语义特征 `f`。
2.  **冻结**所有与几何、颜色相关的原始参数（位置、旋转、不透明度、球谐系数）。
3.  使用一个与颜色渲染并行的**可微分语义渲染器**，将3D点的语义特征渲染到2D图像上。
4.  使用简单的 **L1损失函数**，度量渲染语义图与真值语义图之间的差异，并仅更新语义特征。

---

## 使用指南

### 步骤 1: 环境与模型准备

首先，请确保您已有一个训练好的3D Gaussian Splatting场景。

**环境安装:**
克隆本项目并安装所需依赖：

```bash
git clone [您的项目git地址]
cd [您的项目目录]
pip install -r requirements.txt
```
*(注: `requirements.txt` 应包含 `torch`, `segment-anything-py`, `clip-openai`, `scikit-learn`, `numpy` 等)*

**模型准备 (关键步骤):**

* **下载SAM权重**: 请从 [SAM官方仓库](https://github.com/facebookresearch/segment-anything#model-checkpoints) 下载预训练的模型检查点。推荐使用 `ViT-H` 版本 (`sam_vit_h_4b8939.pth`)。
* **存放权重**: 将下载的 `.pth` 文件放置在项目根目录下的 `checkpoints` 文件夹中（如果不存在请创建一个）。

    ```bash
    mkdir -p checkpoints
    wget -P checkpoints [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
    ```
* **CLIP模型**: `CLIP` 模型权重通常会在您首次运行脚本时由 `clip` 库**自动下载**，请确保此时有网络连接。

### 提取SAM掩码和CLIP特征

此脚本将调用您准备好的SAM和OpenCLIP模型，为指定数据集路径下的所有图片生成掩码和特征文件。

```bash
# 命令格式
python new_process.py --dataset_path [数据集路径]

# 示例
# 假设您的数据集（包含images文件夹）位于 "data/playroom"
# 脚本将在 "data/playroom/sam_clip_features/" 目录下生成结果
python new_process.py --dataset_path data/playroom
```


### 步骤 2: 生成真值语义图

该命令会遍历指定3DGS场景的训练相机，为每张图片生成对应的16维语义真值图。此脚本内部完成了SAM分割、CLIP提取和PCA降维的全过程。

```bash
# 命令格式
python create_gt_maps.py -s [原始3DGS场景路径] --output_path [真值图输出路径]

# 示例
# 假设您的原始场景位于 "data/tandt/truck"
# 我们将生成的真值图保存到 "data/tandt/truck/gt_maps"
python create_gt_maps.py -s data/tandt/truck --output_path data/tandt/truck/gt_maps
```
* `-s, --source_path`: 指向您原始3DGS场景的路径。
* `--output_path`: 用于存储生成的`.npy`格式真值图的目录。

### 步骤 3: 微调语义场

在生成真值图后，我们开始对场景进行语义微调。

```bash
# 命令格式
python train_semantic.py -s [原始3DGS场景路径] -m [语义模型输出路径] --gt_path [真值图路径]

# 示例
# 我们在原始场景 "data/tandt/truck" 的基础上进行训练
# 将训练好的语义模型保存到 "output/truck_semantic"
# 指定我们上一步生成的真值图路径 "data/tandt/truck/gt_maps"
python train_semantic.py -s data/tandt/truck -m output/truck_semantic --gt_path data/tandt/truck/gt_maps
```
* `-s, --source_path`: 原始3DGS场景路径。
* `-m, --model_path`: 训练过程中模型检查点和最终语义模型的保存路径。
* `--gt_path`: 上一步生成的真值图所在的目录。

### 步骤 4: 评估与查询

训练完成后，您可以使用以下命令通过文本查询来可视化语义场的学习效果。

```bash
# 命令格式
python evaluate_semantics.py -m [语义模型路径] --text "您的文本查询" --output_path [热力图保存路径]

# 示例
# 查询我们训练好的 "output/truck_semantic" 模型
# 查询内容为 "the wheel of the truck" (卡车的轮子)
# 将生成的热力图保存为 "output/wheel_heatmap.png"
python evaluate_semantics.py -m output/truck_semantic --text "the wheel of the truck" --output_path output/wheel_heatmap.png
```
* `-m, --model_path`: 指向已完成语义微调的模型路径。
* `--text`: 您希望查询的自然语言描述。
* `--output_path`: 生成的2D热力图的保存文件名。
