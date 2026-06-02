#!/usr/bin/env python3
"""
【统一评测脚本】- 给Python新手的详细注释版
功能说明：
    这个脚本用于评估视觉大语言模型(VLM)在目标检测任务上的表现。
    它可以测试6个Qwen3-VL模型，使用5种不同的"注意力引导"方法。
 
【核心概念解释】
1. 什么是VLM（视觉大语言模型）？
   - 可以同时理解图像和文字的AI模型
   - 输入：一张图片 + 一个问题
   - 输出：文字回答（可以是检测框坐标）
 
2. 什么是目标检测（Object Detection）？
   - 让模型找出图片中特定物体的位置
   - 用矩形框(bbox/bounding box)标出物体位置
   - 坐标格式：[x1, y1, x2, y2]，分别是左上角和右下角的像素坐标
 
3. 5种方法的区别：
   - baseline（基线）：直接问问题，不做任何特殊处理
   - prompt（提示）：在问题中告诉模型"目标实体"的大致位置
   - blur（模糊）：把目标区域模糊，让模型学会关注其他地方
   - blackout（遮挡）：把目标区域涂黑，完全遮住目标
   - crop（裁切）：只保留目标区域附近的内容
 
【适用环境】
    - 需要NVIDIA 3090/4090显卡
    - 使用HuggingFace Transformers库加载Qwen3-VL模型
"""
# from __future__ import annotations
# 作用：让Python 3.11之前的老版本Python也能使用"list[str]"这种新语法
# 如果没有这行，在Python 3.9中要写 List[str]（需要从typing导入）
from __future__ import annotations

import json      # 用于处理JSON格式数据（模型评测结果用JSONL存储）
import os        # 用于操作环境变量、文件路径等
import sys       # 用于退出程序、获取命令行参数等
from pathlib import Path  # Path是更现代的路径操作方式，比字符串更安全
from typing import Any     # Any表示任意类型，用于类型注解

# 设置HuggingFace镜像地址，因为国内直接访问HF比较慢
# 这行代码设置了环境变量，让transformers库使用镜像站下载模型
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 下面是图像处理和深度学习相关的库
import cv2          # OpenCV库，用于图像处理（模糊、裁切等）
import numpy as np # 数值计算库，图像在Python中以numpy数组形式存储
import torch        # PyTorch深度学习框架
from PIL import Image  # Pillow库，用于读取和保存图像文件
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
# transformers是HuggingFace的核心库，用于加载和使用预训练模型
# - AutoProcessor: 自动配置处理器（图像+文本的预处理）
# - Qwen3VLForConditionalGeneration: Qwen3-VL视觉语言模型

# 从config.py导入配置参数
# config文件里定义了模型路径、方法列表、阈值等配置
from config import (
    BASE_DIR, BBOX_PROMPT_TEMPLATE, BBOX_PROMPT_WITH_ENTITIES, BLUR_KERNEL_SIZE,
    DATA_DIR, FIG_JSONL, IOU_THRESHOLD, MAX_IMAGE_DIM, MAX_TOKENS, METHODS,
    MODELS, NUM_SAMPLES, TEST_JSONL,
)


# ============================================================
# 【第二部分：数据加载函数】
# ============================================================
# 这部分负责读取测试数据，包括：
# 1. fig_data: 包含图像中所有实体（Entity）位置信息的JSONL文件
# 2. test_data: 包含问题和标准答案的测试集

def load_fig_data(fig_path: Path) -> dict[str, dict[str, Any]]:
    """
    加载实体（Entity）数据文件。
    
    【什么是Entity数据？】
    - Entity是指图像中的关键物体，比如遥感影像中的建筑、车辆等
    - 每个Entity有：id（标识符）、description（描述）、center（中心点坐标）
    - 这些信息用于构建位置提示（prompt方法）或者计算焦点区域（blur/blackout/crop方法）
    
    【什么是JSONL格式？】
    - JSONL = JSON Lines，每行是一个完整的JSON对象
    - 示例：
      {"image": "0001.jpg", "entities": [{"id": "b1", "center": [0.5, 0.5]}]}
      {"image": "0002.jpg", "entities": [...]}
    
    参数:
        fig_path: fig.jsonl文件的路径
    
    返回值:
        dict[str, dict[str, Any]]: 字典，键是图像文件名，值是包含entities的字典
        形如：{"image1.jpg": {"entities": [{"id": "b1", "center": [0.5, 0.5]}, ...]}}
    """
    # 创建一个空字典，用于存储图像名到实体数据的映射
    fig_map = {}
    
    # 打开JSONL文件，注意使用utf-8编码避免中文乱码
    with fig_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # 去除首尾空白字符
            if line:  # 跳过空行
                data = json.loads(line)  # 把JSON字符串解析成Python字典
                # 使用图像文件名作为键存储数据
                fig_map[data["image"]] = data
    
    return fig_map


def load_test_records(test_path: Path, limit: int) -> list[dict[str, Any]]:
    """
    加载测试记录（测试集）。
    
    【什么是测试记录？】
    - 每条记录包含：image_id（图像ID）、question（问题）、bbox（标准答案框）
    - bbox是Ground Truth（真值/标准答案），用于计算IoU评估模型表现
    
    参数:
        test_path: test.jsonl文件的路径
        limit: 最多加载多少条记录（用于调试，小规模测试）
    
    返回值:
        list[dict[str, Any]]: 测试记录列表
        每条记录格式：{"image_id": "0001.jpg", "question": "找出所有建筑", "bbox": [x1,y1,x2,y2]}
    """
    records = []
    
    with test_path.open("r", encoding="utf-8") as f:
        # enumerate(f) 会给每一行编号，idx从0开始
        for idx, line in enumerate(f):
            if idx >= limit:  # 超过限制数量就停止
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    return records


# ============================================================
# 【第三部分：实体坐标处理函数】
# ============================================================
# 这部分负责处理实体位置信息，包括：
# 1. build_entities_text: 构建文字提示（把坐标信息转成文字描述）
# 2. compute_centroid: 计算焦点中心点（用于blur/blackout/crop方法的圆形区域）

def build_entities_text(fig_data: dict[str, Any], img_width: int, img_height: int) -> str:
    """
    把实体数据转换成文字描述，用于构建带坐标提示的prompt。
    
    【为什么要转换成文字？】
    - 模型只能理解文字，不能直接读取坐标数据
    - 需要把 "[0.5, 0.5]" 这种归一化坐标转成 "像素坐标 (500, 500)" 这样的描述
    
    【什么是归一化坐标？】
    - 归一化坐标是把坐标范围压缩到0-1之间
    - 计算方式：像素坐标 / 图像尺寸
    - 例如：像素(512, 384) / 图像(1024, 768) = (0.5, 0.5)
    - 好处：不同尺寸的图像可以用统一的方式表示位置
    
    参数:
        fig_data: 单个图像的实体数据字典
        img_width: 图像宽度（像素）
        img_height: 图像高度（像素）
    
    返回值:
        str: 格式化的文字描述，示例：
        '''
        Image size: 1024x768 pixels
        
        Entities and their pixel coordinates:
        - b1: 建筑物 -> center at pixel (500, 400)
        - v1: 车辆 -> center at pixel (200, 300)
        '''
    """
    # fig_data.get("entities", []) 表示获取entities字段，如果不存在就返回空列表
    entities = fig_data.get("entities", [])
    if not entities:
        return ""
    
    # 初始化一个列表，逐行构建文字
    lines = [
        f"Image size: {img_width}x{img_height} pixels",  # 第一行：图像尺寸
        "",
        "Entities and their pixel coordinates:",
    ]
    
    # 遍历每个实体，构建描述文字
    for ent in entities:
        eid = ent.get("id", "")           # 实体的ID（如"b1"表示building 1）
        desc = ent.get("description", "")  # 实体描述（如"建筑物"）
        center = ent.get("center", [])    # 归一化中心坐标 [x_norm, y_norm]
        
        if len(center) == 2:
            # 归一化坐标转像素坐标：0.5 * 1024 = 512
            px = int(center[0] * img_width)
            py = int(center[1] * img_height)
            lines.append(f"- {eid}: {desc} -> center at pixel ({px}, {py})")
        else:
            lines.append(f"- {eid}: {desc}")
    
    # 用换行符连接所有行，形成最终的文字描述
    return "\n".join(lines)


def compute_centroid(entities: list[dict[str, Any]], img_width: int, img_height: int) -> tuple[int, int]:
    """
    计算所有实体的"中心点"作为焦点区域圆心。
    
    【什么是焦点区域？】
    - blur/blackout/crop方法需要确定一个"焦点"，这个焦点是所有实体的几何中心
    - 圆形区域的圆心就是所有实体中心点的平均值
    
    【为什么用平均值？】
    - 假设有2个实体，一个在(100,100)，一个在(300,300)
    - 中心点 = ((100+300)/2, (100+300)/2) = (200, 200)
    - 这样可以保证圆形区域覆盖到所有实体
    
    参数:
        entities: 实体列表
        img_width: 图像宽度
        img_height: 图像高度
    
    返回值:
        tuple[int, int]: (center_x, center_y) 圆心坐标（像素）
    """
    if not entities:
        # 如果没有实体，返回图像中心点
        return img_width // 2, img_height // 2

    sum_x, sum_y = 0.0, 0.0  # 累加所有实体的x、y坐标
    count = 0               # 统计有有效坐标的实体数量
    
    for ent in entities:
        center = ent.get("center", [])
        if len(center) == 2:
            # 归一化坐标转像素坐标后累加
            sum_x += center[0] * img_width
            sum_y += center[1] * img_height
            count += 1

    if count == 0:
        # 如果没有任何有效坐标，返回图像中心
        return img_width // 2, img_height // 2

    # 返回平均值（几何中心）
    return int(sum_x / count), int(sum_y / count)


# ============================================================
# 【第四部分：图像预处理函数】
# ============================================================
# 这部分负责图像的三种预处理操作：
# 1. create_circular_mask: 创建一个圆形遮罩（用于blur/blackout/crop）
# 2. process_blur: 高斯模糊焦点区域
# 3. process_blackout: 涂黑焦点区域
# 4. process_crop: 裁切焦点区域

def create_circular_mask(img_width: int, img_height: int, center_x: int, center_y: int, radius: int) -> np.ndarray:
    """
    创建一个圆形遮罩（Mask）。
    
    【什么是Mask？】
    - Mask是一个和原图同样大小的"黑白"图像
    - 白色(255)表示"保留原图"的区域
    - 黑色(0)表示"不保留原图"的区域
    - 通过Mask可以实现"局部处理"：只处理圆圈内，其他部分不变
    
    【参数解释】
    - center_x, center_y: 圆心坐标
    - radius: 圆的半径
    - np.zeros((h, w), dtype=np.uint8): 创建一个HxW的全黑图像
    
    【返回值】
    - np.ndarray: HxW的灰度图，白色圆圈内是255，其他是0
    """
    # 创建一个全黑的画布（H行 x W列）
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    # 在画布上画一个白色实心圆
    # cv2.circle(画布, 圆心, 半径, 颜色, -1表示填充)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    return mask


def process_blur(image_path: Path, fig_data: dict[str, Any], output_path: Path) -> Path:
    """
    【Blur方法核心】模糊焦点区域的图像。
    
    【处理流程图】
    原图 ──> 高斯模糊(整张图) ──┬──> 混合结果 ──> 保存
          └──> Mask遮罩 ────┘
          Mask=1的区域保留原图，Mask=0的区域使用模糊版本
    
    【详细步骤】
    1. 读取原图
    2. 计算焦点中心（所有实体的几何中心）
    3. 创建圆形Mask
    4. 对整张图做高斯模糊
    5. 用Mask混合：Mask内保留原图，Mask外使用模糊版本
    6. 保存结果
    
    【为什么这样做？】
    - 模糊目标区域后，模型无法清楚地看到目标
    - 迫使模型学会通过其他线索（上下文、位置提示等）来定位目标
    - 模拟"部分信息缺失"的场景
    """
    # 读取图像，cv2.imread返回BGR格式的numpy数组
    image = cv2.imread(str(image_path))
    if image is None:
        return image_path  # 读取失败就返回原路径

    # 获取图像尺寸：shape返回(H, W, C)，C是通道数(BGR=3)
    img_height, img_width = image.shape[:2]
    
    # 获取实体数据，计算焦点中心
    entities = fig_data.get("entities", [])
    center_x, center_y = compute_centroid(entities, img_width, img_height)
    
    # 圆形区域半径 = 图像短边的一半（保证覆盖所有实体）
    radius = min(img_width, img_height) // 2

    # 创建圆形遮罩
    mask = create_circular_mask(img_width, img_height, center_x, center_y, radius)
    
    # 对整个图像进行高斯模糊
    # BLUR_KERNEL_SIZE越大越模糊，一般用奇数如31、51
    blurred = cv2.GaussianBlur(image, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
    
    # 将单通道Mask转换为3通道（因为原图是3通道的BGR）
    mask_3ch = cv2.merge([mask, mask, mask])
    
    # 核心混合逻辑：np.where(condition, x, y)
    # 当mask_3ch > 0（即Mask=255白色区域）时，保留原图image
    # 否则（即Mask=0黑色区域）时，使用模糊版本blurred
    result = np.where(mask_3ch > 0, image, blurred)

    # 保存处理后的图像
    cv2.imwrite(str(output_path), result)
    return output_path


def process_blackout(image_path: Path, fig_data: dict[str, Any], output_path: Path) -> Path:
    """
    【Blackout方法核心】把焦点区域涂黑（完全遮挡）。
    
    【处理流程】
    原图 ──> Mask遮罩 ──┬──> 混合结果(黑) ──> 保存
                        └──> 全黑图(值为0)
    
    【与Blur的区别】
    - Blur：模糊处理，还保留部分信息
    - Blackout：完全涂黑，彻底看不见
    - 实验目的是测试模型在"完全无信息"情况下的表现
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return image_path

    img_height, img_width = image.shape[:2]
    entities = fig_data.get("entities", [])
    center_x, center_y = compute_centroid(entities, img_width, img_height)
    radius = min(img_width, img_height) // 2

    mask = create_circular_mask(img_width, img_height, center_x, center_y, radius)
    mask_3ch = cv2.merge([mask, mask, mask])
    
    # 区别在这里：用0（全黑）替代模糊版本
    result = np.where(mask_3ch > 0, image, 0)

    cv2.imwrite(str(output_path), result)
    return output_path


def process_crop(image_path: Path, fig_data: dict[str, Any], output_path: Path) -> tuple[Path, tuple[int, int, int, int]]:
    """
    【Crop方法核心】裁切图像到焦点区域。
    
    【处理流程】
    原图 ──> 根据焦点区域计算裁切坐标 ──> 只保留圆形区域 ──> 保存
    
    【坐标系统说明】
    图像坐标系:
    (0,0) ──────────────→ x
      │
      │    ┌─────────┐
      │    │ 裁切区域 │
      │    │  (ROI)  │
      │    └─────────┘
      │
      ↓ y
    
    【返回值说明】
    - 返回裁切后的图像路径和裁切区域坐标
    - crop_region = (x1, y1, x2, y2)
    - x1,y1是左上角，x2,y2是右下角
    - 注意：坐标是在原图坐标系中的位置，不是裁切后的新坐标
    
    【为什么需要返回crop_region？】
    - 因为后续计算IoU时，需要把GT坐标转换到裁切坐标系
    - crop_region记录了"裁切是从哪里开始的"
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return image_path, (0, 0, 0, 0)

    img_height, img_width = image.shape[:2]
    entities = fig_data.get("entities", [])
    center_x, center_y = compute_centroid(entities, img_width, img_height)
    radius = min(img_width, img_height) // 2

    # 计算裁切区域的边界
    # x1, y1是左上角，x2, y2是右下角
    # max/min确保裁切区域不会超出图像边界
    x1 = max(0, center_x - radius)
    y1 = max(0, center_y - radius)
    x2 = min(img_width, center_x + radius)
    y2 = min(img_height, center_y + radius)

    # numpy数组切片语法：image[y1:y2, x1:x2]
    # 注意是[行范围, 列范围]，即[y范围, x范围]
    cropped = image[y1:y2, x1:x2]
    
    cv2.imwrite(str(output_path), cropped)
    # 返回处理后的图像路径和裁切区域坐标（用于后续坐标转换）
    return output_path, (x1, y1, x2, y2)


# ============================================================
# 【第五部分：坐标转换函数（重要！）】
# ============================================================
# 这部分处理三种不同坐标系之间的转换，是评测准确性的关键：
# 1. denormalize_bbox: 模型归一化坐标 → 像素坐标
# 2. scale_bbox_to_resized: 原始像素坐标 → resize后坐标
# 3. transform_gt_bbox: 原始坐标 → 裁切坐标系

def denormalize_bbox(bbox: list[float], img_width: int, img_height: int) -> list[float]:
    """
    【模型输出坐标转换】将0-1000归一化坐标转换为像素坐标。
    
    【为什么需要这个转换？】
    - Qwen3-VL模型输出的bbox是0-1000范围的归一化坐标
    - 这是为了与模型内部处理分辨率解耦
    - 计算IoU时需要像素坐标，所以我们必须转换回来
    
    【示例说明】
    假设图像尺寸是1000x800，模型输出[250, 200, 750, 600]：
    - x1_new = 250 * 1000 / 1000 = 250（像素）
    - y1_new = 200 * 800 / 1000 = 160（像素）
    - x2_new = 750 * 1000 / 1000 = 750（像素）
    - y2_new = 600 * 800 / 1000 = 480（像素）
    
    【参数】
    - bbox: 模型输出的0-1000归一化坐标 [x1, y1, x2, y2]
    - img_width: 图像宽度（像素）
    - img_height: 图像高度（像素）
    
    【返回值】
    - list[float]: 像素坐标 [x1, y1, x2, y2]
    """
    return [
        bbox[0] * img_width / 1000,
        bbox[1] * img_height / 1000,
        bbox[2] * img_width / 1000,
        bbox[3] * img_height / 1000,
    ]


def scale_bbox_to_resized(
    bbox: list[float],
    orig_w: int,
    orig_h: int,
    resized_w: int,
    resized_h: int,
) -> list[float]:
    """
    【图像缩放坐标转换】将原始尺寸坐标转换为resize后的坐标。
    
    【为什么需要这个转换？】
    - 为了节省显存，大图像会被resize到较小尺寸（如1024x1024）
    - 但GT bbox是针对原图的坐标，不转换的话位置会错
    - 例如：原图2000x1500 → resize到1000x750
    - 原坐标(1000, 750) 对应 resize后坐标(500, 375)
    
    【计算公式】
    scale_x = resized_w / orig_w
    new_x = original_x * scale_x
    
    【参数】
    - bbox: 原始尺寸的坐标 [x1, y1, x2, y2]
    - orig_w, orig_h: 原始图像尺寸
    - resized_w, resized_h: resize后的图像尺寸
    
    【返回值】
    - list[float]: resize后坐标系中的坐标
    """
    # 计算缩放比例
    scale_x = resized_w / orig_w
    scale_y = resized_h / orig_h
    
    # 应用缩放比例到每个坐标点
    return [
        bbox[0] * scale_x,
        bbox[1] * scale_y,
        bbox[2] * scale_x,
        bbox[3] * scale_y,
    ]


def transform_gt_bbox(
    gt_bbox: list[float],
    crop_region: tuple[int, int, int, int],
    crop_width: int,
    crop_height: int,
) -> list[float]:
    """
    【裁切坐标转换】将原始图像坐标转换为裁切图像坐标。
    
    【这是crop方法特有的关键转换】
    - 假设原图1000x1000，裁切区域(250, 250)到(750, 750)
    - GT bbox在原图中是(300, 300, 500, 500)
    - 裁切后，图像左边界从x=250变成x=0
    - 所以新的bbox = (300-250, 300-250, 500-250, 500-250) = (50, 50, 250, 250)
    
    【示意图】
    原图坐标系:                    裁切坐标系:
    (0,0)                         (50,50) ┌────────┐
        ┌───────────────┐                 │  GT   │
        │               │                 │  bbox │
        │   ┌─────────┐ │                 └────────┘
        │   │ 裁切区域 │ │                 
        │   │(250,250)│ │          (0,0)──────────→ x
        │   └─────────┘ │
        │               │          
        └───────────────┘
    
    【参数】
    - gt_bbox: [x1, y1, x2, y2] 在原始图像坐标系中
    - crop_region: (x1, y1, x2, y2) 裁切区域的边界（相对于原图）
    - crop_width: 裁切图像宽度
    - crop_height: 裁切图像高度
    
    【返回值】
    - list[float]: [x1, y1, x2, y2] 在裁切图像坐标系中
    """
    # 裁切区域的左上角坐标
    cx1, cy1 = crop_region[0], crop_region[1]

    # 减去裁切起始坐标，完成坐标系转换
    gt_new = [
        gt_bbox[0] - cx1,  # x1: 从原图坐标转为裁切坐标
        gt_bbox[1] - cy1,  # y1
        gt_bbox[2] - cx1,  # x2
        gt_bbox[3] - cy1,  # y2
    ]

    # 裁切后的图像可能比原GT bbox还小，确保坐标不越界
    gt_new[0] = max(0, min(crop_width, gt_new[0]))
    gt_new[1] = max(0, min(crop_height, gt_new[1]))
    gt_new[2] = max(0, min(crop_width, gt_new[2]))
    gt_new[3] = max(0, min(crop_height, gt_new[3]))

    return gt_new


# ============================================================
# 【第六部分：IoU计算函数】
# ============================================================
# IoU (Intersection over Union) 是目标检测的标准评估指标
# 用于衡量预测框和真实框的重合程度

def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """
    【核心评估指标】计算两个矩形框的IoU（交并比）。
    
    【什么是IoU？】
    IoU = 交集面积 / 并集面积
    - IoU = 1.0：完美预测，两个框完全重合
    - IoU = 0.0：完全没重合
    - IoU > 0.5：一般认为预测正确
    - IoU > 0.7：预测很准确
    - IoU > 0.9：预测非常精确
    
    【IoU计算原理图】
    
    Box A      Box B         交集          并集
    ┌───┐     ┌───┐       ┌─┐           ┌─┐┌─┐
    │   │     │   │       └─┘           └─┘│ │
    │   └──┬──┘   │                     ┌─┘ └─┘
    └──────┴──────┘                     └─────┘
    
    【参数】
    - box_a: 第一个框的坐标 [x1, y1, x2, y2]
    - box_b: 第二个框的坐标 [x1, y1, x2, y2]
    - x1, y1 是左上角坐标
    - x2, y2 是右下角坐标
    
    【返回值】
    - float: IoU值，范围0.0到1.0
    """
    # 计算交集的左上角坐标（两个框左上角的最大值）
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    
    # 计算交集的右下角坐标（两个框右下角的最小值）
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    # 计算交集面积
    # 注意：可能没有交集（x2 < x1 或 y2 < y1），这时面积为0
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    
    if inter_area == 0:
        return 0.0  # 没有交集，IoU直接为0

    # 计算各自框的面积：宽 × 高
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    
    # 并集面积 = A面积 + B面积 - 交集面积
    # (减去交集避免重复计算)
    union_area = area_a + area_b - inter_area

    # IoU = 交集面积 / 并集面积
    return inter_area / union_area if union_area > 0 else 0.0


# ============================================================
# 模型输出解析
# ============================================================

def extract_bboxes_from_output(raw_output: str) -> list[list[float]]:
    """
    从模型的原始输出文本中提取检测框坐标。
    
    【这个函数在做什么？】
    模型输出的是一段文字，不是直接的坐标数字。
    这段文字可能包含：
    - 一段思考过程（<think>...</think>标签，Thinking模型特有）
    - 一段解释文字
    - 一个JSON数组，包含检测结果，如 [{"bbox_2d": [x1, y1, x2, y2]}, ...]
    
    这个函数需要：
    1. 去掉思考过程（如果有）
    2. 找到JSON数组的位置
    3. 从JSON中提取坐标
    
    【为什么要从后向前找？】
    因为模型通常先输出思考/解释，最后才输出最终答案。
    从后往前找可以更快地找到答案。
    
    【参数】
    - raw_output: 模型的原始文字输出
    
    【返回值】
    - list[list[float]]: 提取到的bbox坐标列表
      示例: [[250.0, 200.0, 750.0, 600.0], [100.0, 100.0, 200.0, 200.0]]
      如果没有找到，返回空列表 []
    """
    import re  # 正则表达式库，用于模式匹配
    
    # 第一步：移除Thinking模型的<think>...</think>思考标签
    # re.sub(模式, 替换内容, 原始字符串)
    # r"<think>.*?</think>" 匹配从<think>到</think>的所有内容
    # flags=re.DOTALL 让.也能匹配换行符
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
    
    # 第二步：从后向前查找JSON数组
    # end变量标记要查找的结束位置
    end = len(cleaned)
    
    # while循环：不断从后向前找']'字符
    while end > 0:
        # rfind从后向前查找']'，返回其索引位置
        end = cleaned.rfind("]", 0, end)
        if end == -1:
            return []  # 没找到']'，说明没有JSON数组
        
        # 找到']'后，向前寻找匹配的'['（括号匹配算法）
        # depth记录括号嵌套深度
        # 每遇到一个']'深度+1，每遇到一个'['深度-1
        # 当depth回到0时，说明找到了匹配的'['
        depth = 1
        start = end - 1
        while start >= 0 and depth > 0:
            if cleaned[start] == "]":
                depth += 1      # 遇到']'，嵌套深度增加
            elif cleaned[start] == "[":
                depth -= 1      # 遇到'['，嵌套深度减少
            start -= 1          # 继续向前搜索
        
        if depth != 0:
            continue  # 没有找到匹配的'['，继续查找下一个']'
        
        # start现在指向'['的前一个位置，所以+1回到'['
        start += 1
        json_str = cleaned[start:end + 1]  # 截取JSON字符串
        
        # 尝试解析JSON
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            continue  # JSON格式错误，继续查找
        
        # 确保解析结果是列表
        if not isinstance(parsed, list):
            continue
        
        # 第三步：从JSON列表中提取bbox坐标
        bboxes = []
        for item in parsed:
            # 每个item应该是字典，且包含"bbox_2d"键
            if isinstance(item, dict) and "bbox_2d" in item:
                coords = item["bbox_2d"]
                # 确保坐标是4个数的列表 [x1, y1, x2, y2]
                if isinstance(coords, list) and len(coords) == 4:
                    try:
                        # 转换为浮点数
                        bboxes.append([float(c) for c in coords])
                    except (ValueError, TypeError):
                        continue  # 转换失败，跳过这个item
        
        # 如果成功提取到bbox，立即返回（只取第一个有效JSON）
        if bboxes:
            return bboxes
    
    return []  # 全部搜索完毕都没找到，返回空列表


# ============================================================
# 图像resize
# ============================================================

def resize_image_if_needed(image_path: Path, max_dim: int = MAX_IMAGE_DIM) -> tuple[Path, int, int, int, int]:
    """
    【图像尺寸调整】如果图像太大，缩小到指定的最大尺寸。
    
    【为什么要resize？】
    - 大图像（如4000x3000）会占用大量显存（GPU内存）
    - 显存不够会导致程序崩溃（OOM - Out Of Memory）
    - 缩小图像可以节省显存，加快推理速度
    - 但也不能缩太小，否则模型看不清图像细节
    
    【什么是等比例缩放？】
    - 保持图像的宽高比不变，等比例缩小
    - 例如：2000x1000 的图像，max_dim=1000
    - scale = 1000 / 2000 = 0.5
    - new_w = 2000 * 0.5 = 1000, new_h = 1000 * 0.5 = 500
    - 结果：1000x500（宽高比仍为2:1）
    
    【什么是LANCZOS重采样？】
    - 一种高质量的图像缩放算法
    - 比默认的双线性插值更清晰，适合缩小图像
    - 名字来源于数学家Lanczos，不需要深入理解
    
    【返回值说明】
    - 返回5个值的元组 (tuple)：
      1. resized_path: 处理后的图像路径（可能是临时文件）
      2. resized_w: resize后的宽度
      3. resized_h: resize后的高度
      4. orig_w: 原始图像宽度
      5. orig_h: 原始图像高度
    
    【注意】
    - 如果图像不需要resize，返回原路径，orig和resized尺寸相同
    - resize后的图像保存在 /tmp 目录（临时目录）
    - 调用方负责删除临时文件（在run_inference中处理）
    """
    # 用Pillow打开图像，获取尺寸
    with Image.open(image_path) as img:
        w, h = img.size  # img.size返回(width, height)
        
        # 判断是否需要resize：如果最大边 <= max_dim就不需要
        if max(w, h) <= max_dim:
            return image_path, w, h, w, h  # 不需要resize，返回原路径和原始尺寸

        # 计算缩放比例：保持宽高比，让最大边等于max_dim
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 使用LANCZOS算法进行高质量缩放
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 保存到临时目录
        temp_path = Path("/tmp") / f"resized_{image_path.name}"
        resized.save(temp_path, quality=90)  # quality=90是JPEG质量，90是高质量
        return temp_path, new_w, new_h, w, h


# ============================================================
# 模型推理
# ============================================================

def format_prompt(prompt_text: str, image_path: str) -> list[dict]:
    """
    【格式化输入提示】将文字提示和图像路径组合成模型能理解的格式。
    
    【什么是对话格式（Chat Format）？】
    - 大语言模型使用"对话"格式来接收输入
    - 每条消息有"角色"(role)和"内容"(content)
    - role="user" 表示这是用户说的话
    - content里可以包含多种类型：文字、图像等
    
    【Qwen3-VL的输入格式示例】
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/path/to/image.jpg"},
                {"type": "text", "text": "请找出图中的建筑物，返回bbox坐标"}
            ]
        }
    ]
    
    这个格式告诉模型：
    1. 这是一条用户消息
    2. 内容包含一张图片和一段文字问题
    
    【参数】
    - prompt_text: 文字提示（包含问题和bbox输出格式要求）
    - image_path: 图像文件的路径
    
    【返回值】
    - list[dict]: 符合Qwen3-VL格式的对话列表
    """
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text},
        ],
    }]


def run_inference(
    model,
    processor,
    image_path: Path,
    prompt_text: str,
    max_tokens: int = MAX_TOKENS,
) -> tuple[str, int, int, int, int]:
    """
    【模型推理核心】运行Qwen3-VL模型，让它根据图像回答问题。
    
    【推理的完整流程】
    1. resize图像（如果太大的话）
    2. 构建对话格式的输入（图像+文字）
    3. 用processor把输入转成模型能理解的数字（token化）
    4. 把输入数据放到GPU上
    5. 调用模型的generate方法生成回答
    6. 从生成的token中提取出文字
    7. 清理临时文件
    8. 返回模型的回答文字和图像尺寸信息
    
    【什么是token化（tokenize）？】
    - 模型不认识文字，只认识数字
    - token化就是把文字切成"词元"（token），再转成数字
    - 例如 "找出建筑" → [找到, 建筑] → [12345, 67890]
    
    【什么是generate方法？】
    - 模型自回归生成文字：每次预测一个token，直到结束
    - max_new_tokens限制最多生成多少个token
    - 生成的token序列 = 输入token + 新生成的token
    
    【什么是torch.no_grad()？】
    - 告诉PyTorch不需要计算梯度（反向传播）
    - 因为推理（inference）不需要训练，省去梯度计算可以节省显存
    - 就像告诉程序"我现在只是在用模型，不是在训练它"
    
    【参数】
    - model: 已加载的Qwen3-VL模型
    - processor: 模型的预处理器（负责token化和图像处理）
    - image_path: 输入图像路径
    - prompt_text: 文字提示
    - max_tokens: 最多生成多少个token（默认值在config中定义）
    
    【返回值】
    - tuple[str, int, int, int, int]:
      1. output_text: 模型输出的文字回答
      2. resized_w: 推理时使用的图像宽度
      3. resized_h: 推理时使用的图像高度
      4. orig_w: 原始图像宽度
      5. orig_h: 原始图像高度
    """
    # 第一步：如果图像太大就缩小（节省显存）
    resized_path, resized_w, resized_h, orig_w, orig_h = resize_image_if_needed(image_path)

    # 第二步：构建对话格式的输入
    messages = format_prompt(prompt_text, str(resized_path))
    
    # 第三步：用processor处理输入
    # apply_chat_template 会：
    #   1. 把对话格式转成模型内部的prompt格式
    #   2. 把图像转成模型需要的像素值
    #   3. tokenize文字部分
    #   4. 拼接成一个完整的输入张量（tensor）
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,              # 是否token化文字
        add_generation_prompt=True, # 是否在末尾添加生成提示（告诉模型该生成回答了）
        return_dict=True,           # 返回字典格式
        return_tensors="pt",        # 返回PyTorch张量（pt = PyTorch）
    )
    
    # 第四步：把输入数据放到GPU上（与模型在同一设备）
    inputs = inputs.to(model.device)

    # 第五步：运行模型生成回答
    # torch.no_grad()：不计算梯度，节省显存
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    # generated_ids的形状: [1, 输入长度 + 生成长度]
    # 包含了输入token和新生成的token

    # 第六步：从generated_ids中提取出新生成的部分（去掉输入部分）
    # inputs.input_ids.shape = [1, 输入长度]
    # generated_ids.shape = [1, 输入长度 + 生成长度]
    # 我们只需要 [输入长度:] 之后的部分，即模型的输出
    generated_ids_trimmed = [
        out_ids[len(in_ids):]  # 切片：从输入长度开始到末尾
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # 第七步：把token转回文字
    # batch_decode: 把token ID序列解码成文字
    # skip_special_tokens=True: 不显示特殊token（如[CLS], [SEP]等）
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]  # [0]因为batch_size=1，取第一个结果

    # 第八步：清理临时文件（如果resize产生了临时文件的话）
    if resized_path != image_path and resized_path.exists():
        resized_path.unlink()  # unlink()是删除文件的方法

    return output_text, resized_w, resized_h, orig_w, orig_h


# ============================================================
# 可视化辅助函数
# ============================================================

def draw_bboxes_on_image(
    image: np.ndarray,
    gt_bbox: list[float] | None,
    pred_bbox: list[float] | None,
    img_w: int,
    img_h: int,
    is_norm: bool = True,
) -> np.ndarray:
    """
    【可视化函数】在图像上绘制Ground Truth框（绿色）和预测框（红色）。
    
    【为什么需要可视化？】
    - 光看IoU数字不够直观
    - 画出两个框可以一目了然地看到预测是否准确
    - 绿色框 = 标准答案，红色框 = 模型预测
    
    【OpenCV的绘图函数】
    - cv2.rectangle(图像, 左上角, 右下角, 颜色, 线宽)
    - cv2.putText(图像, 文字, 位置, 字体, 大小, 颜色, 线宽)
    - 颜色格式：(B, G, R)，注意OpenCV用BGR不是RGB
      - 绿色 = (0, 255, 0)
      - 红色 = (0, 0, 255)
    
    【参数说明】
    - image: 原始图像（numpy数组）
    - gt_bbox: Ground Truth框的坐标 [x1, y1, x2, y2]（可以是None）
    - pred_bbox: 预测框的坐标 [x1, y1, x2, y2]（可以是None）
    - img_w, img_h: 图像的宽高
    - is_norm: pred_bbox是否是归一化坐标（0-1000范围）
      - True: 需要先用denormalize_bbox转为像素坐标
      - False: 已经是像素坐标，直接使用
    
    【返回值】
    - np.ndarray: 绘制了标注框的图像
    """
    # 复制原图，避免修改原始图像（copy是为了不影响原图）
    vis = image.copy()

    # 绘制Ground Truth框（绿色，较粗的线）
    if gt_bbox and len(gt_bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in gt_bbox]
        # 画绿色矩形框，线宽=3
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # 在框的左上角上方标注"GT"文字
        cv2.putText(vis, "GT", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 绘制预测框（红色，较细的线）
    if pred_bbox and len(pred_bbox) == 4:
        # 根据is_norm决定是否需要坐标转换
        if is_norm:
            # 模型输出的是0-1000归一化坐标，需要转为像素坐标
            pred_pixel = denormalize_bbox(pred_bbox, img_w, img_h)
        else:
            # 已经是像素坐标（crop方法的情况）
            pred_pixel = pred_bbox
        x1, y1, x2, y2 = [int(v) for v in pred_pixel]
        # 画红色矩形框，线宽=2（比GT细一点，方便区分）
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # 在框的右下角下方标注"Pred"文字
        cv2.putText(vis, "Pred", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return vis


# ============================================================
# 主评测函数
# ============================================================

def run_eval(model_key: str, method: str):
    """
    【主评测函数】运行单个模型的单个方法评测。
    
    【这个函数的整体流程】
    ┌─────────────────────────────────────────────────┐
    │ 1. 参数校验：检查模型名和方法名是否合法           │
    │ 2. 配置加载：获取模型路径、输出目录等配置         │
    │ 3. 断点续传：检查是否有之前未完成的结果            │
    │ 4. 数据加载：读取测试集                           │
    │ 5. 模型加载：把模型从磁盘加载到GPU显存            │
    │ 6. 逐条评测：对每条测试数据：                      │
    │    a. 根据方法预处理图像/构建prompt                │
    │    b. 运行模型推理                                │
    │    c. 解析模型输出，提取bbox                      │
    │    d. 计算IoU，判断是否正确                       │
    │    e. 生成可视化图像                              │
    │    f. 保存结果到JSONL文件                         │
    └─────────────────────────────────────────────────┘
    
    【什么是断点续传？】
    - 评测可能需要很长时间（几百张图，每张几十秒）
    - 如果程序中途崩溃，不希望从头开始
    - 所以每次开始前先检查哪些已经评测过了
    - 只评测剩余未完成的
    
    【参数】
    - model_key: 模型的简称，如 "2b_inst", "8b_inst", "8b_think"
    - method: 评测方法名，如 "baseline", "prompt", "blur", "blackout", "crop"
    
    【用法示例】
    - python eval_study.py 8b_inst baseline
    - python eval_study.py 2b_inst blur
    """
    # ============================================================
    # 第一步：参数校验
    # ============================================================
    # 检查用户输入的模型名和方法名是否在config.py中定义
    if model_key not in MODELS:
        print(f"错误: 未知模型 {model_key}")
        print(f"可用模型: {', '.join(MODELS.keys())}")
        sys.exit(1)  # sys.exit(1) 表示异常退出，1是退出码

    if method not in METHODS:
        print(f"错误: 未知方法 {method}")
        print(f"可用方法: {', '.join(METHODS.keys())}")
        sys.exit(1)

    # ============================================================
    # 第二步：获取配置信息
    # ============================================================
    # 从config.py的字典中获取当前模型和方法的配置
    model_cfg = MODELS[model_key]       # 如 {"name": "Qwen/Qwen3-VL-8B-Instruct", "output_dir": "results_8b_inst"}
    method_cfg = METHODS[method]        # 如 {"need_fig": True, "need_process": True, "results_file": "blur_results.jsonl", ...}
    model_name = model_cfg["name"]      # HuggingFace模型的完整名称（用于加载模型）
    output_dir = Path(BASE_DIR) / model_cfg["output_dir"]  # 结果输出目录
    data_dir = Path(DATA_DIR)           # 数据目录

    # 创建输出目录（如果不存在的话）
    # parents=True: 如果父目录也不存在，一起创建
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 第三步：准备数据和目录
    # ============================================================
    # 结果文件路径（如 results_8b_inst/blur_results.jsonl）
    results_path = output_dir / method_cfg["results_file"]
    
    # 如果方法需要实体数据（prompt/blur/blackout/crop都需要），就加载
    # baseline方法不需要实体数据（need_fig=False）
    fig_map = load_fig_data(data_dir / FIG_JSONL) if method_cfg["need_fig"] else {}

    # 预处理图像的保存目录（只有blur/blackout/crop需要）
    processed_dir = None
    if method_cfg.get("need_process"):
        processed_dir = output_dir / method_cfg["process_dir"]
        processed_dir.mkdir(exist_ok=True)

    # 可视化图像的保存目录（所有方法都需要）
    vis_dir = output_dir / "vis_images"
    vis_dir.mkdir(exist_ok=True)

    # ============================================================
    # 第四步：断点续传 - 检查已有结果
    # ============================================================
    # 如果结果文件已存在，说明之前跑过一部分
    # 读取已有结果，避免重复评测
    all_results: list[dict[str, Any]] = []
    processed_ids: set[str] = set()  # 已评测过的图像ID集合（用set查找更快）
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    all_results.append(rec)
                    processed_ids.add(rec.get("image_id", ""))
        print(f"续传: 发现 {len(all_results)} 条已有结果")

    # ============================================================
    # 第五步：加载测试集，计算剩余待处理数据
    # ============================================================
    records = load_test_records(data_dir / TEST_JSONL, NUM_SAMPLES)
    # 过滤掉已评测过的记录
    remaining = [r for r in records if r.get("image_id", "") not in processed_ids]
    print(f"加载 {len(records)} 条记录, 剩余 {len(remaining)} 条待处理")

    if not remaining:
        print("全部完成!")
        return  # 没有待处理的，直接返回

    # ============================================================
    # 第六步：加载模型到GPU
    # ============================================================
    # 这一步很慢（需要几分钟），因为要把模型从磁盘读取到显存
    # torch.bfloat16: 使用16位浮点数，节省一半显存
    # device_map="auto": 自动分配到可用的GPU上
    print(f"加载模型: {model_name}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # bfloat16是深度学习常用的低精度格式
        device_map="auto",            # 自动选择GPU
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # ============================================================
    # 第七步：逐条评测（核心循环）
    # ============================================================
    # 用 "a"（append）模式打开文件，这样可以在运行过程中逐步写入结果
    # 即使程序崩溃，已经写入的结果也不会丢失
    with results_path.open("a", encoding="utf-8") as f:
        for record in remaining:
            # 从测试记录中提取信息
            image_id = record.get("image_id", "")   # 图像文件名
            question = record.get("question", "")    # 问题文本
            gt_bbox = record.get("bbox")             # 标准答案框
            image_path = data_dir / image_id         # 图像完整路径

            # 打印进度，格式如: [5/100] 0000000_00098_d_0000001.jpg
            print(f"\n[{len(all_results)+1}/{len(records)}] {image_id}")

            # 初始化结果记录字典
            result_rec: dict[str, Any] = {
                "question_id": record.get("question_id"),
                "image_id": image_id,
                "question": question,
                "gt_bbox": gt_bbox,
            }

            # 检查图像文件是否存在
            if not image_path.exists():
                result_rec.update({
                    "status": "image_not_found",
                    "pred_bbox": None,
                    "iou": 0.0,
                    "correct": False,
                })
                all_results.append(result_rec)
                f.write(json.dumps(result_rec, ensure_ascii=False) + "\n")
                continue  # 跳过这张图，处理下一张

            # 初始化一些变量
            fig_data = fig_map.get(image_id, {})  # 获取当前图像的实体数据
            inference_path = image_path             # 推理用的图像路径（可能被修改）
            gt_for_iou = gt_bbox                    # 用于计算IoU的GT坐标（可能被转换）
            crop_region = None                      # 裁切区域（仅crop方法使用）

            # ============================================================
            # 第七步A：根据方法构建prompt和预处理图像
            # ============================================================
            # 每种方法的处理方式不同：
            #
            # baseline: 不修改图像，使用标准prompt
            # prompt:   不修改图像，但在prompt中加入实体位置信息
            # blur:     对图像做模糊处理，使用标准prompt
            # blackout: 对图像做涂黑处理，使用标准prompt
            # crop:     对图像做裁切处理，需要转换GT坐标

            if method == "prompt":
                # Prompt方法：不修改图像，但修改prompt
                with Image.open(image_path) as img:
                    img_w, img_h = img.size
                # 把实体坐标信息转成文字描述
                entities_text = build_entities_text(fig_data, img_w, img_h)
                # 拼接成完整的prompt
                full_prompt = f"{entities_text}\n\nQuestion: {question}"
                prompt_text = BBOX_PROMPT_WITH_ENTITIES.format(question=full_prompt)

            elif method == "blur":
                # Blur方法：模糊焦点区域，使用标准prompt
                processed_path = processed_dir / f"blur_{image_id}"
                inference_path = process_blur(image_path, fig_data, processed_path)
                prompt_text = BBOX_PROMPT_TEMPLATE.format(question=question)

            elif method == "blackout":
                # Blackout方法：涂黑焦点区域，使用标准prompt
                processed_path = processed_dir / f"blackout_{image_id}"
                inference_path = process_blackout(image_path, fig_data, processed_path)
                prompt_text = BBOX_PROMPT_TEMPLATE.format(question=question)

            elif method == "crop":
                # Crop方法：裁切焦点区域
                processed_path = processed_dir / f"crop_{image_id}"
                inference_path, crop_region = process_crop(image_path, fig_data, processed_path)

                # 关键：crop方法需要转换GT坐标到裁切坐标系
                if gt_bbox and len(gt_bbox) == 4:
                    crop_w = crop_region[2] - crop_region[0]
                    crop_h = crop_region[3] - crop_region[1]
                    gt_for_iou = transform_gt_bbox(gt_bbox, crop_region, crop_w, crop_h)
                    # 保存转换后的坐标信息，方便后续分析
                    result_rec["gt_bbox_cropped"] = gt_for_iou
                    result_rec["crop_region"] = list(crop_region)

                prompt_text = BBOX_PROMPT_TEMPLATE.format(question=question)

            else:
                # Baseline和其他方法：使用标准prompt
                prompt_text = BBOX_PROMPT_TEMPLATE.format(question=question)

            # ============================================================
            # 第七步B：运行模型推理
            # ============================================================
            try:
                # 调用模型，获取输出文字和图像尺寸信息
                raw_output, img_w, img_h, orig_w, orig_h = run_inference(
                    model, processor, inference_path, prompt_text, MAX_TOKENS
                )
                # 记录模型的原始输出（方便调试和分析）
                result_rec["raw_output"] = raw_output
                result_rec["img_size_used"] = f"{img_w}x{img_h}"
                result_rec["orig_size"] = f"{orig_w}x{orig_h}"

                # 从模型输出中解析bbox坐标
                bboxes = extract_bboxes_from_output(raw_output)
                result_rec["num_pred_bboxes"] = len(bboxes)

                # ============================================================
                # 第七步C：评估预测结果
                # ============================================================
                if not bboxes:
                    # 情况1：模型没有输出有效的bbox
                    result_rec.update({
                        "status": "no_bbox_parsed",
                        "pred_bbox": None,
                        "iou": 0.0,
                        "correct": False,
                    })
                    print("  未解析到bbox")

                elif gt_for_iou and len(gt_for_iou) == 4:
                    # 情况2：成功解析到bbox，且有GT标准答案
                    # ============================================================
                    # 计算IoU（关键坐标转换逻辑）
                    #
                    # 坐标对齐流程：
                    #   GT坐标 → 缩放到推理图像尺寸 → 与预测框比较
                    #
                    # 对于crop方法:
                    #   gt_for_iou 已经在裁切坐标系中（前面转换过了）
                    #   需要再缩放到推理图像尺寸（可能被resize过）
                    #
                    # 对于其他方法:
                    #   gt_for_iou = 原始GT坐标
                    #   需要缩放到推理图像尺寸（可能被resize过）
                    # ============================================================

                    if method == "crop":
                        crop_w = crop_region[2] - crop_region[0]
                        crop_h = crop_region[3] - crop_region[1]
                        gt_scaled = scale_bbox_to_resized(gt_for_iou, crop_w, crop_h, img_w, img_h)
                    else:
                        gt_scaled = scale_bbox_to_resized(gt_for_iou, orig_w, orig_h, img_w, img_h)

                    # 如果模型输出了多个bbox，选IoU最高的那个作为最终预测
                    best_iou, best_box = 0.0, bboxes[0]
                    for pb in bboxes:
                        # 把模型输出的0-1000归一化坐标转为像素坐标
                        pb_pixel = denormalize_bbox(pb, img_w, img_h)
                        # 计算IoU
                        iou_val = compute_iou(gt_scaled, pb_pixel)
                        if iou_val > best_iou:
                            best_iou, best_box = iou_val, pb

                    # 保存评估结果
                    result_rec.update({
                        "pred_bbox": best_box,           # 最佳预测框（0-1000归一化坐标）
                        "gt_bbox_scaled": gt_scaled,     # GT框（resize后的像素坐标）
                        "iou": round(best_iou, 6),       # IoU值（保留6位小数）
                        "correct": best_iou >= IOU_THRESHOLD,  # 是否正确（IoU >= 阈值）
                        "status": "ok",
                    })
                    # 打印结果标记：OK=预测正确，MISS=预测错误
                    icon = "OK" if result_rec["correct"] else "MISS"
                    print(f"  -> IoU={best_iou:.4f} [{icon}]")

                else:
                    # 情况3：有预测但没有GT（不应该发生，但做防御性处理）
                    result_rec.update({
                        "pred_bbox": bboxes[0],
                        "iou": 0.0,
                        "correct": False,
                        "status": "no_gt_bbox",
                    })

            except Exception as e:
                # 情况4：推理过程中出错（如GPU显存不足、图像格式错误等）
                result_rec.update({
                    "status": "inference_failed",
                    "raw_output": str(e),
                    "pred_bbox": None,
                    "iou": 0.0,
                    "correct": False,
                })
                print(f"  [ERR] {e}")

            # ============================================================
            # 第七步D：生成可视化图像
            # ============================================================
            # 在原图上绘制GT框（绿色）和预测框（红色），方便肉眼检查
            try:
                img_cv = cv2.imread(str(inference_path))
                if img_cv is not None:
                    if method == "crop" and crop_region:
                        # crop方法：GT和预测都在裁切坐标系中
                        crop_w = crop_region[2] - crop_region[0]
                        crop_h = crop_region[3] - crop_region[1]
                        gt_vis = gt_for_iou
                        pred_pixel = denormalize_bbox(result_rec.get("pred_bbox", [0, 0, 0, 0]), img_w, img_h)
                        vis_img = draw_bboxes_on_image(img_cv, gt_vis, pred_pixel, img_w, img_h, is_norm=False)
                    else:
                        # 其他方法：GT是像素坐标，预测是归一化坐标
                        vis_img = draw_bboxes_on_image(img_cv, gt_for_iou, result_rec.get("pred_bbox"), img_w, img_h)
                    cv2.imwrite(str(vis_dir / f"vis_{image_id}"), vis_img)
            except Exception as e:
                print(f"  [VIS ERR] {e}")

            # ============================================================
            # 第七步E：保存结果
            # ============================================================
            all_results.append(result_rec)
            # 把当前这条结果写入JSONL文件（每条一行JSON）
            f.write(json.dumps(result_rec, ensure_ascii=False) + "\n")
            f.flush()  # flush()立即写入磁盘，而不是等缓冲区满

    print(f"\n结果已保存到: {results_path}")


# ============================================================
# 命令行入口
# ============================================================
# 【什么是 if __name__ == "__main__"？】
# - 这是Python的标准写法，表示"当这个文件作为主程序运行时"
# - 如果这个文件被其他文件 import，则不会执行这里的代码
# - 相当于C/Java的 main() 函数
#
# 【用法示例】
# python eval_study.py 8b_inst baseline
#   - sys.argv[0] = "eval_study.py"  （脚本名）
#   - sys.argv[1] = "8b_inst"        （模型key）
#   - sys.argv[2] = "baseline"       （方法名）

if __name__ == "__main__":
    # 检查命令行参数数量是否足够
    if len(sys.argv) < 3:
        print(f"用法: python {sys.argv[0]} <model_key> <method>")
        print(f"可用模型: {', '.join(MODELS.keys())}")
        print(f"可用方法: {', '.join(METHODS.keys())}")
        sys.exit(1)

    # 调用主评测函数
    run_eval(sys.argv[1], sys.argv[2])
