#!/usr/bin/env python3
"""
模型配置文件 - 定义所有要测试的模型和路径
适用于3090/4090服务器 (HuggingFace Transformers)

路径配置:
  - DATA_DIR:  数据目录，可通过环境变量 DATA_DIR 覆盖
  - BASE_DIR:  结果输出目录，可通过环境变量 BASE_DIR 覆盖
"""

import os
from pathlib import Path

# ============================================================
# 模型配置 - 3个Qwen3-VL Instruct模型
# ============================================================
MODELS = {
    "8b_inst": {
        "name": "Qwen/Qwen3-VL-8B-Instruct",
        "output_dir": "results_8b_inst",
    },
    "4b_inst": {
        "name": "Qwen/Qwen3-VL-4B-Instruct",
        "output_dir": "results_4b_inst",
    },
    "2b_inst": {
        "name": "Qwen/Qwen3-VL-2B-Instruct",
        "output_dir": "results_2b_inst",
    },
}

# ============================================================
# 数据路径 - 通过环境变量或默认值配置
# ============================================================
DATA_DIR = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data" / "MCP_test"))
TEST_JSONL = "50_MCP_test.jsonl"
FIG_JSONL = "Prompt_test.jsonl"

# ============================================================
# 基础目录 - 结果输出目录
# ============================================================
BASE_DIR = os.environ.get("BASE_DIR", str(Path(__file__).resolve().parent.parent / "Results"))

# ============================================================
# 常量配置
# ============================================================
MAX_TOKENS = 256
NUM_SAMPLES = 50
MAX_IMAGE_DIM = 1600
BLUR_KERNEL_SIZE = 51
IOU_THRESHOLD = 0.5

# ============================================================
# 5种方法配置
# ============================================================
METHODS = {
    "baseline": {
        "results_file": "baseline_results.jsonl",
        "need_fig": False,
        "need_process": False,
    },
    "prompt": {
        "results_file": "prompt_results.jsonl",
        "need_fig": True,
        "need_process": False,
    },
    "blur": {
        "results_file": "blur_results.jsonl",
        "need_fig": True,
        "need_process": True,
        "process_dir": "blur_images",
    },
    "blackout": {
        "results_file": "blackout_results.jsonl",
        "need_fig": True,
        "need_process": True,
        "process_dir": "blackout_images",
    },
    "crop": {
        "results_file": "crop_results.jsonl",
        "need_fig": True,
        "need_process": True,
        "process_dir": "crop_images",
    },
}

# ============================================================
# Prompt模板
# ============================================================

BBOX_PROMPT_TEMPLATE = """{question}

Please locate the target object in the image. Output ONLY a JSON array (no other text):
[{{"bbox_2d": [x1, y1, x2, y2], "label": "description"}}]

[x1,y1] is top-left corner, [x2,y2] is bottom-right corner, in absolute pixel coordinates."""

BBOX_PROMPT_WITH_ENTITIES = """{question}

IMPORTANT: The pixel coordinates above are the APPROXIMATE CENTER locations of each entity. The target object is NEAR one of these centers.

Your task: Find the EXACT bounding box of the target object described in the question. Look at the region around the matching entity's pixel coordinates.

Output ONLY a JSON array (no other text):
[{{"bbox_2d": [x1, y1, x2, y2], "label": "description"}}]

[x1,y1] is top-left corner, [x2,y2] is bottom-right corner, in absolute pixel coordinates."""
