#!/usr/bin/env python3
"""
统一评测脚本 - 支持6个Qwen3-VL模型 × 5种方法
适用于3090/4090服务器 (HuggingFace Transformers)

5种方法：
- baseline: 直接问问题，无预处理
- prompt: 带实体坐标提示
- blur: 圆形焦点区域高斯模糊
- blackout: 圆形焦点区域抹黑
- crop: 圆形焦点区域裁切
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from config import (
    BASE_DIR, BBOX_PROMPT_TEMPLATE, BBOX_PROMPT_WITH_ENTITIES, BLUR_KERNEL_SIZE,
    DATA_DIR, FIG_JSONL, IOU_THRESHOLD, MAX_IMAGE_DIM, MAX_TOKENS, METHODS,
    MODELS, NUM_SAMPLES, TEST_JSONL,
)


# ============================================================
# 数据加载
# ============================================================

def load_fig_data(fig_path: Path) -> dict[str, dict[str, Any]]:
    fig_map = {}
    with fig_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                fig_map[data["image"]] = data
    return fig_map


def load_test_records(test_path: Path, limit: int) -> list[dict[str, Any]]:
    records = []
    with test_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ============================================================
# 实体坐标处理
# ============================================================

def build_entities_text(fig_data: dict[str, Any], img_width: int, img_height: int) -> str:
    entities = fig_data.get("entities", [])
    if not entities:
        return ""
    lines = [
        f"Image size: {img_width}x{img_height} pixels",
        "",
        "Entities and their pixel coordinates:",
    ]
    for ent in entities:
        eid = ent.get("id", "")
        desc = ent.get("description", "")
        center = ent.get("center", [])
        if len(center) == 2:
            px = int(center[0] * img_width)
            py = int(center[1] * img_height)
            lines.append(f"- {eid}: {desc} -> center at pixel ({px}, {py})")
        else:
            lines.append(f"- {eid}: {desc}")
    return "\n".join(lines)


def compute_centroid(entities: list[dict[str, Any]], img_width: int, img_height: int) -> tuple[int, int]:
    if not entities:
        return img_width // 2, img_height // 2

    sum_x, sum_y = 0.0, 0.0
    count = 0
    for ent in entities:
        center = ent.get("center", [])
        if len(center) == 2:
            sum_x += center[0] * img_width
            sum_y += center[1] * img_height
            count += 1

    if count == 0:
        return img_width // 2, img_height // 2

    return int(sum_x / count), int(sum_y / count)


# ============================================================
# 图像预处理
# ============================================================

def create_circular_mask(img_width: int, img_height: int, center_x: int, center_y: int, radius: int) -> np.ndarray:
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    return mask


def process_blur(image_path: Path, fig_data: dict[str, Any], output_path: Path) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        return image_path

    img_height, img_width = image.shape[:2]
    entities = fig_data.get("entities", [])
    center_x, center_y = compute_centroid(entities, img_width, img_height)
    radius = min(img_width, img_height) // 2

    mask = create_circular_mask(img_width, img_height, center_x, center_y, radius)
    blurred = cv2.GaussianBlur(image, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
    mask_3ch = cv2.merge([mask, mask, mask])
    result = np.where(mask_3ch > 0, image, blurred)

    cv2.imwrite(str(output_path), result)
    return output_path


def process_blackout(image_path: Path, fig_data: dict[str, Any], output_path: Path) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        return image_path

    img_height, img_width = image.shape[:2]
    entities = fig_data.get("entities", [])
    center_x, center_y = compute_centroid(entities, img_width, img_height)
    radius = min(img_width, img_height) // 2

    mask = create_circular_mask(img_width, img_height, center_x, center_y, radius)
    mask_3ch = cv2.merge([mask, mask, mask])
    result = np.where(mask_3ch > 0, image, 0)

    cv2.imwrite(str(output_path), result)
    return output_path


def process_crop(image_path: Path, fig_data: dict[str, Any], output_path: Path) -> tuple[Path, tuple[int, int, int, int]]:
    """
    裁切图像到焦点区域，返回裁切后的图像路径和裁切区域坐标。

    Returns:
        (output_path, crop_region)
        crop_region = (x1, y1, x2, y2) 在原始图像坐标系中
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return image_path, (0, 0, 0, 0)

    img_height, img_width = image.shape[:2]
    entities = fig_data.get("entities", [])
    center_x, center_y = compute_centroid(entities, img_width, img_height)
    radius = min(img_width, img_height) // 2

    x1 = max(0, center_x - radius)
    y1 = max(0, center_y - radius)
    x2 = min(img_width, center_x + radius)
    y2 = min(img_height, center_y + radius)

    cropped = image[y1:y2, x1:x2]
    cv2.imwrite(str(output_path), cropped)
    return output_path, (x1, y1, x2, y2)


# ============================================================
# 坐标转换函数（关键！）
# ============================================================

def denormalize_bbox(bbox: list[float], img_width: int, img_height: int) -> list[float]:
    """
    将模型输出的0-1000归一化坐标转换为像素坐标。

    模型输出格式: [x1, y1, x2, y2] 范围0-1000
    像素坐标格式: [x1, y1, x2, y2] 范围0-img_width/0-img_height
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
    将bbox从原始尺寸坐标系缩放到resize后的坐标系。

    用于处理resize_image_if_needed()导致的坐标偏移。
    """
    scale_x = resized_w / orig_w
    scale_y = resized_h / orig_h
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
    将GT bbox从原始图像坐标系转换到裁切图像坐标系。

    Args:
        gt_bbox: [x1, y1, x2, y2] 在原始图像坐标系中
        crop_region: (x1, y1, x2, y2) 裁切区域在原始图像坐标系中
        crop_width: 裁切图像宽度
        crop_height: 裁切图像高度

    Returns:
        [x1, y1, x2, y2] 在裁切图像坐标系中
    """
    cx1, cy1 = crop_region[0], crop_region[1]

    gt_new = [
        gt_bbox[0] - cx1,
        gt_bbox[1] - cy1,
        gt_bbox[2] - cx1,
        gt_bbox[3] - cy1,
    ]

    gt_new[0] = max(0, min(crop_width, gt_new[0]))
    gt_new[1] = max(0, min(crop_height, gt_new[1]))
    gt_new[2] = max(0, min(crop_width, gt_new[2]))
    gt_new[3] = max(0, min(crop_height, gt_new[3]))

    return gt_new


# ============================================================
# IoU计算
# ============================================================

def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# ============================================================
# 模型输出解析 - Thinking模型优化版
# ============================================================

def extract_bbox_from_json_str(json_str: str) -> list[float] | None:
    """从JSON字符串中提取bbox（支持单个对象或数组）"""
    try:
        obj = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None

    # 单个对象: {"bbox_2d": [x1, y1, x2, y2]}
    if isinstance(obj, dict) and "bbox_2d" in obj:
        coords = obj["bbox_2d"]
        if isinstance(coords, list) and len(coords) == 4:
            try:
                return [float(c) for c in coords]
            except (ValueError, TypeError):
                pass

    # 数组格式: [{"bbox_2d": [...]}]
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and "bbox_2d" in item:
                coords = item["bbox_2d"]
                if isinstance(coords, list) and len(coords) == 4:
                    try:
                        return [float(c) for c in coords]
                    except (ValueError, TypeError):
                        continue

    return None


def extract_bboxes_from_output(raw_output: str) -> list[list[float]]:
    """
    从Thinking模型输出中提取bbox坐标。

    支持的格式：
    1. <think>...{"bbox_2d": [...]}...</think>  (坐标在思考过程中)
    2. <think>...</think> <grounding>{"bbox_2d": [...]}</grounding>
    3. <think>...</think> {"bbox_2d": [...]}
    4. <think>...</think> [{"bbox_2d": [...]}]
    5. 纯JSON输出（无think标签）
    """
    import re

    # 策略1: 使用正则表达式查找所有bbox_2d模式
    # 匹配 {"bbox_2d": [x1, y1, x2, y2]} 格式
    pattern = r'\{\s*"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\s*\}'
    matches = re.findall(pattern, raw_output)

    if matches:
        bboxes = []
        for match in matches:
            try:
                bbox = [float(x) for x in match]
                bboxes.append(bbox)
            except (ValueError, TypeError):
                continue
        if bboxes:
            return bboxes

    # 策略2: 在<think>标签内查找
    think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
    if think_match:
        think_content = think_match.group(1)
        bbox = extract_bbox_from_json_str(think_content)
        if bbox:
            return [bbox]

    # 策略3: 在<grounding>标签内查找
    grounding_match = re.search(r"<grounding>(.*?)</grounding>", raw_output, re.DOTALL)
    if grounding_match:
        grounding_content = grounding_match.group(1)
        bbox = extract_bbox_from_json_str(grounding_content)
        if bbox:
            return [bbox]

    # 策略4: 在整个输出中查找JSON数组
    start = 0
    while True:
        start = raw_output.find("[", start)
        if start == -1:
            break

        depth = 1
        end = start + 1
        while end < len(raw_output) and depth > 0:
            if raw_output[end] == "[":
                depth += 1
            elif raw_output[end] == "]":
                depth -= 1
            end += 1

        if depth == 0:
            json_str = raw_output[start:end]
            bbox = extract_bbox_from_json_str(json_str)
            if bbox:
                return [bbox]

        start += 1

    # 策略5: 查找单个JSON对象
    obj_pattern = r'\{[^{}]*"bbox_2d"[^{}]*\}'
    obj_matches = re.findall(obj_pattern, raw_output)
    for match in obj_matches:
        bbox = extract_bbox_from_json_str(match)
        if bbox:
            return [bbox]

    return []


# ============================================================
# 图像resize
# ============================================================

def resize_image_if_needed(image_path: Path, max_dim: int = MAX_IMAGE_DIM) -> tuple[Path, int, int, int, int]:
    """
    如果图像最大维度超过max_dim，resize图像。

    Returns:
        (resized_path, resized_w, resized_h, orig_w, orig_h)
    """
    with Image.open(image_path) as img:
        w, h = img.size
        if max(w, h) <= max_dim:
            return image_path, w, h, w, h

        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        temp_path = Path("/tmp") / f"resized_{image_path.name}"
        resized.save(temp_path, quality=90)
        return temp_path, new_w, new_h, w, h


# ============================================================
# 模型推理
# ============================================================

def format_prompt(prompt_text: str, image_path: str) -> list[dict]:
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
    运行模型推理。

    Returns:
        (output_text, resized_w, resized_h, orig_w, orig_h)
    """
    resized_path, resized_w, resized_h, orig_w, orig_h = resize_image_if_needed(image_path)

    messages = format_prompt(prompt_text, str(resized_path))
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=False,  # Thinking模型需要保留<think>标签
        clean_up_tokenization_spaces=False,
    )[0]

    if resized_path != image_path and resized_path.exists():
        resized_path.unlink()

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
    vis = image.copy()

    if gt_bbox and len(gt_bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in gt_bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(vis, "GT", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if pred_bbox and len(pred_bbox) == 4:
        if is_norm:
            pred_pixel = denormalize_bbox(pred_bbox, img_w, img_h)
        else:
            pred_pixel = pred_bbox
        x1, y1, x2, y2 = [int(v) for v in pred_pixel]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, "Pred", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return vis


# ============================================================
# 主评测函数
# ============================================================

def run_eval(model_key: str, method: str):
    """
    运行单个模型的单个方法评测。

    Args:
        model_key: 模型key (如 "8b_inst")
        method: 方法名 (如 "baseline", "prompt", "blur", "blackout", "crop")
    """
    if model_key not in MODELS:
        print(f"错误: 未知模型 {model_key}")
        print(f"可用模型: {', '.join(MODELS.keys())}")
        sys.exit(1)

    if method not in METHODS:
        print(f"错误: 未知方法 {method}")
        print(f"可用方法: {', '.join(METHODS.keys())}")
        sys.exit(1)

    model_cfg = MODELS[model_key]
    method_cfg = METHODS[method]
    model_name = model_cfg["name"]
    output_dir = Path(BASE_DIR) / model_cfg["output_dir"]
    data_dir = Path(DATA_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / method_cfg["results_file"]
    fig_map = load_fig_data(data_dir / FIG_JSONL) if method_cfg["need_fig"] else {}

    processed_dir = None
    if method_cfg.get("need_process"):
        processed_dir = output_dir / method_cfg["process_dir"]
        processed_dir.mkdir(exist_ok=True)

    vis_dir = output_dir / "vis_images"
    vis_dir.mkdir(exist_ok=True)

    all_results: list[dict[str, Any]] = []
    processed_ids: set[str] = set()
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    all_results.append(rec)
                    processed_ids.add(rec.get("image_id", ""))
        print(f"续传: 发现 {len(all_results)} 条已有结果")

    records = load_test_records(data_dir / TEST_JSONL, NUM_SAMPLES)
    remaining = [r for r in records if r.get("image_id", "") not in processed_ids]
    print(f"加载 {len(records)} 条记录, 剩余 {len(remaining)} 条待处理")

    if not remaining:
        print("全部完成!")
        return

    print(f"加载模型: {model_name}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    with results_path.open("a", encoding="utf-8") as f:
        for record in remaining:
            image_id = record.get("image_id", "")
            question = record.get("question", "")
            gt_bbox = record.get("bbox")
            image_path = data_dir / image_id

            print(f"\n[{len(all_results)+1}/{len(records)}] {image_id}")

            result_rec: dict[str, Any] = {
                "question_id": record.get("question_id"),
                "image_id": image_id,
                "question": question,
                "gt_bbox": gt_bbox,
            }

            if not image_path.exists():
                result_rec.update({
                    "status": "image_not_found",
                    "pred_bbox": None,
                    "iou": 0.0,
                    "correct": False,
                })
                all_results.append(result_rec)
                f.write(json.dumps(result_rec, ensure_ascii=False) + "\n")
                continue

            fig_data = fig_map.get(image_id, {})
            inference_path = image_path
            gt_for_iou = gt_bbox
            crop_region = None

            # ============================================================
            # 根据方法构建prompt和预处理图像
            # ============================================================

            if method == "prompt":
                with Image.open(image_path) as img:
                    img_w, img_h = img.size
                entities_text = build_entities_text(fig_data, img_w, img_h)
                full_prompt = f"{entities_text}\n\nQuestion: {question}"
                prompt_text = BBOX_PROMPT_WITH_ENTITIES.format(question=full_prompt)

            elif method == "blur":
                processed_path = processed_dir / f"blur_{image_id}"
                inference_path = process_blur(image_path, fig_data, processed_path)
                prompt_text = BBOX_PROMPT_TEMPLATE.format(question=question)

            elif method == "blackout":
                processed_path = processed_dir / f"blackout_{image_id}"
                inference_path = process_blackout(image_path, fig_data, processed_path)
                prompt_text = BBOX_PROMPT_TEMPLATE.format(question=question)

            elif method == "crop":
                processed_path = processed_dir / f"crop_{image_id}"
                inference_path, crop_region = process_crop(image_path, fig_data, processed_path)

                if gt_bbox and len(gt_bbox) == 4:
                    crop_w = crop_region[2] - crop_region[0]
                    crop_h = crop_region[3] - crop_region[1]
                    gt_for_iou = transform_gt_bbox(gt_bbox, crop_region, crop_w, crop_h)
                    result_rec["gt_bbox_cropped"] = gt_for_iou
                    result_rec["crop_region"] = list(crop_region)

                prompt_text = BBOX_PROMPT_TEMPLATE.format(question=question)

            else:
                prompt_text = BBOX_PROMPT_TEMPLATE.format(question=question)

            # ============================================================
            # 运行推理
            # ============================================================

            try:
                raw_output, img_w, img_h, orig_w, orig_h = run_inference(
                    model, processor, inference_path, prompt_text, MAX_TOKENS
                )
                result_rec["raw_output"] = raw_output
                result_rec["img_size_used"] = f"{img_w}x{img_h}"
                result_rec["orig_size"] = f"{orig_w}x{orig_h}"

                bboxes = extract_bboxes_from_output(raw_output)
                result_rec["num_pred_bboxes"] = len(bboxes)

                if not bboxes:
                    result_rec.update({
                        "status": "no_bbox_parsed",
                        "pred_bbox": None,
                        "iou": 0.0,
                        "correct": False,
                    })
                    print("  未解析到bbox")

                elif gt_for_iou and len(gt_for_iou) == 4:
                    # ============================================================
                    # 计算IoU（关键坐标转换逻辑）
                    #
                    # 对于crop方法:
                    #   gt_for_iou 已经在裁切坐标系中
                    #   需要缩放到推理图像尺寸 (可能被resize)
                    #   img_w, img_h = 推理图像尺寸 (裁切后可能resize)
                    #   crop_w, crop_h = 裁切图像原始尺寸
                    #
                    # 对于其他方法:
                    #   gt_for_iou = 原始GT坐标
                    #   需要缩放到推理图像尺寸 (可能被resize)
                    #   orig_w, orig_h = 原始图像尺寸
                    # ============================================================

                    if method == "crop":
                        crop_w = crop_region[2] - crop_region[0]
                        crop_h = crop_region[3] - crop_region[1]
                        gt_scaled = scale_bbox_to_resized(gt_for_iou, crop_w, crop_h, img_w, img_h)
                    else:
                        gt_scaled = scale_bbox_to_resized(gt_for_iou, orig_w, orig_h, img_w, img_h)

                    best_iou, best_box = 0.0, bboxes[0]
                    for pb in bboxes:
                        pb_pixel = denormalize_bbox(pb, img_w, img_h)
                        iou_val = compute_iou(gt_scaled, pb_pixel)
                        if iou_val > best_iou:
                            best_iou, best_box = iou_val, pb

                    result_rec.update({
                        "pred_bbox": best_box,
                        "gt_bbox_scaled": gt_scaled,
                        "iou": round(best_iou, 6),
                        "correct": best_iou >= IOU_THRESHOLD,
                        "status": "ok",
                    })
                    icon = "OK" if result_rec["correct"] else "MISS"
                    print(f"  -> IoU={best_iou:.4f} [{icon}]")

                else:
                    result_rec.update({
                        "pred_bbox": bboxes[0],
                        "iou": 0.0,
                        "correct": False,
                        "status": "no_gt_bbox",
                    })

            except Exception as e:
                result_rec.update({
                    "status": "inference_failed",
                    "raw_output": str(e),
                    "pred_bbox": None,
                    "iou": 0.0,
                    "correct": False,
                })
                print(f"  [ERR] {e}")

            # ============================================================
            # 生成可视化图像
            # ============================================================

            try:
                img_cv = cv2.imread(str(inference_path))
                if img_cv is not None:
                    if method == "crop" and crop_region:
                        crop_w = crop_region[2] - crop_region[0]
                        crop_h = crop_region[3] - crop_region[1]
                        gt_vis = gt_for_iou
                        pred_pixel = denormalize_bbox(result_rec.get("pred_bbox", [0, 0, 0, 0]), img_w, img_h)
                        vis_img = draw_bboxes_on_image(img_cv, gt_vis, pred_pixel, img_w, img_h, is_norm=False)
                    else:
                        vis_img = draw_bboxes_on_image(img_cv, gt_for_iou, result_rec.get("pred_bbox"), img_w, img_h)
                    cv2.imwrite(str(vis_dir / f"vis_{image_id}"), vis_img)
            except Exception as e:
                print(f"  [VIS ERR] {e}")

            all_results.append(result_rec)
            f.write(json.dumps(result_rec, ensure_ascii=False) + "\n")
            f.flush()

    print(f"\n结果已保存到: {results_path}")


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"用法: python {sys.argv[0]} <model_key> <method>")
        print(f"可用模型: {', '.join(MODELS.keys())}")
        print(f"可用方法: {', '.join(METHODS.keys())}")
        sys.exit(1)

    run_eval(sys.argv[1], sys.argv[2])
