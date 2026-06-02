#!/usr/bin/env python3
"""跨模型/方法的可视化对比脚本"""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw

# 路径配置 - 通过环境变量或相对路径
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(Path(__file__).resolve().parent / "Results")))
IMAGES_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent / "data" / "MCP_test")))

MODELS = [
    "results_8b_inst",
    "results_4b_inst",
    "results_2b_inst",
    "results_8b_think"
]

METHODS = {
    "baseline": {"file": "baseline_results.jsonl", "color": (255, 0, 0), "label": "Baseline"},
    "prompt":   {"file": "prompt_results.jsonl",   "color": (0, 100, 255), "label": "Prompt"},
    "blur":     {"file": "blur_results.jsonl",     "color": (255, 165, 0), "label": "Blur"},
    "blackout": {"file": "blackout_results.jsonl", "color": (180, 0, 255), "label": "Blackout"},
    "crop":     {"file": "crop_results.jsonl",     "color": (0, 200, 0), "label": "Crop"}
}

GT_COLOR = (255, 255, 255)
GT_OUTLINE = (0, 0, 0)

def load_jsonl(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            data[record["image_id"]] = record
    return data

def draw_box_with_label(draw, bbox, color, label, is_gt=False):
    x1, y1, x2, y2 = bbox
    
    if is_gt:
        for offset in [-1, 0, 1]:
            draw.rectangle([x1+offset, y1+offset, x2+offset, y2+offset], outline=GT_OUTLINE, width=3)
        draw.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=2)
        text_y = max(y1 - 20, 5)
        text_bbox = draw.textbbox((x1, text_y), "GT")
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=GT_OUTLINE)
        draw.text((x1, text_y), "GT", fill=GT_COLOR)
    else:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_y = min(y2 + 5, draw.im.size[1] - 20)
        text_bbox = draw.textbbox((x1, text_y), label)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=(0, 0, 0))
        draw.text((x1, text_y), label, fill=color)

def denormalize_bbox(bbox, w, h):
    return [bbox[0] * w / 1000, bbox[1] * h / 1000, bbox[2] * w / 1000, bbox[3] * h / 1000]

def visualize_model(model_name):
    model_dir = RESULTS_DIR / model_name
    output_dir = model_dir / "vis_comparison"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nProcessing {model_name}...")
    
    method_data = {}
    for method_key, method_info in METHODS.items():
        jsonl_path = model_dir / method_info["file"]
        if jsonl_path.exists():
            method_data[method_key] = load_jsonl(jsonl_path)
            print(f"  Loaded {method_key}: {len(method_data[method_key])} records")
    
    all_image_ids = set()
    for data in method_data.values():
        all_image_ids.update(data.keys())
    
    print(f"  Total unique images: {len(all_image_ids)}")
    
    for idx, image_id in enumerate(sorted(all_image_ids), 1):
        img_path = IMAGES_DIR / image_id
        if not img_path.exists():
            print(f"  Warning: {img_path} not found, skipping")
            continue
        
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        for method_key in ["baseline", "prompt", "blur", "blackout", "crop"]:
            if method_key in method_data and image_id in method_data[method_key]:
                gt_bbox = method_data[method_key][image_id]["gt_bbox"]
                draw_box_with_label(draw, gt_bbox, GT_COLOR, "GT", is_gt=True)
                break
        
        for method_key, method_info in METHODS.items():
            if method_key not in method_data or image_id not in method_data[method_key]:
                continue
            
            record = method_data[method_key][image_id]
            pred_bbox = record.get("pred_bbox")
            
            if pred_bbox is None:
                continue
            
            w, h = img.size
            
            if method_key == "crop":
                img_size_used = record.get("img_size_used", "")
                if img_size_used:
                    parts = img_size_used.split("x")
                    resize_w, resize_h = int(parts[0]), int(parts[1])
                    pred_in_crop = denormalize_bbox(pred_bbox, resize_w, resize_h)
                    crop_region = record.get("crop_region", [0, 0, 0, 0])
                    pred_pixel = [
                        pred_in_crop[0] + crop_region[0],
                        pred_in_crop[1] + crop_region[1],
                        pred_in_crop[2] + crop_region[0],
                        pred_in_crop[3] + crop_region[1]
                    ]
                else:
                    pred_pixel = denormalize_bbox(pred_bbox, w, h)
            else:
                pred_pixel = denormalize_bbox(pred_bbox, w, h)
            
            draw_box_with_label(draw, pred_pixel, method_info["color"], method_info["label"], is_gt=False)
        
        output_path = output_dir / image_id
        img.save(output_path, quality=95)
        
        if idx % 10 == 0:
            print(f"  Processed {idx}/{len(all_image_ids)} images")
    
    print(f"  ✓ Completed {model_name}: {len(all_image_ids)} images saved to {output_dir}")

def main():
    print("=" * 60)
    print("Visualizing all methods vs GT for each model")
    print("=" * 60)
    
    for model_name in MODELS:
        if (RESULTS_DIR / model_name).exists():
            visualize_model(model_name)
        else:
            print(f"\nSkipping {model_name}: directory not found")
    
    print("\n" + "=" * 60)
    print("All visualizations complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
