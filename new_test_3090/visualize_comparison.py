#!/usr/bin/env python3
"""生成HTML对比报告 - 支持多模型"""
import json
import sys
import base64
import io
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from config import BASE_DIR, DATA_DIR, METHODS, MODELS

METHOD_COLORS = {
    "baseline": (255, 0, 0),
    "prompt": (0, 0, 255),
    "blur": (255, 165, 0),
    "blackout": (128, 0, 128),
    "crop": (0, 128, 0),
}


def denormalize_bbox(bbox: list[float], w: int, h: int) -> list[float]:
    return [bbox[0] * w / 1000, bbox[1] * h / 1000, bbox[2] * w / 1000, bbox[3] * h / 1000]


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    results = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    results[rec.get("image_id", "")] = rec
    return results


def draw_comparison_on_image(
    image: Image.Image,
    gt_bbox: list[float] | None,
    predictions: dict[str, dict[str, Any]],
) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    if gt_bbox and len(gt_bbox) == 4:
        x1, y1, x2, y2 = gt_bbox
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        draw.text((x1, y1 - 15), "GT", fill=(0, 255, 0))

    for method, pred_data in predictions.items():
        pred_bbox_1000 = pred_data.get("pred_bbox")
        if pred_bbox_1000 and len(pred_bbox_1000) == 4:
            if method == "crop":
                crop_region = pred_data.get("crop_region")
                img_size_used = pred_data.get("img_size_used", "")
                if crop_region and img_size_used:
                    parts = img_size_used.split("x")
                    resize_w, resize_h = int(parts[0]), int(parts[1])
                    pred_in_crop = denormalize_bbox(pred_bbox_1000, resize_w, resize_h)
                    pred_pixel = [
                        pred_in_crop[0] + crop_region[0],
                        pred_in_crop[1] + crop_region[1],
                        pred_in_crop[2] + crop_region[0],
                        pred_in_crop[3] + crop_region[1],
                    ]
                else:
                    pred_pixel = denormalize_bbox(pred_bbox_1000, w, h)
            else:
                pred_pixel = denormalize_bbox(pred_bbox_1000, w, h)

            x1, y1, x2, y2 = pred_pixel
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            color = METHOD_COLORS.get(method, (128, 128, 128))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y2 + 5), f"{method}: {pred_data.get('iou', 0):.2f}", fill=color)

    return img


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


def create_report(model_key: str):
    model_cfg = MODELS[model_key]
    results_dir = Path(BASE_DIR) / model_cfg["output_dir"]
    data_dir = Path(DATA_DIR)
    output_path = results_dir / f"comparison_report_{model_key}.html"

    all_results = {}
    for method, method_cfg in METHODS.items():
        all_results[method] = load_results(results_dir / method_cfg["results_file"])

    method_stats = {}
    for method, results in all_results.items():
        total = len(results)
        correct = sum(1 for r in results.values() if r.get("correct"))
        iou_vals = [r.get("iou", 0) for r in results.values()]
        method_stats[method] = {
            "total": total,
            "correct": correct,
            "acc": correct / total if total > 0 else 0,
            "avg_iou": sum(iou_vals) / len(iou_vals) if iou_vals else 0,
        }

    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>VLM定位方法对比报告</title>
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 1600px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        h1 { text-align: center; color: #333; }
        h2 { color: #555; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        .summary { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .comparison-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .comparison-table th, .comparison-table td { padding: 12px; text-align: center; border: 1px solid #ddd; }
        .comparison-table th { background: #2196F3; color: white; }
        .comparison-table tr:nth-child(even) { background: #f9f9f9; }
        .best { font-weight: bold; color: #4CAF50; }
        .worst { color: #f44336; }
        .sample { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .sample-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
        .sample-title { font-size: 1.2em; font-weight: bold; }
        .badge { padding: 5px 15px; border-radius: 20px; font-weight: bold; }
        .badge-all-correct { background: #4CAF50; color: white; }
        .badge-partial { background: #FF9800; color: white; }
        .badge-all-wrong { background: #f44336; color: white; }
        .image-container { text-align: center; margin: 15px 0; }
        .image-container img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }
        .method-results { display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin-top: 15px; }
        .method-box { background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #ddd; }
        .method-box.baseline { border-left-color: #f44336; }
        .method-box.prompt { border-left-color: #2196F3; }
        .method-box.blur { border-left-color: #FF9800; }
        .method-box.blackout { border-left-color: #800080; }
        .method-box.crop { border-left-color: #008000; }
        .method-name { font-weight: bold; margin-bottom: 5px; }
        .method-iou { font-family: monospace; }
        .legend { text-align: center; margin: 10px 0; color: #666; }
        .legend span { margin: 0 15px; }
        .chart-container { background: white; padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .bar-chart { display: flex; align-items: flex-end; height: 200px; gap: 40px; justify-content: center; }
        .bar-group { display: flex; flex-direction: column; align-items: center; }
        .bar { width: 60px; border-radius: 5px 5px 0 0; transition: height 0.3s; }
        .bar.baseline { background: #f44336; }
        .bar.prompt { background: #2196F3; }
        .bar.blur { background: #FF9800; }
        .bar.blackout { background: #800080; }
        .bar.crop { background: #008000; }
        .bar-label { margin-top: 10px; font-size: 0.9em; color: #666; }
        .bar-value { margin-bottom: 5px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>VLM定位方法对比报告</h1>
    <p style="text-align: center; color: #666;">数据集: DVGBench MCP_test (50张图) | 模型: """ + model_cfg["name"] + """</p>

    <div class="summary">
        <h2>方法对比总结</h2>
        <table class="comparison-table">
            <tr><th>方法</th><th>正确数</th><th>总样本</th><th>Acc@0.5</th><th>平均IoU</th><th>排名</th></tr>
"""

    sorted_methods = sorted(method_stats.items(), key=lambda x: x[1]["acc"], reverse=True)
    for rank, (method, stats) in enumerate(sorted_methods, 1):
        acc_class = "best" if rank == 1 else ("worst" if rank == len(sorted_methods) else "")
        html += f'<tr><td><strong>{method}</strong></td><td>{stats["correct"]}</td><td>{stats["total"]}</td><td class="{acc_class}">{stats["acc"]:.2%}</td><td class="{acc_class}">{stats["avg_iou"]:.4f}</td><td>{rank}</td></tr>\n'

    html += """</table></div>
    <div class="chart-container">
        <h2>准确率对比</h2>
        <div class="bar-chart">
"""

    max_acc = max(s["acc"] for s in method_stats.values()) if method_stats else 1
    for method, stats in method_stats.items():
        bar_height = int(stats["acc"] / max_acc * 180) if max_acc > 0 else 0
        html += f'<div class="bar-group"><div class="bar-value">{stats["acc"]:.1%}</div><div class="bar {method}" style="height: {bar_height}px;"></div><div class="bar-label">{method}</div></div>\n'

    html += """</div></div>
    <div class="legend">
        <span style="color:#4CAF50;font-weight:bold;">■ GT框</span>
        <span style="color:#f44336;font-weight:bold;">■ Baseline</span>
        <span style="color:#2196F3;font-weight:bold;">■ Prompt</span>
        <span style="color:#FF9800;font-weight:bold;">■ Blur</span>
        <span style="color:#800080;font-weight:bold;">■ Blackout</span>
        <span style="color:#008000;font-weight:bold;">■ Crop</span>
    </div>
    <h2>逐样本对比</h2>
"""

    records = []
    with (data_dir / "50_MCP_test.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    for i, record in enumerate(records):
        image_id = record.get("image_id", "")
        question = record.get("question", "")
        gt_bbox = record.get("bbox")
        image_path = data_dir / image_id
        if not image_path.exists():
            continue

        predictions = {}
        correct_count = 0
        for method, results in all_results.items():
            if image_id in results:
                predictions[method] = results[image_id]
                if results[image_id].get("correct"):
                    correct_count += 1

        image = Image.open(image_path)
        img_with_boxes = draw_comparison_on_image(image, gt_bbox, predictions)
        img_base64 = image_to_base64(img_with_boxes)

        badge_class = "badge-all-correct" if correct_count == len(all_results) else ("badge-all-wrong" if correct_count == 0 else "badge-partial")
        badge_text = "全部正确" if correct_count == len(all_results) else ("全部错误" if correct_count == 0 else f"{correct_count}/{len(all_results)} 正确")

        html += f"""
    <div class="sample">
        <div class="sample-header">
            <span class="sample-title">Sample {i+1}: {image_id}</span>
            <span class="badge {badge_class}">{badge_text}</span>
        </div>
        <p><strong>Question:</strong> {question}</p>
        <div class="image-container"><img src="data:image/jpeg;base64,{img_base64}" alt="Sample {i+1}"></div>
        <div class="method-results">
"""
        for method, pred in predictions.items():
            iou = pred.get("iou", 0)
            correct = pred.get("correct", False)
            html += f'<div class="method-box {method}"><div class="method-name">{method} {"✓" if correct else "✗"}</div><div class="method-iou">IoU: {iou:.4f}</div></div>\n'

        html += """</div></div>"""

    html += "</body></html>"
    output_path.write_text(html, encoding="utf-8")
    print(f"HTML报告已保存: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"用法: python {sys.argv[0]} <model_key>")
        print(f"可用模型: {', '.join(MODELS.keys())}")
        sys.exit(1)
    create_report(sys.argv[1])
