# Figures

This directory stores paper figures.

## Figures

- `Fig1.jpg` — Illustration of the five attention-guiding methods
- `Fig2.jpg` — Predicted and ground-truth bounding boxes comparison
- `Fig3.jpg` — Acc@0.5 and mean IoU of 2B/4B/8B Instruct models across five methods
- `Fig4.png` — Acc@0.5 and mean IoU comparison between 8B Instruct and Thinking

## Generating Visualisations

To generate comparison visualisations, run:

```bash
cd new_test_3090
python visualize_comparison.py 8b_inst
```

Or use the cross-model visualisation script:

```bash
python visualize_all_methods.py
```
