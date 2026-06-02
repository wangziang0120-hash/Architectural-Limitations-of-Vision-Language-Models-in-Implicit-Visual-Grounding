# Data

This directory stores the evaluation dataset.

## Data Source

The evaluation is built upon **DVGBench**, a UAV visual grounding benchmark. Please download the original dataset from the DVGBench repository.

## Directory Structure

Place the data as follows:

```
data/
└── MCP_test/
    ├── 50_MCP_test.jsonl          # 50 test samples (questions + GT bboxes)
    ├── Prompt_test.jsonl          # Entity-level coordinate annotations
    ├── 0000000_00098_d_0000001.jpg
    ├── 0000000_01013_d_0000003.jpg
    └── ... (50 UAV images in total)
```

## JSONL Format

### 50_MCP_test.jsonl

One record per line:

```json
{
    "question_id": 0,
    "image_id": "0000000_00098_d_0000001.jpg",
    "question": "There are two white planes whose fans are larger than those of the other white planes",
    "bbox": [215.2, 66.2, 487.3, 183.9]
}
```

### Prompt_test.jsonl

One record per line, containing entity-level coordinate annotations:

```json
{
    "image": "0000000_00098_d_0000001.jpg",
    "entities": [
        {"id": "entity_1", "description": "...", "center": [0.5, 0.3]},
        ...
    ]
}
```

## Custom Data Path

To use a different data directory, set the environment variable:

```bash
export DATA_DIR=/path/to/your/MCP_test
```
