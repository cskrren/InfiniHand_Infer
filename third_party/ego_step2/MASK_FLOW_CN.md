# Step 2：手部检测与 SAM3 Mask 流程

本目录完成独立的 **手部检测 -> SAM3 分割 -> 原分辨率 mask 与可视化**。它不运行
SLAM、MANO 或完整 HaWoR pipeline。

## 1. 固定流程

```text
输入视频/有序图片
  -> 按 output_fps 解码或读取原图
  -> YOLO 批量 predict 左右手框
  -> 每帧、每侧只保留面积最大的框
  -> 帧与框同步缩放到 SAM3 最长边 1008
  -> point seed + joint tracking + single-anchor propagation
  -> 得到 tight mask / bbox
  -> mask 恢复至原始解码帧坐标
  -> 写 PNG、JSONL、overlay.mp4、contact_sheet.jpg
```

关键生产默认值来自 `configs/default.yaml`：

- detector confidence：`0.2`
- detector prepass batch：`16`
- handedness：`0=left, 1=right`
- SAM3 seed threshold：`0.75`
- seed window：`5 s`
- tracking / init：`joint / point`
- prompt geometry：`box_below`
- propagation：`single_anchor`
- mask archive：`npz`
- ego-hand bbox gate：开启
- SAM3 输入最长边：`1008`
- 输出 FPS：`10`

完整参数映射见 `docs/PRODUCTION_DEFAULTS.md`；代码来源映射见
`docs/SOURCE_MAP.md`。

## 2. 环境

仓库已经包含 detector、SAM3 checkpoint 和固定 SAM3 源码：

```text
weights/detector.pt
checkpoints/facebook_sam3/sam3.pt
checkpoints/facebook_sam3/config.json
third_party/sam3/
```

首次使用时：

```bash
cd /mnt/petrelfs/share_data/renkerui/Ego3R/preprocess/step2
bash scripts/setup_env.sh
```

如果已经有兼容的 Python 3.12、PyTorch/CUDA 环境，可改用现有 venv 或
Apptainer；不要重复下载仓库内已有权重。

## 3. 推荐：先跑 120 帧 Slurm smoke

登录节点只负责提交任务，不直接搬数据或跑推理：

```bash
cd /mnt/petrelfs/share_data/renkerui/Ego3R/preprocess/step2
mkdir -p logs outputs

MAX_FRAMES=120 \
PROJECT_DIR=$PWD \
INPUT_TYPE=video \
INPUT_PATH=/path/to/input.mp4 \
OUTPUT_DIR=$PWD/outputs/input_smoke \
VENV=$PWD/.venv \
sbatch -D "$PWD" \
  -o "$PWD/logs/%x_%j.out" \
  -e "$PWD/logs/%x_%j.err" \
  slurm/run_hand_sam3_mask.sbatch
```

图片目录输入时改为：

```bash
INPUT_TYPE=frames INPUT_PATH=/path/to/ordered_frames
```

图片按文件名排序，支持 `.jpg/.jpeg/.png/.bmp/.webp`，且必须同分辨率。

## 4. Smoke 验收后跑完整输入

去掉 `MAX_FRAMES` 或将其留空：

```bash
PROJECT_DIR=$PWD \
INPUT_TYPE=video \
INPUT_PATH=/path/to/input.mp4 \
OUTPUT_DIR=$PWD/outputs/input_full \
VENV=$PWD/.venv \
sbatch -D "$PWD" \
  -o "$PWD/logs/%x_%j.out" \
  -e "$PWD/logs/%x_%j.err" \
  slurm/run_hand_sam3_mask.sbatch
```

输出目录已存在时任务会直接失败；确实要覆盖时显式传 `OVERWRITE=1`。

## 5. 输出与下游约定

```text
OUTPUT_DIR/
  run_config.json
  frame_manifest.json
  summary.json
  frames/                         # 原始解码帧坐标
  detection/bboxes_2d.pth         # 原始坐标的 YOLO 框
  detections.jsonl
  sam3_input/                     # 仅供 SAM3 的 resize 中间件
  sam3_raw/                       # SAM3 原生结果与诊断
  masks/left/000000.png
  masks/right/000000.png
  masks.jsonl
  mask_summary.json
  overlay.mp4
  contact_sheet.jpg
  visualization_summary.json
  wandb/
```

Mask 约定：

- 单通道二值 PNG，前景 `255`，背景 `0`；
- 每帧固定写 left/right 两张图；缺失手为全零 mask；
- `masks.jsonl.status` 为 `ok` 或 `empty_mask`；
- mask、bbox、overlay 与 `frames/` 坐标完全一致；
- 左手使用蓝色，右手使用红色。

## 6. 必做 QA

Smoke 至少检查：

1. `summary.json` 与 `mask_summary.json` 没有失败记录；
2. `frames/`、左右手 `masks/` 数量一致；
3. mask 分辨率与对应原始帧一致；
4. `overlay.mp4` 中左右手颜色、bbox 和 mask 对齐；
5. `contact_sheet.jpg` 没有整体坐标偏移；
6. 空 mask 是全零图，不是缺文件；
7. `run_config.json` 记录的 checkpoint、config 和 commit 正确。

已有 mask 只需重画可视化，不重新推理：

```bash
bash scripts/visualize.sh /path/to/OUTPUT_DIR
```

代码与 CLI 的模型无关检查：

```bash
bash scripts/check.sh
```

## 7. 失败语义

流程采取 fail-loud：无有效 SAM3 seed、权重/源码不匹配、图片尺寸不一致或输出
目录冲突都会直接停止，不做静默 fallback。默认只有一条 left lane 和一条 right
lane，适合 egocentric wearer hands；多人或多只同侧手需要额外设计 instance ID。
