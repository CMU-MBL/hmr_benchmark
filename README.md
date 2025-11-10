# HMR Benchmark

A unified Human Mesh Recovery (HMR) benchmarking framework that integrates multiple state-of-the-art HMR models for easy comparison and evaluation.

## 📋 Table of Contents

- [Features](#features)
- [Supported Models](#supported-models)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Output Format](#output-format)
- [Dependencies](#dependencies)


## ✨ Features

- **Unified Interface**: Run different HMR models through a unified command-line interface
- **Multi-Model Support**: Integrates multiple state-of-the-art HMR models including WHAM, TokenHMR, CameraHMR, 4D Human, PromptHMR, and more (<span style="color:#FF5733"><b>WIP</b></span>)
- **Video Processing**: Supports human mesh recovery on video inputs
- **Standardized Output**: All models output in a unified format for easy comparison
- **Easy to Extend**: Modular design makes it easy to add new models

## 🤖 Supported Models

This project integrates the following HMR models:

1. **WHAM** - Whole-Body Human Activity and Motion
2. **TokenHMR** - Token-based Human Mesh Recovery
3. **CameraHMR** - Camera-aware Human Mesh Recovery
4. **4D Human (HMR2.0)** - 4D Human Mesh Recovery
5. **PromptHMR** - Prompt-based Human Mesh Recovery
6. **NLF** - Neural Light Field (<span style="color:#FF5733"><b>WIP</b></span>)

## 🚀 Installation

### Requirements

- Python 3.10+
- CUDA 12.8+ (recommended)
- Linux system

### Clone Repository

```bash
git clone https://github.com/SijiaLii/hmr_benchmark.git --recursive
cd hmr_benchmark
```

### Create Conda Environment

```bash
conda create -n hmr python=3.10 -y
conda activate hmr
```

### Install Dependencies

For detailed installation instructions, please refer to [docs/INSTALL.md](docs/INSTALL.md).

Basic dependency installation:

```bash
pip install -r requirements.txt
```

### Install Individual Models

Each model needs to be installed separately. Please follow the instructions in `docs/INSTALL.md`.

## 🎯 Quick Start

### Run Models Using Unified Interface

```bash
python run.py --model WHAM --video_pth videos/gymnasts.mp4 --output_pth output/WHAM
```

### Run Specific Models Directly

```bash
# Run WHAM
python scripts/run_WHAM.py --video videos/gymnasts.mp4 --output_pth output/WHAM

# Run TokenHMR
python scripts/run_TokenHMR.py --video videos/gymnasts.mp4 --output_pth output/TokenHMR

# Run CameraHMR
python scripts/run_CameraHMR.py --video videos/gymnasts.mp4 --output_folder output/CameraHMR

# Run 4D Human
python scripts/run_4dHuman.py --video videos/gymnasts.mp4 --output_pth output/4dHuman
```

## 📖 Usage

### Unified Interface (`run.py`)

```bash
python run.py \
    --model <MODEL_NAME> \
    --video_pth <VIDEO_PATH> \
    --output_pth <OUTPUT_DIR> \
    [--extra_args ...]
```

**Arguments:**
- `--model`: Model name (e.g., WHAM, TokenHMR, CameraHMR, etc.)
- `--video_pth`: Path to input video file
- `--output_pth`: Path to output directory
- `--extra_args`: Additional arguments passed to the specific model

### Individual Model Scripts

Each model has its own run script in the `scripts/` directory:

- `run_WHAM.py`: Run WHAM model
- `run_TokenHMR.py`: Run TokenHMR model
- `run_CameraHMR.py`: Run CameraHMR model
- `run_4dHuman.py`: Run 4D Human model

For specific arguments of each script, use `--help`:

```bash
python scripts/run_WHAM.py --help
```

## 📊 Output Format

All models output in a unified format, saved in `output/<MODEL_NAME>/<VIDEO_NAME>/` directory:

- `<model>_output.pkl`: Pickle file containing model predictions
  - `pose`: SMPL pose parameters (N, 72) - includes root and body joints
  - `trans`: Translation vector (N, 3)
  - `betas`: Shape parameters (10,)
  - `verts`: 3D vertex coordinates (N, V, 3)
  - `frame_ids`: Frame indices (N,)

- `output.mp4`: Visualization video (if supported by the model)

### Reading Output Results

```python
import joblib

# Load results
results = joblib.load('output/WHAM/gymnasts/wham_output.pkl')

# Access results for specific person
for person_id, data in results.items():
    pose = data['pose']      # (N, 72)
    trans = data['trans']    # (N, 3)
    betas = data['betas']    # (10,)
    verts = data['verts']    # (N, V, 3)
    frame_ids = data['frame_ids']  # (N,)
```

## 🔧 Dependencies

This project depends on multiple third-party libraries. Main dependencies include:

- PyTorch (recommended with CUDA 12.8)
- SMPL-X
- Detectron2
- OpenCV
- NumPy, SciPy
- Hydra (configuration management)
- Other model-specific dependencies

For a complete list of dependencies, please refer to `requirements.txt` and `docs/INSTALL.md`.

## 📝 License

This project integrates multiple third-party models, each of which may have its own license. Please check the license files of each model before use.

## 🙏 Acknowledgments

This project integrates the following excellent open-source projects:

- [WHAM](https://github.com/SijiaLii/WHAM)
- [TokenHMR](https://github.com/saidwivedi/TokenHMR)
- [CameraHMR](https://github.com/SijiaLii/CameraHMR)
- [4D-Humans](https://github.com/shubham-goel/4D-Humans)
- [PromptHMR](https://github.com/yufu-wang/PromptHMR)

Thanks to all contributors for their hard work!

