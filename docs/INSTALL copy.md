## Human Mesh Recovery benchmark
output format(pkl):
results[_id]['pose'] = pred_pose
results[_id]['trans'] = pred_trans
results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
results[_id]['verts'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
results[_id]['frame_ids'] = frame_id

<!-- git submodule deinit -f third_party/WHAM
git rm -f third_party/WHAM
rm -rf .git/modules/third_party/WHAM -->

use_cuda121

git clone https://github.com/SijiaLii/hmr_benchmark.git --recursive

(conda env remove -n hmr)
conda create -n hmr_benchmark ppython=3.11.9 -y
conda activate hmr_benchmark

# PromptHMR
cd third_party/PromptHMR
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade setuptools pip
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
conda install -c conda-forge suitesparse -y
conda install -c conda-forge pytorch3d
pip install -r requirements.txt (最好一个一个安装)
pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 --no-deps
gdown --folder -O ./data/ https://drive.google.com/drive/folders/1IXyhVqL25ofI-tYqyUZCqF-h4V20795H?usp=sharing

pip install data/wheels/detectron2-0.8-cp311-cp311-linux_x86_64.whl
pip install data/wheels/droid_backends_intr-0.3-cp311-cp311-linux_x86_64.whl
pip install data/wheels/lietorch-0.3-cp311-cp311-linux_x86_64.whl
pip install data/wheels/sam2-1.5-cp311-cp311-linux_x86_64.whl

pip install tyro

# WHAM
git submodule add https://github.com/SijiaLii/WHAM.git third_party/WHAM
git submodule update --init --recursive third_party/WHAM

cd third_party/WHAM
# Install PyTorch libraries
<!-- pip uninstall torch torchvision torchaudio -y -->
<!-- pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 -->
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install PyTorch3D (optional) for visualization
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
cd third-party
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
# Install ViTPose
cd ../..
pip install -v -e third-party/ViTPose
bash ./fetch_demo_data.sh


# TokenHMR
git submodule add https://github.com/saidwivedi/TokenHMR.git third_party/TokenHMR

cd third_party/TokenHMR
Ensure CUDA_HOME is set to the CUDA version installed with PyTorch.
<!-- pip install git+https://github.com/facebookresearch/detectron2 -->
<!-- (mkdir third-party
cd third-party
git clone https://github.com/SijiaLii/NMR.git
cd NMR
pip install -e .
cd ../..) -->
pip install "git+https://github.com/SijiaLii/PHALP.git#egg=phalp[all]"
bash ./fetch_demo_data.sh
PHALP needs SMPL neutral model for running video demo. Copy the model to appropriate location.
cp data/body_models/smpl/SMPL_NEUTRAL.pkl $HOME/.cache/phalp/3D/models/smpl/

<!-- (cv2.destroyAllWindows()
sudo apt update
sudo apt install libgtk2.0-dev pkg-config
pip uninstall opencv-python opencv-python-headless
pip install opencv-python) -->

# CameraHMR
git submodule add https://github.com/SijiaLii/CameraHMR.git third_party/CameraHMR
cd third_party/CameraHMR
bash scripts/fetch_demo_data.sh

# 4D Human
git submodule add https://github.com/shubham-goel/4D-Humans.git third_party/4D_Human
cd third_party/4D_Human/
mkdir data
comment # cv2.destroyAllWindows()


# PromptHMR
git submodule add https://github.com/yufu-wang/PromptHMR.git third_party/PromptHMR

pip install --upgrade setuptools pip
pip install -U xformers --index-url https://download.pytorch.org/whl/cu128 --no-deps

sudo apt update
sudo apt install -y cmake build-essential pkg-config libprotobuf-dev protobuf-compiler
sudo apt install -y libsentencepiece-dev
pip install sentencepiece

mkdir third_party
cd third_party/
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .

git clone --recursive https://github.com/princeton-vl/lietorch.git
cd lietorch
pip install --no-build-isolation .

<!-- git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git
cd DROID-SLAM/
pip install thirdparty/pytorch_scatter
python setup.py bdist_wheel(change name in setup.py)
pip install dist/droid_backends_intr-*.whl -->
pip install data/wheels/droid_backends_intr-0.4-cp310-cp310-linux_x86_64.whl

bash scripts/fetch_smplx.sh
bash scripts/fetch_data.sh

<!-- strat -->
pip install --upgrade setuptools pip
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
conda install -c conda-forge suitesparse -y
pip install -U xformers --index-url https://download.pytorch.org/whl/cu128 --no-deps
gdown --folder -O ./data/ https://drive.google.com/drive/folders/1IXyhVqL25ofI-tYqyUZCqF-h4V20795H?usp=sharing


pip install data/wheels/detectron2-0.8-cp311-cp311-linux_x86_64.whl
pip install data/wheels/droid_backends_intr-0.3-cp311-cp311-linux_x86_64.whl
pip install data/wheels/sam2-1.5-cp311-cp311-linux_x86_64.whl

mkdir third_party
cd third_party/
git clone --recursive https://github.com/princeton-vl/lietorch.git
cd lietorch
pip install --no-build-isolation .

cd ../..
bash scripts/fetch_demo_data.sh
pip install -r requirements.txt

WHAM, TokenHMR, HMR2.0, PromptHMR, CameraHMR, NLF