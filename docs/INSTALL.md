## Human Mesh Recovery benchmark
git clone https://github.com/SijiaLii/hmr_benchmark.git --recursive

use_cuda128
(conda env remove -n hmr)
conda create -n hmr python=3.10 -y
conda activate hmr

<!-- git submodule deinit -f third_party/WHAM
git rm -f third_party/WHAM
rm -rf .git/modules/third_party/WHAM -->

# WHAM
git submodule add https://github.com/SijiaLii/WHAM.git third_party/WHAM
git submodule update --init --recursive third_party/WHAM

cd third_party/WHAM
# Install PyTorch libraries
<!-- pip uninstall torch torchvision torchaudio -y -->
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
<!-- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -->

# Install PyTorch3D (optional) for visualization
<!-- conda install -c fvcore -c iopath -c conda-forge fvcore iopath -->
pip install -U 'git+https://github.com/facebookresearch/fvcore'
<!-- pip install --upgrade "fvcore<0.1.6,>=0.1.5" -->
pip install -U 'git+https://github.com/facebookresearch/iopath'
<!-- pip install --upgrade "iopath<0.1.10,>=0.1.7" -->


cd third-party
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
<!-- pip install -e . -->
python -m pip install . 
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

# NLF
git submodule add https://github.com/SijiaLii/nlf.git third_party/NLF
pip:
tensorflow==2.15
tensorflow-hub
embreex
importlib_resources
more_itertools
tetgen
pymeshfix
git+https://github.com/isarandi/fleras.git
git+https://github.com/isarandi/cameralib.git
git+https://github.com/isarandi/boxlib.git
git+https://github.com/isarandi/poseviz.git
git+https://github.com/isarandi/smplfitter.git
git+https://github.com/isarandi/simplepyutils.git
git+https://github.com/isarandi/tf-parallel-map.git
git+https://github.com/isarandi/BareCat.git
git+https://github.com/SijiaLii/rlemasklib.git
git+https://github.com/isarandi/tensorflow-inputs.git
cd third_party/NLF/
wget -q -O models/nlf_l_multi.torchscript https://bit.ly/nlf_l_pt



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