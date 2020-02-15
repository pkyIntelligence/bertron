# bertron

bertron is a project to create an end to end neural network which can analyze an image and describe it in a natural human voice. Currently captioning is complete, please see the [CaptioningDemo](https://github.com/pkyIntelligence/bertron/blob/master/CaptioningDemo.ipynb)

## Installation

### Requirements
- An NVIDIA GPU
- Linux (Tested on Ubuntu 18.04)
- Cuda Drivers
- gcc & g++ ≥ 4.9
- Python
- PyTorch, torchvision, and cudatoolkit
- matplotlib
- requests
- validators
- cython
- OpenCV
- pycocotools
- boto3
- detectron2
- apex

### Steps (Conda highly recommended)

Please install Cuda Drivers appropriate for your GPU setup: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

Install gcc if you don't have it (gcc --version):
```
sudo apt install gcc
```

Create a conda environmnet:
```
conda create -n bertron python
```

Enter your new environment:
```
conda activate bertron
```

Install pytorch, torchvision, and cudatoolkit
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

Install the rest available through conda:
```
conda install matplotlib requests boto3
```

Install opencv
```
conda install -c conda-forge opencv
```

Install the rest available through pip:
```
pip install validators cython
```

Install pycocotools:
```
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Clone this repo recursively along with the submodules if you haven't already (you can ignore the failed commit):
```
git clone --recurse-submodules https://github.com/pkyIntelligence/bertron.git
```

Install this version of detectron2
```
cd bertron/detectron2
pip install -e .
cd ../..
```

Install apex
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```

Download pretrained model weights
```
cd bertron
wget -O e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212014&authkey=AAHgqN3Y-LXcBvU"
wget -O coco_g4_lr1e-6_batch64_scst.tar.gz "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212027&authkey=ACM1UXlFxgfWyt0"
```

Move weights and clean up
```
mv e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl model_weights/detectron
tar -xf coco_g4_lr1e-6_batch64_scst.tar.gz
mv coco_g4_lr1e-6_batch64_scst/model.19.bin model_weights/bert
rm -rf coco_g4_lr1e-6_batch64_scst*
```

### Done!!!
