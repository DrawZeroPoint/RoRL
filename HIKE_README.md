### Environment setup

In a fresh virtual environment inherit from the system interpreter (on an Ubuntu 18.04 machine, this means 3.6),
create the venv named hike in `~/.venv/hike`, and install the following dependence:

#### Install PyTorch

GPU version with CUDA>=11.1

```shell
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

CPU version

```shell
pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Install PyTorch geometric

GPU version with CUDA>=11.1:

```shell
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
```

CPU version:

```shell
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html
```

#### Other Python packages may need

```shell
pip install rosbag
```

## How to use

The training script is [train_motion_graph](scripts/train_motion_graph.py), before using this, you may need to modify 
the data relevant setting as defined in `train_dataset_info` and `test_dataset_info`.

