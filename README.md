# LasVegas_Python


### Getting Started
[![python version](https://img.shields.io/badge/python-3.6-black)](https://www.python.org/)
[![pytorch version](https://img.shields.io/badge/PyTorch-1.4.0-red)](https://pytorch.org/)
[![opencv version](https://img.shields.io/badge/opencv-4.1.1-green)](https://opencv.org/)
[![numpy version](https://img.shields.io/badge/numpy-1.17.4-blue)](https://numpy.org/)
```
pip install -r requirements.txt
```
### 1. Train Model
```sh
python3 pretrain.py
python3 train_ml_player.py
```
### 2. Test Model
```sh
python3 test_game.py
```
### 3. Convert Model to ONNX
```sh
python3 torch_to_onnx.py
```