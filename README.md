# PAMol

## Dependency

```bash
conda create -n Model python=3.8
conda activate Model
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install keras-nightly==2.5.0.dev2021032900 -i https://pypi.org/simple/
pip install tensorflow-gpu==2.5.0

python
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2';
import tensorflow as tf
version = tf.__version__
print("tf version:", version) # 输出tensorflow版本, gpu可否使用（True/False）
# 判断tensorflow是否是CUDA版本，返回【ture】则说明OK True
print(tf.test.is_built_with_cuda())
from tensorflow.python.client import device_lib
# 判断目前可用设备，返回【CPU、GPU明细】则说明OK
print(device_lib.list_local_devices()) 
print(tf.config.list_physical_devices('CPU')) # 列出CPU设备，返回【CPU】则说明OK
print(tf.config.list_physical_devices('GPU')) # 列出GPU设备 ，返回【GPU】则说明OK
nvidia-smi
exit();

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install rdkit
pip3 install scikit-learn
pip install tqdm
pip install pandas==1.4.2
pip install tape_proteins
pip install wandb==0.16.5

# HGNN
conda create -n HGNN python=3.8
conda activate HGNN
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install torch_sparse-0.6.13-cp38-cp38-linux_x86_64.whl
pip install torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl
pip install -U -i https://pypi.tuna.tsinghua.edu.cn/simple torch_geometric==2.3.0
pip install rdkit
pip install pandas==1.4.2
# 蛋白超图
conda install -c conda-forge biopython
sudo apt-get install dssp
```

## Data

We trained/tested PAMol using the same data sets as [Pocket2Mol](https://github.com/pengxingang/Pocket2Mol) model.

1. Download the dataset archive `crossdocked_pocket10.tar.gz` and the split file `split_by_name.pt` from [this link](https://drive.google.com/drive/folders/1CzwxmTpjbrt83z_wBzcQncq84OVDPurM).
2. Extract the TAR archive using the command: `tar -xzvf crossdocked_pocket10.tar.gz`.
3. Without altering the original division of the dataset, filter out data that cannot be constructed into a hypergraph. If the machine performance is good, you can try replacing the protein pocket hypergraph with a protein hypergraph.

## Training

1. Obtain the latent vector features of molecules via `encode.py`

2. Obtain protein sequence features via `proteins_seq_encode.py` and `prepare_data_pair.py`

3. Obtain protein pocket hypergraph features via `ProteinDataset.py`

4. Obtain fused protein latent vector features via `crossfusion.py`

5. Modify the paths of the aforementioned features in `run.py`

6. Run `run.py`

## Sampling

Execute `runner.test()` in `run.py`

<!-- # The README.md will be further improved after subsequent organization. -->
