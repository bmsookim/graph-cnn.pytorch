<p align="center"><img width="40%" src="./imgs/pytorch.png"></p>

# graph-cnn.pytorch
Pytorch implementation of Graph Convolution Networks.
This project is made by Bumsoo Kim.

Korea University, Master-Ph.D intergrated Course.

## Graph Convolutional Networks
Many important real-world datasets come in the form of graphs or networks: social networks, knowledge graphs, protein-interaction networks, the World Wide Web, etc. In this repository, we introduce a basic tutorial for generalizing neural netowrks to work on arbitrarily structured graphs, along with a proposal of a new structure that outperforms the state-of-the-art performance of current Graph Convolutional Networks([Attention GCN]()).

Currently, most graph neural network models have a somewhat universal architecture in common. They are referred as Graph Convoutional Networks(GCNs) since filter parameters are typically shared over all locations in the graph.

<p align="center"><img width="80%" src="./imgs/gcn_web.png"></p>

For these models, the goal is to learn a function of signals/features on a graph G=(V, E), which takes as 

**Input**
- N x D feature matrix (N : Number of nodes, D : number of input features)
- representative description of the graph structure in matrix form; typically in the form of adjacency matrix A

**Output**
- N x F feature matrix (N : Number of nodes, F : number of output features)

Graph-level outputs can be modeled by introducing some form of pooling operation.

Every neural network layer can then be written as a non-linear function

![H equation](http://latex.codecogs.com/gif.latex?H%5E%7B%28l&plus;1%29%7D%3Df%28H%5El%2C%20A%29)

with ![H(0)](http://latex.codecogs.com/gif.latex?H%5E%7B%280%29%7D%3DX) and ![H(L)](http://latex.codecogs.com/gif.latex?H%5E%7B%28L%29%7D%3DZ), where ***L*** is the number of layers. The specific models then differ only in how function ***f*** is chosen and parameterized.

In this repo, the layer-wise propagation is consisted as

<p align="center"><img src="http://latex.codecogs.com/gif.latex?f%28H%28l%29%2CA%29%3D%5Csigma%28AH%28l%29W%28l%29%29"></p>

As the activation function is a non-linear ReLU (Rectified Linear Unit), this becomes

<p align="center"><img src="http://latex.codecogs.com/gif.latex?f%28H%28l%29%2CA%29%3DReLU%28AH%28l%29W%28l%29%29"></p>

**Implementation detail 1 :**

Multiplication with ***A*** means that, for every node, we sum up all the feature vectors of all neighboring nodes but not the node itself. To address this, we add the identity matrix to ***A***.

**Implementation detail 2 :**

***A*** is typically not normalized and therfore the multiplication and therefore the multiplication with ***A*** will completely change the scale of the feature vectors. Normalizing A such that all rows sum to one, i.e. ![row sum](http://latex.codecogs.com/gif.latex?D%5E%7B-1%7DA).


**Final Implementation :**

<p align="center"><img src="http://latex.codecogs.com/gif.latex?f%28H%5E%7B%28l%29%7D%2CA%29%3D%5Chat%7BD%7D%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D%5Chat%7BA%7D%5Chat%7BD%7D%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D"></p>

## Requirements
See the [installation instruction](INSTALL.md) for a step-by-step installation guide.
See the [server instruction](SERVER.md) for server settup.
- Install [cuda-8.0](https://developer.nvidia.com/cuda-downloads)
- Install [cudnn v5.1](https://developer.nvidia.com/cudnn)
- Download [Pytorch for python-2.7](https://pytorch.org) and clone the repository.
```bash
pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp27-none-linux_x86_64.whl
pip install torchvision
git clone https://github.com/meliketoy/resnet-fine-tuning.pytorch
```

## Basic Setups
After you have cloned this repository into your file system, open [config.py](./config.py),
And edit the lines below to your data directory.
```bash
data_base = [:dir to your original dataset]
aug_base =  [:dir to your actually trained dataset]
```

For training, your data file system should be in the following hierarchy.
Organizing codes for your data into the given requirements will be provided [here](https://github.com/meliketoy/image-preprocessing)

```bash
[:data file name]

    |-train
        |-[:class 0]
        |-[:class 1]
        |-[:class 2]
        ...
        |-[:class n]
    |-val
        |-[:class 0]
        |-[:class 1]
        |-[:class 2]
        ...
        |-[:class n]
```

## How to run
After you have cloned the repository, you can train the dataset by running the script below.

You can set the dimension of the additional layer in [config.py](./config.py)

The resetClassifier option will automatically detect the number of classes in your data folder and reset the last classifier layer to the according number.

```bash
# zero-base training
python main.py --lr [:lr] --depth [:depth] --resetClassifier

# fine-tuning
python main.py --finetune --lr [:lr] --depth [:depth]

# fine-tuning with additional linear layers
python main.py --finetune --addlayer --lr [:lr] --depth [:depth]
```

## Train various networks

I have added fine-tuning & transfer learning script for alexnet, VGG(11, 13, 16, 19),
ResNet(18, 34, 50, 101, 152).

Please modify the [scripts](./train) and run the line below.

```bash

$ ./train/[:network].sh 

# For example, if you want to pretrain alexnet, just run
$ ./train/alexnet.sh

```

## Test (Inference) various networks

For testing out your fine-tuned model on alexnet, VGG(11, 13, 16, 19), ResNet(18, 34, 50, 101, 152),

First, set your data directory as test_dir in [config.py](./config.py).

Please modify the [scripts](./test) and run the line below.

```bash

$ ./test/[:network].sh

```
For example, if you have trained ResNet with 50 layers, first modify the [resnet test script](./test/resnet.sh)

```bash
$ vi ./test/resnet.sh

python main.py \
    --net_type resnet \
    --depth 50
    --testOnly

$ ./test/resnet.sh

```

The code above will automatically download weights from the given depth data, and train your dataset with a very small learning rate.

## Feature extraction
For various training mechanisms, extracted feature vectors are needed.

This repository will provide you not only feature extraction from pre-trained networks,

but also extractions from a model that was trained by yourself.

Just set the test directory in the [config.py](config.py) and run the code below.

```bash
python extract_features.py
```

This will automatically create pickles in a newly created 'vector' directory,

which will contain dictionary pickles which contains the below.

Currently, the 'score' will only cover 0~1 scores for binary classes.

Confidence scores for multiple class will be updated afterwards.

```bash
pickle_file [name : image base name]
    |- 'file_name' : file name of the test image
    |- 'features'  : extracted feature vector
    |- 'score'     : Score for binary classification
```

Enjoy :-)
