# dnn-gating

```dnn-gating``` is a collective repository of [precision-gating](https://arxiv.org/abs/2002.07136) and [channel-gating](https://arxiv.org/abs/1805.12549) reimplemented in Pytorch.

## Precision Gating (PG)

### Requirments of PG

```
python 3.6.8
torch >= 1.3.0
numpy 1.16.4
matplotlib 3.1.0
```

### Usage

With this repo, you can:
  - Evaluate uniform quantization and [PACT](https://arxiv.org/abs/1805.06085).
  - Evaluate PG on ResNet CIFAR-10.
  - Apply PG to your own models and datasets.

##### Example
The following example trains ResNet-20 on CIFAR-10 with activations quantized to 3 bits, 2 MSBs out of which for prediction.

```sh
$ cd scripts
$ source train_pg_pact.sh
```

##### Specify the Flags

Make sure to tune the training parameters in to achieve a good model prediction accuracy.
```
  -w : bitwidth of weights (floating-point if set to 0)
  -a : bitwidth of activations (floating-point if set to 0)
  -pact : use parameterized clipping for activatons
  -pg : use PG
  -pb : prediction bitwidth (only valid if -pg is turned on, and the bitwidth of prediction must smaller than that of activations)
  -gtar : the gating target
  -sg : the penalty factor on the gating loss
  -spbp : use sparse back-prop
```

## Channel Gating (CG)

### Requirments of CG
```
python 2.7.12
torch 1.1.0
numpy 1.16.4
matplotlib 2.1.0
```
### Usage

With this repo, you can:
  - Evaluate CG on ResNet CIFAR-10 (both the original and modified post-activated ResNets).
  - The post-activated ResNet allows applying channel gating to all convolutional layers in a residual module.
  - Apply CG to your own models and datasets.

##### Example
The following examples use one fourth and half of input channels in the base path for the original and post-activated ResNets, respectively.

```sh
$ cd scripts
$ source train_cg.sh
$ source train_cg_postact.sh
```

##### Specify the Flags

The training parameters can be tuned to achieve different FLOP reduction and model accuracy.
```
  -lr : initial learning rate
  -wd: weigth decaying factor
  -pt: use 1/pt fraction of channels for prediction
  -gi: the intital value of gating thresholds
  -gtar: the target value of gating thresholds
  -spbp : use sparse back-prop
  -group: use group conv in the base path
  -cg : use CG
  -postact: use post-activated ResNet
```

## Apply PG/CG to Your Own Models & Datasets

The following steps allows you to apply PG/CG to your own models.
  1. Copy the model file to ```model/```.
  2. Import ```utils/pg_utils.py``` /```utils/cg_utils.py``` in the model file, replace convolutional layers followed by activation functions with the ```PGConv2d```/ ```CGConv2d```  module.
  3. Import ```model/your_model.py``` in the ```generate_model()``` function in ```pg-cifar10.py```/ ```cg-cifar10.py```.

If you prepare your own training scripts, remember to add the **gating loss** to the model prediction loss before doing back-prop.

##### Note
  - The way of exporting sparsity in the update phase we are using is only valid while training on a single GPU. This is because Pytorch modifies each model replica on a GPU instead of a global model if ```DataParallel``` is activated. For multi-GPU training, we suggest users turn off the sparsity printing during training, save the trained model, and print the sparsity only when testing.



### Citation
If you use CG or PG in your research, please cite our NeurIPS'19 and ICLR'20 papers.

**Channel Gating Neural Networks**
```

@incollection{NIPS2019_8464,
title = {Channel Gating Neural Networks},
author = {Hua, Weizhe and Zhou, Yuan and De Sa, Christopher M and Zhang, Zhiru and Suh, G. Edward},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {1886--1896},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/8464-channel-gating-neural-networks.pdf}
}
```
**Precision Gating: Improving Neural Network Efficiency with Dynamic Dual-Precision Activations**
```
@inproceedings{
Zhang2020Precision,
title={Precision Gating: Improving Neural Network Efficiency with Dynamic Dual-Precision Activations},
author={Yichi Zhang and Ritchie Zhao and Weizhe Hua and Nayun Xu and G. Edward Suh and Zhiru Zhang},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=SJgVU0EKwS}
}
```
