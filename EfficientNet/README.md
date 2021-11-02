# EfficientNet implementation

This is the unofficial pytorch implementation of EfficientNet. To change the configuration, go to config.py and modify the attributes or add new attributes. This model uses compound scaling, inverted residual blocks from MobileNetV2 and stochastic depth for better performance. 


## Config file attributes

| Attributes | Default |
| --- | --- |
| number of classes | 10 |
| device | cuda |
| batch size | 64 |
| epoch | 50 |
| model save path | cifar_net.pth |

## Model architecture
![alt text](./featured.png)

To train the model

```
python3 train.py
```
Report: [Training for 50 epochs](https://wandb.ai/omar_11234/EfficientNet/reports/EfficientNet-PyTorch-implementation--VmlldzoxMTc0OTc0?accessToken=6sfrom392x616yedyhtufbxm3mvvsuhbcg3t9x5nwgy9v2j0bvxho0vb32povhmu)

## Checkpoint 
| Checkpoints | Accuracy |
| ----- | ---- |
| cifar_net.pth | 52.18% | 