# ResNet Pytorch Implementation

This is the unofficial Pytorch implementation of ResNet. To change the architecture of your model add or modify the config file. This model incorporates skip connections to improve performance as the depth of the network increases. 

## Config file attributes

| Attributes | Default |
| --- | --- |
| image channels | 3 |
| number of classes | 10 |
| device | cuda |
| batch size | 64 |
| epoch | 50 |
| model save path | cifar_net.pth |
| learning rate | 3e-5 | 

## Model architecture
![alt text](./featured.png)


To run the script

```
python train.py
```