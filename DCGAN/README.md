# DCGAN
This is the unofficial implementation of PyTorch DCGAN

DCGAN, or Deep Convolutional GAN, is a generative adversarial network architecture. It uses a couple of guidelines, in particular:

- Replacing any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
- Using batchnorm in both the generator and the discriminator.
- Removing fully connected hidden layers for deeper architectures.
- Using ReLU activation in generator for all layers except for the output, which uses tanh.
- Using LeakyReLU activation in the discriminator for all layer.

## Config file attributes

| Attributes | Default |
| --- | --- |
| number of classes | 10 |
| device | cuda |
| batch size | 64 |
| epoch | 50 |
| generator learning rate | 0.0002 |
| discriminator learning rate | 0.0002 | 
| image size | 64, 64, 3 |
| latent dimension | 100 |

## Model architecture
![alt text](./featured.png)

## Training

```
python3 train.py
```

