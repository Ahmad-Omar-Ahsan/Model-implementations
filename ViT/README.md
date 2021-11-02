## Vision Transformer

This is the unofficial TensorFlow implementation of ViT. To modify the architecture change the attributes in `config.py`. This model shows how attention mechanism can be used to capture channel and spatial information of images to classify them paving the way for future models that utilizes patch based models

## Config file attributes

| Attributes | Default |
| --- | --- |
| image size | 32, 32 |
| patch size | 4, 4 |
| batch size | 64 |
| epoch | 50 |
| model save path | ViT/ |
| number of layers | 8 |
| number of classes | 10 |
| dimension of model | 192 |
| number of heads | 3 |
| mlp dimension | 256 |
| dropout | 0 |

## Model architecture
![alt text](./featured.png)

To train the model

```
python train.py
```