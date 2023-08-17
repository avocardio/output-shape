# output-shape

[![PyPI version](https://badge.fury.io/py/output-shape.svg)](https://badge.fury.io/py/output-shape)

A very lightweight and minimalistic output shape examiner of layers and models.

** Currently working for PyTorch models only. **

# Installation
```bash
pip install output-shape
```

# Usage

You need to decorate the forward method of your model with the decorator and add a debug flag to the init of your model.

```python
import torch
import output_shape

class Model(torch.nn.Module):
    def __init__(self, debug=False):
        self.debug = debug
        ...

    @output_shape
    def forward(self, x):
        ...

model = Model(debug=True)(torch.randn(2, 1, 128, 128))
```

```python
Input                           torch.Size([2, 1, 128, 128])
Conv2d                          torch.Size([2, 768, 8, 8])
PatchEmbed                      torch.Size([2, 64, 768])
LayerNorm                       torch.Size([2, 13, 768])
Linear                          torch.Size([2, 13, 2304])
Linear                          torch.Size([2, 13, 768])
Dropout                         torch.Size([2, 13, 768])
Attention                       torch.Size([2, 13, 768])
PreNorm                         torch.Size([2, 13, 768])
LayerNorm                       torch.Size([2, 13, 768])
Linear                          torch.Size([2, 13, 3072])
GELU                            torch.Size([2, 13, 3072])
Dropout                         torch.Size([2, 13, 3072])
Linear                          torch.Size([2, 13, 768])
Dropout                         torch.Size([2, 13, 768])
FeedForward                     torch.Size([2, 13, 768])
PreNorm                         torch.Size([2, 13, 768])
Transformer                     torch.Size([2, 13, 768])
LayerNorm                       torch.Size([2, 13, 768])
Linear                          torch.Size([2, 12, 512])
LayerNorm                       torch.Size([2, 8, 8, 512])
CyclicShift                     torch.Size([2, 8, 8, 512])
Linear                          torch.Size([2, 8, 8, 2016])
Linear                          torch.Size([2, 8, 8, 512])
CyclicShift                     torch.Size([2, 8, 8, 512])
WindowAttention                 torch.Size([2, 8, 8, 512])
PreNorm                         torch.Size([2, 8, 8, 512])
Residual                        torch.Size([2, 8, 8, 512])
LayerNorm                       torch.Size([2, 8, 8, 512])
Linear                          torch.Size([2, 8, 8, 2048])
GELU                            torch.Size([2, 8, 8, 2048])
Dropout                         torch.Size([2, 8, 8, 2048])
Linear                          torch.Size([2, 8, 8, 512])
Dropout                         torch.Size([2, 8, 8, 512])
FeedForward                     torch.Size([2, 8, 8, 512])
PreNorm                         torch.Size([2, 8, 8, 512])
Residual                        torch.Size([2, 8, 8, 512])
SwinBlock                       torch.Size([2, 8, 8, 512])
LayerNorm                       torch.Size([2, 64, 512])
Linear                          torch.Size([2, 64, 256])
```