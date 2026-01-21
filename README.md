# output-shape 

[![PyPI version](https://badge.fury.io/py/output-shape.svg)](https://badge.fury.io/py/output-shape)

A very lightweight and minimalistic output shape examiner of layers and models.

** Currently working for PyTorch models only. Keras / Jax soon! **

# Installation
```bash
pip install output-shape
```

# Usage

Decorate the forward method with `@output_shape`, then use the context manager:

```python
import torch
from output_shape import output_shape, debug_shapes

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ...

    @output_shape
    def forward(self, x):
        ...

model = Model()
with debug_shapes():
    model(torch.randn(2, 1, 128, 128))
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
