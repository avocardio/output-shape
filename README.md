# output-shape

[![PyPI version](https://badge.fury.io/py/output-shape.svg)](https://badge.fury.io/py/output-shape)

A very lightweight and minimalistic output shape debugger for PyTorch models. Shows module paths, output shapes, and dtypes for every layer during a forward pass.

# Installation
```bash
pip install output-shape
```

# Usage

Decorate the forward method with `@output_shape`, then use either option:

```python
import torch
from output_shape import output_shape, debug_shapes

class Model(torch.nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        ...

    @output_shape
    def forward(self, x):
        ...

# Option 1: Context manager (recommended)
model = Model()
with debug_shapes():
    model(torch.randn(2, 3, 32, 32))

# Option 2: Instance flag
model = Model(debug=True)
model(torch.randn(2, 3, 32, 32))
```

Example output for a tiny ViT:
```
Input                                   (2, 3, 32, 32)                float32
patch_embed                             (2, 128, 8, 8)                float32
blocks.0.norm1                          (2, 64, 128)                  float32
blocks.0.attn.qkv                       (2, 64, 384)                  float32
blocks.0.attn.proj                      (2, 64, 128)                  float32
blocks.0.attn                           (2, 64, 128)                  float32
blocks.0.norm2                          (2, 64, 128)                  float32
blocks.0.mlp.0                          (2, 64, 512)                  float32
blocks.0.mlp.1                          (2, 64, 512)                  float32
blocks.0.mlp.2                          (2, 64, 128)                  float32
blocks.0.mlp                            (2, 64, 128)                  float32
blocks.0                                (2, 64, 128)                  float32
blocks.1.norm1                          (2, 64, 128)                  float32
blocks.1.attn.qkv                       (2, 64, 384)                  float32
blocks.1.attn.proj                      (2, 64, 128)                  float32
blocks.1.attn                           (2, 64, 128)                  float32
blocks.1.norm2                          (2, 64, 128)                  float32
blocks.1.mlp.0                          (2, 64, 512)                  float32
blocks.1.mlp.1                          (2, 64, 512)                  float32
blocks.1.mlp.2                          (2, 64, 128)                  float32
blocks.1.mlp                            (2, 64, 128)                  float32
blocks.1                                (2, 64, 128)                  float32
blocks                                  (2, 64, 128)                  float32
norm                                    (2, 128)                      float32
head                                    (2, 10)                       float32
```

# Programmatic Access

Use `debug_shapes()` as a context manager to capture shapes as structured data:

```python
with debug_shapes(print_shapes=False) as shapes:
    model(torch.randn(2, 3, 32, 32))

# shapes = [("patch_embed", (2, 128, 8, 8), "float32"), ...]
```
