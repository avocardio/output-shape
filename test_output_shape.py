import torch
import torch.nn as nn
from output_shape import output_shape, debug_shapes


class SimpleCNN(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    @output_shape
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class DictOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(64, 128)
        self.mha = nn.MultiheadAttention(128, 4, batch_first=True)
        self.fc = nn.Linear(128, 10)

    @output_shape
    def forward(self, x):
        x = self.embed(x)
        attn_out, _ = self.mha(x, x, x)
        return {"logits": self.fc(attn_out), "hidden": attn_out}


class SequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))

    @output_shape
    def forward(self, x):
        return self.features(x)


def test_self_debug():
    model = SimpleCNN(debug=True)
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_context_manager():
    model = SimpleCNN()
    with debug_shapes():
        out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_silent_without_debug():
    model = SimpleCNN()
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_dict_output():
    model = DictOutputModel()
    with debug_shapes():
        out = model(torch.randn(2, 10, 64))
    assert out["logits"].shape == (2, 10, 10)
    assert out["hidden"].shape == (2, 10, 128)


def test_sequential():
    model = SequentialModel()
    with debug_shapes():
        out = model(torch.randn(4, 32))
    assert out.shape == (4, 10)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
