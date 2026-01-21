import torch
from contextlib import contextmanager

_debug_enabled = False

@contextmanager
def debug_shapes():
    global _debug_enabled
    _debug_enabled = True
    yield
    _debug_enabled = False

def output_shape(func):
    def wrapper(self, *args, **kwargs):
        debug = _debug_enabled or getattr(self, 'debug', False)
        if not debug:
            return func(self, *args, **kwargs)

        def get_shape(x):
            if isinstance(x, torch.Tensor):
                return tuple(x.shape)
            elif isinstance(x, dict):
                return {k: get_shape(v) for k, v in x.items()}
            elif isinstance(x, (tuple, list)):
                return type(x)(get_shape(i) for i in x)
            return None

        def hook(module, inp, out):
            shape = get_shape(out)
            if shape is not None:
                print(f"{module.__class__.__name__.ljust(25)}\t{shape}")

        if args:
            print(f"\n{'Input'.ljust(25)}\t{get_shape(args[0])}")

        handles = [m.register_forward_hook(hook) for m in self.modules()]
        result = func(self, *args, **kwargs)
        for h in handles:
            h.remove()
        print()
        return result
    return wrapper