import torch
from contextlib import contextmanager

_debug_enabled = False
_shape_log = None

def _fmt(name, shape, dtype=None):
    line = f"{name.ljust(40)}\t{str(shape).ljust(30)}"
    if dtype is not None:
        line += f"\t{dtype}"
    return line

@contextmanager
def debug_shapes(print_shapes=True):
    global _debug_enabled, _shape_log
    _debug_enabled = True
    _shape_log = []
    yield _shape_log
    _debug_enabled = False
    if print_shapes:
        for entry in _shape_log:
            print(_fmt(*entry))
        print()
    _shape_log = None

def output_shape(func):
    def wrapper(self, *args, **kwargs):
        debug = _debug_enabled or getattr(self, 'debug', False)
        if not debug:
            return func(self, *args, **kwargs)

        name_map = {id(m): n or m.__class__.__name__ for n, m in self.named_modules()}

        def get_shape(x):
            if isinstance(x, torch.Tensor):
                return tuple(x.shape)
            elif isinstance(x, dict):
                return {k: get_shape(v) for k, v in x.items()}
            elif isinstance(x, (tuple, list)):
                return type(x)(get_shape(i) for i in x)
            return None

        def get_dtype(x):
            if isinstance(x, torch.Tensor):
                return str(x.dtype).replace('torch.', '')
            elif isinstance(x, (tuple, list)):
                for i in x:
                    d = get_dtype(i)
                    if d is not None:
                        return d
            elif isinstance(x, dict):
                for v in x.values():
                    d = get_dtype(v)
                    if d is not None:
                        return d
            return None

        def hook(module, inp, out):
            shape = get_shape(out)
            if shape is not None:
                path = name_map.get(id(module), module.__class__.__name__)
                dtype = get_dtype(out)
                entry = (path, shape, dtype)
                if _shape_log is not None:
                    _shape_log.append(entry)
                else:
                    print(_fmt(*entry))

        if args:
            input_shape = get_shape(args[0])
            input_dtype = get_dtype(args[0]) if isinstance(args[0], torch.Tensor) else None
            entry = ("Input", input_shape, input_dtype)
            if _shape_log is not None:
                _shape_log.append(entry)
            else:
                print(f"\n{_fmt(*entry)}")

        handles = [m.register_forward_hook(hook) for m in self.modules()]
        result = func(self, *args, **kwargs)
        for h in handles:
            h.remove()
        if _shape_log is None:
            print()
        return result
    return wrapper
