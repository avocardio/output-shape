import torch

def output_shape(func):
    def wrapper(self, *args, **kwargs):
        def hook(module, input, output):
            if self.debug:
                name = module.__class__.__name__
                if isinstance(module, torch.nn.Sequential):
                    if hasattr(module, 'name') or hasattr(module, name):
                        name += f" ({module.name})"
                        print(f"{name.ljust(25)}\t{output.shape}")
                # if the output is from a Transformer
                elif isinstance(output, tuple):
                    if isinstance(module, torch.nn.MultiheadAttention):
                        attention_output_shape = output[0].shape if output[0] is not None else None
                        attention_weights_shape = output[1].shape if output[1] is not None else None
                        if attention_output_shape is not None and attention_weights_shape is not None:
                            print(f"{name.ljust(25)}\t{attention_output_shape} {{Attention Output}} {attention_weights_shape} {{Attention Weights}}")
                        elif attention_output_shape is not None:
                            print(f"{name.ljust(25)}\t{attention_output_shape}")
                        elif attention_weights_shape is not None:
                            print(f"{name.ljust(25)}\t{attention_weights_shape} {{Attention Weights}}")
                    else:
                        print(f"{name.ljust(25)}\tTuple Output: {[out.shape if out is not None else None for out in output]}")
                else:
                    print(f"{name.ljust(25)}\t{output.shape}")
        if self.debug and args:
            print("")
            print(f"{'Input'.ljust(25)}\t{args[0].shape}")
        module = self
        handle_list = []
        for name, module in module.named_modules():
            handle = module.register_forward_hook(hook)
            handle_list.append(handle)
        result = func(self, *args, **kwargs)
        if self.debug:
            print("")
        for handle in handle_list:
            handle.remove()
        return result
    return wrapper