from __future__ import annotations
MODEL_REGISTRY = {}

def register_model(cls):
    name = cls.__name__
    if name in MODEL_REGISTRY:
        raise ValueError(f'{name} is already registered.')
    MODEL_REGISTRY[name] = cls
    return cls

def build_model(args):
    model_cls = MODEL_REGISTRY[args.model_name]
    return model_cls.from_args(args)
