from .PINNs import PINNAdapter
from .Transformers import GhostTransformerAdapter, TransformerAdapter
_ADAPTERS = {
    'PINN': PINNAdapter(), 'GhostTransformer': GhostTransformerAdapter(), 'Transformer': TransformerAdapter()
}

def build_adapter(model):
    return _ADAPTERS[model.input_type]
