import torch

def build_optimizer(model, args):
    name = args.optimizer.lower()
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f'Unknown optimizer: {args.optimizer}')
