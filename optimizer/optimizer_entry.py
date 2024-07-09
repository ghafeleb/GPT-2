
from torch.optim import AdamW, SGD

type2optimizer = {
    'sgd': SGD,
    'adam': AdamW,
}

optimizer_choice = [key for key in type2optimizer.keys()]


def select_optimizer(args, pretrain=False):
    if pretrain:
        return type2optimizer['adam']
    optimizer = type2optimizer[args.optimizer]
    return optimizer