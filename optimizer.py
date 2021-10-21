# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from torch import optim as optim


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """

    parameters = list(model.named_parameters())
    for n, v in parameters:
        if ("score" not in n) and v.requires_grad:
            print(n, "weight_para")
    for n, v in parameters:
        if ("score" in n) and v.requires_grad:
            print(n, "score_para")
    weight_params = [v for n, v in parameters if ("score" not in n) and v.requires_grad]
    score_params = [v for n, v in parameters if ("score" in n) and v.requires_grad]

    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    weight_params = set_weight_decay([(n, v) for n, v in parameters if ("score" not in n) and v.requires_grad], skip, skip_keywords)
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer, score_optimizer = None, None
    if opt_lower == 'sgd':
        if config.train_weights_at_the_same_time:
            optimizer = optim.SGD(weight_params, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
            score_optimizer = optim.Adam(score_params, lr=12e-3, weight_decay=0)
        else:
            optimizer = optim.SGD(weight_params, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                                  lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        if config.train_weights_at_the_same_time:
            optimizer = optim.AdamW(weight_params, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                    lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
            score_optimizer = optim.Adam(score_params, lr=12e-3, weight_decay=0)
        else:
            optimizer = optim.AdamW(weight_params, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                    lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    return optimizer, score_optimizer, weight_params


def set_weight_decay(weight_params, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in weight_params:
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
            # if len(param.shape) == 1:
            #     print("1")
            # elif name.endswith(".bias"):
            #     print("2")
            # elif (name in skip_list):
            #     print("3")
            # elif check_keywords_in_name(name, skip_keywords):
            #     print("4")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
