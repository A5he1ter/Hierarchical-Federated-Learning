import torch

def agg_average(params):
    # 把需要聚合的模型参数，放到一个tensor中
    grads = []
    for item in params:
        parameters = params[item]
        grads = parameters[None, :] if len(grads) == 0 else torch.cat(
            (grads, parameters[None, :]), 0
        )

    # 用torch平均参数
    avg_params = torch.mean(grads, dim=0)

    return avg_params