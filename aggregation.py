import numpy as np
import torch

from utils.utils import euclidean_clients


def average(params):
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

def multi_krum(params, num_adv):
    grads = []
    clients_in_comm = []

    for c in params:
        clients_in_comm.append(c)
        local_parameters = params[c]
        grads = local_parameters[None, :] if len(grads) == 0 else torch.cat(
            (grads, local_parameters[None, :]), 0
        )

    euclidean_matrix = euclidean_clients(grads)

    scores = []

    for list in euclidean_matrix:
        clients_dis = sorted(list)
        clients_dis1 = clients_dis[1: len(params) - num_adv]
        score = np.sum(np.array(clients_dis1))
        scores.append(score)
    clients_scores = dict(zip(clients_in_comm, scores))
    clients_scores = sorted(clients_scores.items(), key=lambda d: d[1], reverse=False)

    benign_client = clients_scores[: len(params) - num_adv]
    benign_client = [idx for idx, val in benign_client]
    malicious_client = clients_scores[len(params) - num_adv: ]
    malicious_client = [idx for idx, val in malicious_client]

    benign_client_params = {}
    for c in params:
        if c in benign_client:
            benign_client_params[c] = params[c]

    global_parameters = average(benign_client_params)

    return global_parameters, malicious_client