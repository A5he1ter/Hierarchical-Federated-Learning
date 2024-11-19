import argparse
import copy
import json
import random
from copy import deepcopy

import torch
import numpy as np

import datasets
from client import Client
from models.aggregation import agg_average
from server import Server
from edge_server import EdgeServer
from utils.utils import test_accuracy

import matplotlib.pyplot as plt

device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hierarchical Federated Learning")
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    with open(args.conf) as f:
        conf = json.load(f)

    # 准备数据集
    train_datasets, eval_datasets = datasets.get_dataset("./data", conf['dataset'])

    # 设置服务器，边缘服务器，客户端以及标记恶意客户端
    edge_servers = []
    clients = []
    client_groups = []
    num_client_groups = conf['num_edge_servers']
    num_malicious_clients = conf['num_malicious_models']
    num_clients = conf['num_models']

    print("num of edge servers:", num_client_groups)
    print("num of clients:", conf["num_models"])
    print("num of global epochs:", conf["global_epochs"])
    print("device:", device)
    print("dataset:", conf["dataset"])
    print("attack type:", conf["attack_type"])
    print("detect type:", conf["detect_type"])

    server = Server(conf, eval_datasets)

    for c in range(num_clients):
        clients.append(Client(conf, server.global_model, train_datasets, eval_datasets, False, c))

    print("all clients:", clients)

    if conf["attack_type"] != "none":
        malicious_clients = random.sample(clients, num_malicious_clients)
        for c in malicious_clients:
            c.is_malicious = True
        print("malicious clients:", malicious_clients)

    # 对所有客户端进行分组，将其分配至边缘服务器下，并将边缘服务器分配至云服务器下
    group_size = len(clients) // num_client_groups
    for i in range(num_client_groups):
        client_groups.append(clients[i * group_size:(i + 1) * group_size])

    for i in range(num_client_groups):
        edge_servers.append(EdgeServer(conf, i, server.global_model, client_groups[i], eval_datasets))
        edge_servers[i].know_num_malicious()

    server.set_edge_servers(edge_servers)

    # 用于收集指标的容器
    global_acc_list = []
    global_loss_list = []
    global_asr_list = []
    defense_acc_list = []
    malicious_precision_list = []
    malicious_recall_list = []

    # 设置优化器与损失函数
    optim = torch.optim.Adam(server.global_model.parameters(), lr=conf["lr"])
    loss_func = torch.nn.functional.cross_entropy

    print("---------------------------start training---------------------------\n\n")
    global_epochs = conf["global_epochs"]
    attack_type = conf["attack_type"]
    for e in range(global_epochs):
        # 设置子全局模型参数收集器
        edge_server_params = {}
        edge_agg_params = {}

        for i in range(num_client_groups):
            edge_servers[i].set_global_model(server.global_model)
            global_model = deepcopy(edge_servers[i].global_model)
            edge_client_params = {}
            for c in edge_servers[i].clients:
                local_params = None
                if not c.is_malicious:
                    local_params = deepcopy(c.local_train(global_model, loss_func, optim,))
                else:
                    if attack_type == "SA":
                        local_params = deepcopy(c.scaling_attack_train(global_model, loss_func, optim, num_clients,
                                                                       num_malicious_clients))
                    elif attack_type == "LFA":
                        local_params = deepcopy(c.label_flipping_attack_train(global_model, loss_func, optim))
                    elif attack_type == "RLFA":
                        local_params = deepcopy(c.random_label_flipping_attack_train(global_model, loss_func, optim))
                    elif attack_type == "GA":
                        local_params = deepcopy(c.gaussian_attack_train(global_model, loss_func, optim))

                local_params_flatten = torch.cat([param.data.clone().view(-1) for key, param in local_params.items()],
                                                dim=0)
                edge_client_params[c] = deepcopy(local_params_flatten.cpu())
            edge_server_params[edge_servers[i]] = deepcopy(edge_client_params)

        for edge_server in edge_server_params:
            edge_params = edge_server_params[edge_server]
            edge_agg_params[edge_server] = deepcopy(agg_average(edge_params))

        agg_params = agg_average(edge_agg_params)

        start_idx = 0
        global_parameters = deepcopy(server.global_model.state_dict())
        for key, var in global_parameters.items():
            param = agg_params[start_idx:start_idx + len(var.data.view(-1))].reshape(var.data.shape)
            start_idx = start_idx + len(var.data.view(-1))
            global_parameters[key] = deepcopy(param)

        server.set_global_model(global_parameters)

        global_loss, global_acc = test_accuracy(server)
        print('[Round: %d] >> Global Model Test accuracy: %f' % (e, global_acc))
        print('[Round: %d] >> Global Model Test loss: %f' % (e, global_loss))
        global_acc_list.append(global_acc)
        global_loss_list.append(global_loss)

    fig, axes = plt.subplots(1, 2, figsize=(18, 12))

    # highlight_indices = np.array([0, 24, 49, 74, 99])
    # epoch_list = np.array(range(len(global_acc_list)))
    epoch_list = range(len(global_acc_list))
    # global_acc_list = np.array(global_acc_list)
    # global_loss_list = np.array(global_loss_list)
    # global_asr_list = np.array(global_asr_list)
    # defense_acc_list = np.array(defense_acc_list)
    # malicious_precision_list = np.array(malicious_precision_list)
    # malicious_recall_list = np.array(malicious_recall_list)

    # global model accuracy fig
    # highlight_y = global_acc_list[highlight_indices]
    # highlight_x = epoch_list[highlight_indices]
    axes[0].plot(epoch_list, global_acc_list, "r", label='Accuracy')
    # axes[0, 0].scatter(highlight_x, highlight_y, color="red", label='Global Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Global Model Accuracy')
    axes[0].legend()

    # global model loss fig
    # highlight_y = global_loss_list[highlight_indices]
    # highlight_x = epoch_list[highlight_indices]
    axes[1].plot(epoch_list, global_loss_list, "b", label='Loss')
    # axes[0, 1].scatter(highlight_x, highlight_y, color="blue", label='Global Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Global Model Loss')
    axes[1].legend()

    # if conf["attack_type"] == "scaling attack" or conf["attack_type"] == "a little enough attack" or conf[
    #     "attack_type"] == "mixed attack":
    #     highlight_y = global_asr_list[highlight_indices]
    #     highlight_x = epoch_list[highlight_indices]
    #     axes[0, 2].plot(epoch_list, global_asr_list, "green", label='ASR')
    #     axes[0, 2].scatter(highlight_x, highlight_y, color="green", label='ASR')
    #     axes[0, 2].set_xlabel('Epoch')
    #     axes[0, 2].set_ylabel('ASR')
    #     axes[0, 2].set_title('ASR')
    #     axes[0, 2].legend()
    #
    # if conf["attack_type"] != "no attack":
    #     highlight_y = defense_acc_list[highlight_indices]
    #     highlight_x = epoch_list[highlight_indices]
    #     axes[1, 0].plot(epoch_list, defense_acc_list, "yellow", label='Defense Accuracy')
    #     axes[1, 0].scatter(highlight_x, highlight_y, color="yellow", label='Defense Accuracy')
    #     axes[1, 0].set_xlabel('Epoch')
    #     axes[1, 0].set_ylabel('Defense Accuracy')
    #     axes[1, 0].set_title('Defense Accuracy')
    #     axes[1, 0].legend()
    #
    #     highlight_y = malicious_precision_list[highlight_indices]
    #     highlight_x = epoch_list[highlight_indices]
    #     axes[1, 1].plot(epoch_list, malicious_precision_list, "black", label='Malicious Precision')
    #     axes[1, 1].scatter(highlight_x, highlight_y, color="black", label='Malicious Precision')
    #     axes[1, 1].set_xlabel('Epoch')
    #     axes[1, 1].set_ylabel('Malicious Precision')
    #     axes[1, 1].set_title('Malicious Precision')
    #     axes[1, 1].legend()
    #
    #     highlight_y = malicious_recall_list[highlight_indices]
    #     highlight_x = epoch_list[highlight_indices]
    #     axes[1, 2].plot(epoch_list, malicious_recall_list, "purple", label='Malicious Recall')
    #     axes[1, 2].scatter(highlight_x, highlight_y, color="purple", label='Malicious Recall')
    #     axes[1, 2].set_xlabel('Epoch')
    #     axes[1, 2].set_ylabel('Malicious Recall')
    #     axes[1, 2].set_title('Malicious Recall')
    #     axes[1, 2].legend()

    plt.tight_layout(pad=0.1)

    plt.savefig('./fig/' + conf["type"] + " " + conf["attack_type"] + " " + conf["detect_type"] + ".png")

    plt.show()