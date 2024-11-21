import argparse
import json
import random
import sys
import threading
import time
from copy import deepcopy

import datasets
from client import Client
from aggregation import average, multi_krum
from server import Server
from edge_server import EdgeServer
from utils.utils import *

from torch.utils.tensorboard import SummaryWriter

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

    log_dir = f"./log/{time.strftime('%Y-%m-%d/%H.%M.%S')}"
    writer = SummaryWriter(log_dir=log_dir)

    file = open(log_dir + '/readme.md', mode='a', encoding='utf-8')
    json_str = json.dumps(conf, indent=4)
    file.write("### 实验参数\n")
    for key, value in conf.items():
        file.write(f'    {key}: {value}\n')
    file.close()

    tb_port = 6007
    tb_host = "127.0.0.1"
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=([log_dir, tb_port, tb_host]),
        daemon=True
    ).start()
    time.sleep(3.0)

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

    if conf["attack_type"] != "":
        malicious_clients = random.sample(clients, num_malicious_clients)
        for c in malicious_clients:
            c.is_malicious = True

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
    lr = conf["lr"]
    optim = torch.optim.Adam(server.global_model.parameters(), lr=lr)
    loss_func = torch.nn.functional.cross_entropy

    print("---------------------------start training---------------------------\n\n")
    global_epochs = conf["global_epochs"]
    attack_type = conf["attack_type"]
    detect_type = conf["detect_type"]
    for e in range(global_epochs):
        # 设置子全局模型参数收集器
        edge_server_params = {}
        edge_agg_params = {}
        # all_client_params = {}
        # lie_list = []
        for i in range(num_client_groups):
            edge_servers[i].set_global_model(server.global_model)
            global_model = deepcopy(edge_servers[i].global_model)
            edge_client_params = {}
            all_client_params = {}
            for c in edge_servers[i].clients:
                local_params = {}
                if not c.is_malicious:
                    local_params = deepcopy(c.local_train(global_model, loss_func, optim))
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

            # if attack_type == "LIEA":
            #     lie_list.append(edge_client_params)
            #     edge_server_params[edge_servers[i]] = deepcopy(edge_client_params)
            # else:
            #     edge_server_params[edge_servers[i]] = deepcopy(edge_client_params)
            edge_server_params[edge_servers[i]] = deepcopy(edge_client_params)

        # if attack_type == "LIEA":
        #     for d in lie_list:
        #         all_client_params.update(d)
        #     edge_server_params = deepcopy(LIE_attack(server.global_model.state_dict(), lr, all_client_params, edge_server_params, loss_func, optim))

        if detect_type == "multi krum":
            for edge_server in edge_server_params:
                edge_params = edge_server_params[edge_server]
                edge_agg_params[edge_server], d_m_c = deepcopy(multi_krum(edge_params, edge_server.num_malicious))
        else:
            for edge_server in edge_server_params:
                edge_params = edge_server_params[edge_server]
                edge_agg_params[edge_server] = deepcopy(average(edge_params))

        agg_params = average(edge_agg_params)

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

        writer.add_scalar('scalar/Accuracy', global_acc, e)
        writer.add_scalar('scalar/Loss', global_loss, e)

        if attack_type == "SA" or attack_type == "LIEA":
            global_asr = test_attack_success_rate(server)
            print('[Round: %d] >> Global Model Test ASR: %f' % (e, global_asr))

            writer.add_scalar('scalar/ASR', global_asr, e)


    writer.close()
    sys.exit(0)