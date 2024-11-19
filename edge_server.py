from email.policy import strict

import torch
import copy

device = None
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class EdgeServer:
    """
    初始化边缘服务器，接收从云服务器传来的初始全局模型。

    :param server_id: 边缘服务器的ID
    :param clients: 客户端列表
    :param initial_model: 从云服务器获取的初始全局模型
    """
    def __init__(self, conf, server_id, global_model, clients, eval_dataset):
        self.conf = conf
        self.server_id = server_id
        self.global_model = global_model
        self.clients = clients
        self.local_params_list = {}
        self.num_malicious = 0
        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

    def set_global_model(self, global_model):
        self.global_model.load_state_dict(global_model.state_dict(), strict=True)

    def know_num_malicious(self):
        for c in self.clients:
            if c.is_malicious:
                self.num_malicious += 1