from copy import deepcopy

import torch

from models import models
from edge_server import *
from utils.utils import Adding_Trigger

device = None
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Server(object):
    def __init__(self, conf, eval_dataset):
        self.conf = conf
        self.global_model = models.get_model(self.conf["model_name"])
        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)
        self.edge_servers = None

    def set_edge_servers(self, edge_servers):
        self.edge_servers = edge_servers

    def set_global_model(self, global_parameters):
        self.global_model.load_state_dict(global_parameters, strict=True)