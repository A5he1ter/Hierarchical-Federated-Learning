import random
import argparse, json
from cProfile import label

from sympy.core.parameters import global_parameters

import models
import torch
import copy
import numpy as np
from utils import utils
from utils.utils import Adding_Trigger

device = None
if torch.backends.mps.is_available():
	device = torch.device('mps')
elif torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

class Client(object):
	def __init__(self, conf, model, train_dataset, eval_dataset, is_malicious, id = -1):
		self.conf = conf
		self.local_model = model
		self.client_id = id
		self.train_dataset = train_dataset
		self.eval_dataset = eval_dataset
		self.is_malicious = is_malicious
		
		all_range = list(range(len(self.train_dataset)))
		data_len = int(len(self.train_dataset) / self.conf['num_models'])
		train_indices = all_range[id * data_len: (id + 1) * data_len]

		self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"], 
									sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))
		self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)
									
	def local_train(self, model, loss_func, optim):
		self.local_model.load_state_dict(model.state_dict(), strict=True)

		self.local_model.train()
		for e in range(self.conf["local_epochs"]):
			
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				data, target = data.to(device), target.to(device)

				optim.zero_grad()
				output = self.local_model(data)
				loss = loss_func(output, target.long())
				loss.backward()
			
				optim.step()
			# print("Client", c,"Epoch %d done." % e)
		# print("Client", c, "done.")
		# self.eval_local_model()

		return self.local_model.state_dict()

	def scaling_attack_train(self, model, loss_func, optim, num_clients, num_malicious_clients):
		self.local_model.load_state_dict(model.state_dict(), strict=True)

		self.local_model.train()

		for e in range(self.conf["local_epochs"]):

			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch

				for example_id in range(data.shape[0] // 2):
					data[example_id] = Adding_Trigger(data[example_id])
					target[example_id] = 0

				data = data.to(device)
				target = target.to(device)

				optim.zero_grad()
				output = self.local_model(data)
				loss = loss_func(output, target)
				loss.backward()
				optim.step()
		local_params = self.local_model.state_dict()

		clip_rate = (num_clients / num_malicious_clients) / 2
		for name, data in self.local_model.state_dict().items():
			global_value = model.state_dict()[name].to(device)
			new_value = global_value + (data - global_value) * clip_rate
			local_params[name].copy_(new_value)
			# diff[name] = (data - local_params[name])

		# print("Client", c, "done. --scaling attack--")

		return local_params

	def label_flipping_attack_train(self, model, loss_func, optim):
		nclass = np.max(np.array(self.train_dataset.targets)) + 1

		self.local_model.load_state_dict(model.state_dict(), strict=True)

		self.local_model.train()

		for e in range(self.conf["local_epochs"]):
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch

				target = nclass - 1 - target

				data, target = data.to(device), target.to(device)

				optim.zero_grad()
				output = self.local_model(data)
				loss = loss_func(output, target.long())
				loss.backward()
				optim.step()

		# print("Client", c, "done. --LF attack--")

		return self.local_model.state_dict()

	def random_label_flipping_attack_train(self, model, loss_func, optim):
		nclass = np.max(np.array(self.train_dataset.targets)) + 1

		self.local_model.load_state_dict(model.state_dict(), strict=True)

		self.local_model.train()

		for e in range(self.conf["local_epochs"]):
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch

				target = (random.randint(0, nclass - 1) + target) % nclass

				data, target = data.to(device), target.to(device)

				optim.zero_grad()
				output = self.local_model(data)
				loss = loss_func(output, target.long())
				loss.backward()
				optim.step()

			# print("Client", c, "done. --LF attack--")

			return self.local_model.state_dict()


	def gaussian_attack_train(self, model, loss_func, optim):
		self.local_model.load_state_dict(model.state_dict(), strict=True)

		self.local_model.train()

		for e in range(self.conf["local_epochs"]):
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch

				data, target = data.to(device), target.to(device)

				optim.zero_grad()
				output = self.local_model(data)
				loss = loss_func(output, target.long())
				loss.backward()
				optim.step()

		local_params = self.local_model.state_dict()

		for name, data in local_params.items():
			noise = torch.randn(data.shape).to(device)
			a = torch.mean(data.float())
			b = torch.std(data.float())
			data_GS = a + noise * b
			local_params[name].copy_(data_GS)

		# self.eval_local_model()
		return local_params

	def backdoor_attack_train(self, g_model, c, alpha=0.2):

		# self.local_model.load_state_dict(model, strict=True)

		model = self.local_model
		model.load_state_dict(g_model, strict=True)
		model.train()

		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'])

		# self.local_model.train()
		model.train()

		for e in range(self.conf["local_epochs"]):

			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				for example_id in range(data.shape[0]):
					data[example_id] = Adding_Trigger(data[example_id])
					target[example_id] = 0
				data = data.to(device)
				target = target.to(device)

				# output = self.local_model(data)
				output = model(data)
				loss = torch.nn.functional.cross_entropy(output, target.long())
				dist_loss_func = torch.nn.MSELoss()

				if alpha > 0:
					dist_loss = 0
					for name, data in model.state_dict().items():
						dist_loss += dist_loss_func(data, g_model[name].to(device))

					loss += dist_loss * alpha

				loss.backward()
				optimizer.step()
				optimizer.zero_grad()

		# local_params = self.local_model.state_dict()
		local_params = model.state_dict()
		# print("Client", c, "done. --LIE attack--")
		return local_params

	def eval_local_model(self):
		self.local_model.eval()

		total_loss = 0.0
		correct = 0
		dataset_size = 0
		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch
			dataset_size += data.size()[0]

			data = data.to(device)
			target = target.to(device)

			# if torch.cuda.is_available():
			# 	data = data.cuda()
			# 	target = target.cuda()

			output = self.local_model(data)

			total_loss += torch.nn.functional.cross_entropy(output, target,
															reduction='sum').item()  # sum up batch loss
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

		acc = float(correct) / float(dataset_size)
		total_l = total_loss / dataset_size

		print("client", self.client_id, "acc:", acc, "loss:", total_l)