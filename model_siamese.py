import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import numpy as np 
import logging
from torch.utils.data import DataLoader


import matplotlib
import matplotlib.pyplot as plt


# Get embeddings from files
def load_glove_embeddings():
	logging.info("Loading Glove Embeddings........")
	embed_g = []
	with open('glove_embeddings.128.full.txt', 'r') as gfile:
		for line in gfile:
			splitLine = line.split(' ')
			vec_g = np.array(splitLine[1:], dtype='float32')
			embed_g.append(vec_g)
	logging.info("Done. Loaded %d words from GloVe embeddings" % len(embed_g))
	return embed_g


def load_bert_embeddings():
	logging.info("Loading BERT Embeddings........")
	embed_c = []
	with open('bert_embeddings.128.full.txt', 'r') as cfile:
		for line in cfile:
			splitLine = line.split(' ')
			vec_c = np.array(splitLine[1:], dtype='float32')
			embed_c.append(vec_c)
	logging.info("Done. Loaded %d words from BERT embeddings" % len(embed_c))
	return embed_c






# def creat_emb_layer(weight_matrix, non_trainable=False):
# 	num_embeddings, embedding_dim = weight_matrix.size()
# 	emb_layer = nn.Embedding(num_embeddings, embedding_dim)
# 	emb_layer.load_state_dict({'weight': weights_matrix})
# 	if non_trainable:
# 		emb_layer.weight.requires_grad = False

# 	return emb_layer, num_embeddings, embedding_dim


# class SiameseModel(nn.Module):
# 	def __int__(self):
				# super(SiameseModel, self).__int__()
				# self.model1 = nn.Sequential(
				# 		nn.Linear(300, 300),
				# 		nn.ReLU(inplace = True),
				# 		)

				# self.model2 = nn.Sequential(
				# 		# nn.Embedding(input_dim, emb_dimg)
				# 		nn.Linear(1024, 300),
				# 		nn.ReLU(inplace = True),
				# 		)

	# def forward(self, input1, input2):
	# 		output1 = self.model1(input1)
	# 		output2 = self.model2(input2)
	# 		return output1, output2
def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

class SiameseNetwork(nn.Module):
		def __init__(self):
			super(SiameseNetwork, self).__init__()
			self.model1 = nn.Sequential(
					nn.Linear(300, 300),
					nn.ReLU(inplace = True),
					)

			self.model2 = nn.Sequential(
					# nn.Embedding(input_dim, emb_dimg)
					nn.Linear(1024, 300),
					nn.ReLU(inplace = True),
					)

		def forward_once(self, x):
			output = self.model1(x)
			return output

		def forward_twice(self, x):
			output = self.model2(x)
			return output

		def forward(self, input1, input2):
			 output1 = self.forward_once(input1)
			 output2 = self.forward_twice(input2)
			 return output1, output2

class ContrastiveLoss(torch.nn.Module):
		def __init__(self, margin=2.0):
				super(ContrastiveLoss, self).__init__()
				
 
		def forward(self, output1, output2):
				euclidean_distance = F.pairwise_distance(output1, output2)
				loss_contrastive = euclidean_distance**2
 
				return loss_contrastive
		
if __name__ == '__main__':
	# Initialising inputs (contextualised embeddings and word embeddings), and the matrics W and A_i 
	# input_c and input_g should only contain the vectors instead of word+vectors
	embed_g = load_glove_embeddings()
	embed_c = load_bert_embeddings()
	embed_g = np.array(embed_g)
	embed_c = np.array(embed_c)


	# input_c = torch.from_numpy(embed_c)
	# input_g = torch.from_numpy(embed_g)
	# w = torch.randn(1024, 300, require_grad=True)
	# a = np.zeros(300, 300)
	# for i in range(300):
	   #  a[i][i] = 1
	# a = torch.from_numpy(a)
	# a.require_grad_(True)

	# a = load_glove_embeddings()
	# b = load_bert_embeddings()

	print(embed_g.shape)
	print(embed_c.shape)



	net = SiameseNetwork()
	params = net.parameters() # GPU加速
	# print(params)
	criterion = ContrastiveLoss()
	optimizer = optim.Adam(params, lr=0.0005)
	 
	counter = []
	loss_history =[]
	iteration_number =0



	# dataset = dsets.MNIST(root='./data',
	#                       train=True,
	#                       transform=transforms.ToTensor(),
	#                       download=True)
	embed_g = torch.from_numpy(embed_g)
	embed_c = torch.from_numpy(embed_c)

	data_loader1 = torch.utils.data.DataLoader(dataset=embed_g,
	                                          batch_size=200,
	                                          shuffle=False)

	# dataset = dsets.SVHN(root='./data/',
	#                      split='train',
	#                      transform=svhn_transform,
	#                      download=True)
	data_loader2 = torch.utils.data.DataLoader(dataset=embed_c,
	                                          batch_size=200,
	                                          shuffle=False)

	# Covert the dataset to TensorDataset
	#torch_dataset = Data.TensorDataset(input_g=embed_g, input_c=embed_c)

	# Feed the dataset into DataLoader
	# data_loader = Data.DataLoader(dataset=torch_dataset,
	#                                           batch_size=200,
	#                                           shuffle=False)

	# mnist_iter_per_epoch = len(mnist_train_loader) # length of this loader : 600
	# svhn_iter_per_epoch = len(svhn_train_loader) # length of this loader : 733

	# for epoch in range(100):
	#    # Reset the data_iter
	#    if (epoch+1) % mnist_iter_per_epoch ==0:
	#          mnist_iter = iter(mnist_train_loader)
	#    if (epoch+1) % svhn_iter_per_epoch ==0:
	#          svhn_iter = iter(svhn_train_loader

	for epoch in range (0,10):
		#for step, (batch_g, batch_c) in enumerate(data_loader):
			g = iter(data_loader1)
			c = iter(data_loader2)
			print(len(g))
			for i in range (len(g)):
					wordEmbed = next(g)
					# np.array(wordEmbed)
					# print(wordEmbed.shape)
					contEmbed = next(c)
					np.array(contEmbed)
					# print(contEmbed.shape)
					for j in range(len(wordEmbed)):
						input_g = wordEmbed[j]
						input_c = contEmbed[j]
						#print(input_g.shape)
						input_g = np.reshape(input_g, (1, 300))
						input_c = np.reshape(input_c, (1, 1024))

						output1, output2 = net(input_g,input_c)
						optimizer.zero_grad()
						loss_contrastive = criterion(output1, output2)
						loss_contrastive.mean().backward()
						optimizer.step()

						if j%10 == 0:
								print("Epoch:{},  Current loss {}\n".format(epoch,loss_contrastive.data[0]))
								iteration_number += 10
								counter.append(iteration_number)
								loss_history.append(loss_contrastive.data[0])
	show_plot(counter, loss_history)     # plot 损失函数变化曲线


	 
	# for epoch in range(0, Config.train_number_epochs):
	# 			for i, data in enumerate(train_dataloader, 0):
	# 						img0, img1, label = data
	# 						img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()
	# 						output1, output2 = net(img0, img1)
	# 						optimizer.zero_grad()
	# 						loss_contrastive = criterion(output1, output2, label)
	# 						loss_contrastive.backward()
	# 						optimizer.step()
	 
	# 						if i%10 == 0:
	# 									print("Epoch:{},  Current loss {}\n".format(epoch,loss_contrastive.data[0]))
	# 									iteration_number += 10
	# 									counter.append(iteration_number)
	# 									loss_history.append(loss_contrastive.data[0])
	# show_plot(counter, loss_history)     # plot 损失函数变化曲线

					





