import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

import matplotlib
import matplotlib.pyplot as plt

import logging
from datetime import datetime


# Get embeddings from files
def load_glove_embeddings():
	logging.info("Loading Glove Embeddings........")
	embed_g = []
	with open('data/vectors/glove_embeddings.full.txt', 'r') as gfile:
		for line in gfile:
			splitLine = line.split(' ')
			vec_g = np.array(splitLine[1:], dtype='float32')
			embed_g.append(vec_g)
	logging.info("Done. Loaded %d words from GloVe embeddings" % len(embed_g))
	return embed_g


def load_bert_embeddings():
	logging.info("Loading BERT Embeddings........")
	embed_c = []
	with open('data/vectors/bert_embeddings.full.txt', 'r') as cfile:
		for line in cfile:
			splitLine = line.split(' ')
			vec_c = np.array(splitLine[1:], dtype='float32')
			embed_c.append(vec_c)
	logging.info("Done. Loaded %d words from BERT embeddings" % len(embed_c))
	return embed_c

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
	train_counter = []
	valid_counter = []
	train_loss =[]
	valid_loss = []

	min_valid_loss = np.inf
	iteration_number_train = 0
	iteration_number_valid = 0
	EPOCH = 20
	batch_size = 200 
	validation_split = 0.2
	shuffle_dataset = True
	lr = 0.001

	embed_g = load_glove_embeddings()
	embed_c = load_bert_embeddings()
	embed_g = np.array(embed_g)
	embed_c = np.array(embed_c)

	print("Shape of embed_g: ", embed_g.shape)
	print("Shape of embed_g: ", embed_c.shape)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#print (torch.cuda.is_available())

	siamese = SiameseNetwork().to(device)
	params = siamese.parameters() 
	
	
	criterion = ContrastiveLoss()
	optimizer = optim.Adam(params, lr)

	# mult_step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
 #                            milestones=[EPOCH//2, EPOCH//4*3], gamma=0.1)

	mult_step_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
							mode='min', factor=0.1, patience=10, verbose=True)



	embed_g = torch.from_numpy(embed_g)
	embed_c = torch.from_numpy(embed_c)


	# Creating data indices for training and validation splits:
	dataset_size = len(embed_g)
	indices = list(range(dataset_size))
	split = int(np.floor(validation_split * dataset_size))
	if shuffle_dataset :
		np.random.seed(42)
		np.random.shuffle(indices)
	train_indices, val_indices = indices[split:], indices[:split]

	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)


	# Covert the dataset to TensorDataset
	torch_dataset = Data.TensorDataset(embed_g, embed_c)

	# Feed the dataset into DataLoader
	train_loader = Data.DataLoader(dataset=torch_dataset,
											  batch_size=200,
											  sampler=train_sampler)

	valid_loader = Data.DataLoader(dataset=torch_dataset,
											  batch_size=200,
											  sampler=valid_sampler)


	
	for epoch in range (EPOCH):

		# Training
		total_train_loss = []

		siamese.train()
		for step, (train_g, train_c) in enumerate(train_loader):			
			for i in range (len(train_g)):
				wordEmbed = train_g[i]
				wordEmbed = np.reshape(wordEmbed, (1, 300)).to(device)
				contEmbed = train_c[i]
				contEmbed = np.reshape(contEmbed, (1, 1024)).to(device)
				output1, output2 = siamese(wordEmbed,contEmbed)
				optimizer.zero_grad()
				loss_contrastive = criterion(output1, output2)
				#loss_contrastive.mean().backward()
				loss_contrastive.backward()
				optimizer.step()

				total_train_loss.append(loss_contrastive.item())
		train_loss.append(np.mean(total_train_loss))


		# Validation
		total_valid_loss = []
		
		siamese.eval()
		for step, (valid_g, valid_c) in enumerate(valid_loader):
			for j in range (len(valid_g)):
				wordEmbed = valid_g[j]
				wordEmbed = np.reshape(wordEmbed, (1, 300)).to(device)
				contEmbed = valid_c[j]
				contEmbed = np.reshape(contEmbed, (1, 1024)).to(device)
				output1, output2 = siamese(wordEmbed, contEmbed)	
				loss_contrastive = criterion(output1, output2)
				total_valid_loss.append(loss_contrastive.item()) 
		valid_loss.append(np.mean(total_valid_loss))


		if (valid_loss[-1] < min_valid_loss):     
			torch.save({'epoch': i, 'model': siamese, 'train_loss': train_loss,
					'valid_loss': valid_loss},'model/Siamese.model') 
	        
			min_valid_loss = valid_loss[-1]


		log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, '
						  'best_valid_loss: {:0.6f}, lr: {:0.7f}').format((epoch + 1), EPOCH,
																		  train_loss[-1],
																		  valid_loss[-1],
																		  min_valid_loss,
																		  optimizer.param_groups[0]['lr'])

		mult_step_scheduler.step(valid_loss[-1])

		print(log_string)
		