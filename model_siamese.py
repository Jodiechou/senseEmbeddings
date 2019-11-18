import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 



# Initialising inputs (contextualised embeddings and word embeddings), and the matrics W and A_i 
# input_c and input_g should only contain the vectors instead of word+vectors
input_c = torch.from_numpy(emb_c)
input_g = torch.from_numpy(emb_g)
w = torch.randn(1024, 300, require_grad=True)
a = np.zeros(300, 300)
for i in range(300):
	a[i][i] = 1
a = torch.from_numpy(a)
a.require_grad_(True)

def creat_emb_layer(weight_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weight_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class SiameseModel(nn.Module):
	def __int__(self, input_dim, emb_dimc, emb_dimg):
        super(SiameseModel, self).__int__()
        self.model1 = nn.Sequential(
            nn.Embedding(input_dim, emb_dimc)
            nn.Linear(emb_dimc, 300)
            nn.ReLU(inplace = True))

        self.model2 = nn.Sequential(
            nn.Embedding(input_dim, emb_dimg)
            nn.Linear(emb_dimg, 300)
            nn.ReLU(inplace = True))
		# self.input_dim = input_dim
  #       #self.emb_dim = emb_dim
  #       self.hid_dim = hid_dim
  #       self.n_layers = n_layers
  #       self.dropout = dropout
        
  #       self.embedding = nn.Embedding(input_dim, emb_dim) 
  #       self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
  #       self.out = nn.Linear(hid_dim, out_dim)

  #       self.dropout = nn.Dropout(dropout)

    def forward(self, input1, input2):
        output1 = self.model1(input1)
        output2 = self. model2(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        
 
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = euclidean_distance**2
 
        return loss_contrastive
    
#     def loss = 

net = Siamesemodel()    # GPU加速
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)
 
counter = []
loss_history =[]
iteration_number =0



# dataset = dsets.MNIST(root='./data',
#                       train=True,
#                       transform=transforms.ToTensor(),
#                       download=True)

# mnist_data_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                           batch_size=100,
#                                           shuffle=True)

# dataset = dsets.SVHN(root='./data/',
#                      split='train',
#                      transform=svhn_transform,
#                      download=True)
# svhn_data_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                           batch_size=100,
#                                           shuffle=True)

# mnist_iter_per_epoch = len(mnist_train_loader) # length of this loader : 600
# svhn_iter_per_epoch = len(svhn_train_loader) # length of this loader : 733

# for epoch in range(100):
#    # Reset the data_iter
#    if (epoch+1) % mnist_iter_per_epoch ==0:
#          mnist_iter = iter(mnist_train_loader)
#    if (epoch+1) % svhn_iter_per_epoch ==0:
#          svhn_iter = iter(svhn_train_loader

for epoch in range (0,10):
    c = iter(dataload1)
    g = iter(dataload2)
    for i in range (len):
        wordm = next(c)
        conm = next(g)
        output1, output2 = net(wordm,conm)
        optimizer.zero_grad()
        loss_contrastive = criterion(output1, output2)
        loss_contrastive.backward()
        optimizer.step()

        if i%10 == 0:
            print("Epoch:{},  Current loss {}\n".format(epoch,loss_contrastive.data[0]))
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.data[0])
show_plot(counter, loss_history)     # plot 损失函数变化曲线


 
for epoch in range(0, Config.train_number_epochs):
      for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()
            output1, output2 = net(img0, img1)
            optimizer.zero_grad()
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
 
            if i%10 == 0:
                  print("Epoch:{},  Current loss {}\n".format(epoch,loss_contrastive.data[0]))
                  iteration_number += 10
                  counter.append(iteration_number)
                  loss_history.append(loss_contrastive.data[0])
show_plot(counter, loss_history)     # plot 损失函数变化曲线

        





