
import cPickle
import numpy as np
import warnings
import time
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import cPickle
import numpy
import os




def get_batch(train, train_label, batch_index, batch_size):
    data_size = len(train_label)
    if data_size>=(batch_index+1)*batch_size:
        train_x = train[batch_index*batch_size:(batch_index+1)*batch_size]
        train_y = train_label[batch_index*batch_size:(batch_index+1)*batch_size]    
    else:
        train_x = train[batch_index*batch_size:data_size]
        train_y = train_label[batch_index*batch_size:data_size]        
    return train_x, train_y

class Lenet(nn.Module):
     def __init__(self):
         super(Lenet, self).__init__()

         self.conv1 = nn.Conv2d(1, 200, kernel_size = (3,300), stride = 1,padding=0)   
         self.pool1 = nn.MaxPool2d((62,1),1)
#         
         self.fc1 = nn.Linear(200, 100)
         self.fc2 = nn.Linear(100, 2)
  
       
     def forward(self, x_input):
         x = F.relu(self.conv1(x_input))
         x = self.pool1(x)
        
         x = x.view(x.size(0), -1)
         x = self.fc1(x)
         x = self.fc2(x)
         out = F.softmax(x, dim=1)

         return out

class LSTM_for_text(nn.Module):
     def __init__(self, embedding_dim=300, hidden_dim=300, batch_size = 40, use_gpu = False):
#     def __init__(self, batch_size = 40):
          super(LSTM_for_text, self).__init__()
          
          self.batch_size = batch_size
          self.embedding_dim = embedding_dim
          self.hidden_dim = hidden_dim
          self.use_gpu = use_gpu
    
          self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)
          
          self.hidden = self.init_hidden()
          
#          self.conv1 = nn.Conv2d(1, 200, kernel_size = (3,300), stride = 1,padding=0)   
#          self.pool1 = nn.MaxPool2d((62,1),1)
#          
          self.fc1 = nn.Linear(hidden_dim, 100)
          self.fc2 = nn.Linear(100, 2)

     def init_hidden(self):
#          Before we've done anything, we dont have any hidden state.
#          Refer to the Pytorch documentation to see exactly
#          why they have this dimensionality.
         # The axes semantics are (num_layers, minibatch_size, hidden_dim)
         if self.use_gpu:
             original_hidden = torch.zeros(1, self.batch_size, self.hidden_dim).cuda(), torch.zeros(1, self.batch_size, self.hidden_dim).cuda()
         else:
             original_hidden = torch.zeros(1, self.batch_size, self.hidden_dim), torch.zeros(1, self.batch_size, self.hidden_dim)
         return original_hidden 
        
     def forward(self, x_input):         
          inputs = x_input.view(x_input.size(2), self.batch_size, -1)
          print "inputs: ", inputs.size()
          print "hidden: ", self.hidden[0].size(), self.hidden[1].size()
          lstm_out, self.hidden = self.lstm(inputs, self.hidden)
          fc1_in = lstm_out[-1]
          fc2_in = self.fc1(fc1_in)
          fc2_out = self.fc2(fc2_in)
          out = F.softmax(fc2_out, dim=1)
#          lstm_out = fc1_in
          print "fc1_in: ", out.size()
          return out
# =============================================================================
#     def __init__(self):
# 
#         super(LSTMtext, self).__init__()
# #        self.embedding_dim = embedding_dim
# #        self.hidden_dim = hidden_dim
# #        self.batch_size = batch_size
# 
#         # The LSTM takes word embeddings as inputs, and outputs hidden states
#         # with dimensionality hidden_dim.
# #        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)
# #        self.lstm = nn.LSTM(300, 300)
# 
#         # The linear layer that maps from hidden state space to tag space
#         #self.fc1 = nn.Linear(hidden_dim, tagset_size)
#         self.fc1 = nn.Linear(300, 100)
# #        self.fc2 = nn.Linear(100, 2)
#          
# #        self.hidden = self.init_hidden()
# 
#     def init_hidden(self):
#         # Before we've done anything, we dont have any hidden state.
#         # Refer to the Pytorch documentation to see exactly
#         # why they have this dimensionality.
#         # The axes semantics are (num_layers, minibatch_size, hidden_dim)
#         return (torch.zeros(1, self.batch_size, self.hidden_dim),
#                 torch.zeros(1, self.batch_size, self.hidden_dim))
#             
#     def foward(self, x_input):
# #         lstm_out, self.hidden = self.lstm(
# #                 x_input.view(len(x_input), 1, -1), self.hidden)
# #     
# #         out = lstm_out
# #         out = x_input.view(x_input.shape[2], self.batch_size, -1)
#         out = x_input
#         return out
#     
# #    def forward(self, sentence):
# #        embeds = self.word_embeddings(sentence)
# #        lstm_out, self.hidden = self.lstm(
# #            embeds.view(len(sentence), 1, -1), self.hidden)
# #        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
# #        tag_scores = F.log_softmax(tag_space, dim=1)
# #        return tag_scores
# =============================================================================


def get_correct_num(predict, y_input):
    if use_gpu:
        correct_num = (float)((predict.data.cpu().numpy() == y_input.data.cpu().numpy()).sum())
    else:
        correct_num = (float)((predict.data.numpy() == y_input.data.numpy()).sum())
    return correct_num

def print_results(confidence, prediction, label, cv_index) :
    tobecheckdir =  "results/" + str(cv_index)
    if os.path.isdir(tobecheckdir)==False:
        os.makedirs(tobecheckdir)
    
#    write_f = file(tobecheckdir+ "/confidence.out",'w+')
#    write_f.write(str(confidence)+ "\n")
#    write_f.close()
    
    numpy.savetxt(tobecheckdir+ "/confidence.out", confidence)
    numpy.savetxt(tobecheckdir+ "/prediction.out", prediction)
    numpy.savetxt(tobecheckdir+ "/label.out", label)

#def remove_fillin(data_list):
       
   
if __name__=="__main__":
    use_gpu = torch.cuda.is_available()
#    use_gpu = False
    print "use_gpu:", use_gpu
    print "loading data...",
    x = cPickle.load(open("data_test.p","rb"))
    W, word_idx_map, vocab, train_list, test_list = x[0], x[1], x[2], x[3], x[4]
    
#    W = W[:,:3]
    print "data loaded!"
    U = W
    results = []    
    learning_rate = 1e-2
    batch_size = 40
     
    acc_all = []
    cv_times = 1  
    for cv_index in xrange(0,cv_times):
        train_set = train_list[cv_index]
        train_set_x = train_set[:,0:-1]
        train_set_y = train_set[:,-1]    
# =============================================================================
#         test_set = test_list[cv_index]
#         test_set_x = test_set[:,0:-1]
#         test_set_y = test_set[:,-1]
#         
#         x_test = W[test_set_x.flatten()].reshape((test_set_x.shape[0], 1, test_set_x.shape[1],W.shape[1]))  
#         y_test = test_set_y        
#         if use_gpu:
#             x_test_input = Variable(torch.from_numpy(x_test).cuda())
#             y_test_input = Variable(torch.from_numpy(y_test).type(torch.LongTensor).cuda())
#         else:
#             x_test_input = Variable(torch.from_numpy(x_test))
#             y_test_input = Variable(torch.from_numpy(y_test).type(torch.LongTensor))  
#    
# =============================================================================
    
        if use_gpu:
#            lenet = Lenet().cuda()
            lenet = LSTM_for_text(embedding_dim=300, hidden_dim=300, batch_size=batch_size, use_gpu = use_gpu).cuda()
        else:
#            lenet = Lenet()
            lenet = LSTM_for_text(embedding_dim=300, hidden_dim=300, batch_size=batch_size, use_gpu = use_gpu)
            
        criterian_cross = nn.CrossEntropyLoss(size_average=False)
        
        optimizer = optim.SGD(lenet.parameters(), lr=learning_rate)
#        optimizer = optim.Adadelta(lenet.parameters(), lr=1.0, rho=0.95)
#        optimizer = optim.Adam(lenet.parameters(), weight_decay=1e-4)
        
# =============================================================================
#         params = lenet.parameters()
#         for name, parameters in lenet.named_parameters():
#                  print name, ": ", parameters.size()
# =============================================================================
        

        epochs = 1
        for epoch_i in range(epochs):
             since = time.time()
             running_loss = 0.
             running_acc = 0.
             correct_num = 0
             
             train_size = len(train_set_y)
             batch_num = train_size/batch_size+1
                     
             for batch_index in xrange(batch_num):
                 print "batch_index: ", batch_index
                 train_batch_x, train_batch_y = get_batch(train_set_x, train_set_y, batch_index, batch_size)   

                 x = W[train_batch_x.flatten()].reshape((train_batch_x.shape[0], 1, train_batch_x.shape[1],W.shape[1]))  
                 y = train_batch_y
                 
                 if use_gpu:
                     x_input = Variable(torch.from_numpy(x).cuda())
                     y_input = Variable(torch.from_numpy(y).type(torch.LongTensor).cuda())
                 else:
                     x_input = Variable(torch.from_numpy(x))
                     y_input = Variable(torch.from_numpy(y).type(torch.LongTensor))

                 optimizer.zero_grad()
                 
#                 print x_input.shape[0]
                 
                 lenet.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
                 lenet.hidden = lenet.init_hidden()
                 
                 print "x_inputs: ", x_input.size()
                 output_f = lenet(x_input)
#                 print "output: ", output_f
#                 print output_f
#                 print output_f
             
                 cross_loss = criterian_cross(output_f, y_input)      
#             
                 _, predict = torch.max(output_f, 1)

#    
                 correct_num = correct_num +get_correct_num(predict,y_input)
                 running_loss = running_loss + cross_loss
#                 
#                 cross_loss.backward()
#                 optimizer.step()
                 
#                 break
#             running_los = (float)(running_loss)/len(train_set_y)
#             running_acc = (float)(correct_num)/len(train_set_y)
#             print("[%d/%d] Loss: %.5f, Acc: %.2f%%,Time: %.1f s" %(epoch_i+1, epochs, running_los, 100*running_acc, time.time()-since))

# =============================================================================
#              output_test = lenet(x_test_input)
#              test_loss = criterian_cross(output_test, y_test_input) 
#              _, predict_test = torch.max(output_test, 1)
#              
#              if use_gpu:
#                 test_confidence = output_test.cpu().data[:,1].numpy()
#                 test_prediction = predict_test.cpu().data.numpy()    
#              else:
#                 test_confidence = output_test.data[:,1].numpy()
#                 test_prediction = predict_test.data.numpy()
#                                 
#              test_correct = get_correct_num(predict_test, y_test_input)
#              test_acc = (float)(test_correct)/len(test_set)
#              test_loss = (float)(test_loss)/len(test_set)
#              
#              print("[%d/%d] Loss: %.5f, Acc: %.2f%%, Test Loss: %.5f, Test Acc: %.2f%%,Time: %.1f s" %(epoch_i+1, epochs, cross_loss, 100*running_acc, test_loss, 100*test_acc, time.time()-since))
#         
#         acc_all.append(test_acc)
#         print_results(test_confidence, test_prediction, y_test, cv_index )
#    
#     writer = file("results/results_cnn_pytorch.txt","w+")
#     
#     for i in xrange(cv_times):
#          writer.write(str(acc_all[i])+"\n")
#     writer.write(str(numpy.mean(acc_all)))
#     writer.close()             
# =============================================================================
    
    
    
             

