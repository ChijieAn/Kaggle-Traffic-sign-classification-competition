#construct the new STN model
#explaination of the model structure:
#the STN model mainly contain to parts, the core part of this model is the STN block.
#The STN block mainly consists of a local network, which is a two layer concolutional neural network. Also a local function, which is a two layer perceptron
#The two layer perceptron, which outpouts the prediction for the parameter to implement transformations on the original image. Then the transformed image is fed into the CNN
#By applying this transformation to the image, the model can capture the details in the image better.
#The overall model consists of three layers of convolutional neural network and two STN blocks.
#The detailed parameters and the structure of the model is printed in the code below.
#I refered to the baseline code of this model on Github. The initial model was for the input of size Batch*3*48*48, and by changing some structures and parameters in the model,
#I made the model can successfully make a good performanc on the dataset of this challenge
#This model is based on the paper https://arxiv.org/pdf/1506.02025.pdf
#Code resource: Github Reference: https://github.com/topics/spatial-transformer-network

import torch
import torch.nn as nn
import torch.nn.functional as F
class STN_block(nn.Module):
  def __init__(self,in_channels,in_wh,out_channels_1,out_channels_2,fc_loc_out,kernel_size,stride,padding,index):
    super(STN_block,self).__init__()
    #construct the localization network
    self.index=index
    if index==1:
      self.loc_net=nn.Sequential(
          #first convolution layer
          nn.Conv2d(in_channels,out_channels_1,kernel_size,stride,padding=padding),
          nn.BatchNorm2d(out_channels_1),
          nn.ReLU(),
          nn.MaxPool2d(2),
          #second convolution layer
          nn.Conv2d(out_channels_1,out_channels_2,kernel_size,stride,padding=padding),
          nn.BatchNorm2d(out_channels_2),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
    elif index==2:
       self.loc_net=nn.Sequential(
          #first convolution layer
          nn.Conv2d(in_channels,out_channels_1,kernel_size,stride,padding=padding),
          nn.BatchNorm2d(out_channels_1),
          nn.ReLU(),
          nn.MaxPool2d(2),
          #second convolution layer
          nn.Conv2d(out_channels_1,out_channels_2,kernel_size-2,stride,padding=padding),
          nn.BatchNorm2d(out_channels_2),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
    if index==1:
      self.out_size = (((in_wh - kernel_size + 1) // 2 + padding)  - kernel_size + 1) // 2 + padding
      self.fc_loc_in=out_channels_2*self.out_size*self.out_size
    elif index==2:
      self.out_size = (((in_wh - kernel_size + 1) // 2 + padding)  - (kernel_size-2) + 1) // 2 + padding
      self.fc_loc_in=out_channels_2*self.out_size*self.out_size

    #Regressor for the 3*2 affine matrix
    self.fc_loc=nn.Sequential(
        nn.Linear(self.fc_loc_in,fc_loc_out),
        nn.BatchNorm1d(fc_loc_out),
        nn.ReLU(),
        nn.Linear(fc_loc_out,3*2)
    )

    #initialize the weights/bias with identity transformation
    self.fc_loc[-1].weight.data.zero_()
    self.fc_loc[-1].bias.data.copy_(torch.tensor([1,0,0,0,1,0],dtype=torch.float))

  #define the forward function
  def forward(self,x):
    #print(self.index)
    #print(x.shape)
    x_loc=self.loc_net(x)
    #print(self.out_size,self.fc_loc_in)
    #print(x_loc.shape)
    x_loc=x_loc.view(-1,self.fc_loc_in)
    theta=self.fc_loc(x_loc)
    theta=theta.view(-1,2,3)
    grid=F.affine_grid(theta,x.size(),align_corners=True)
    x=F.grid_sample(x,grid,align_corners=True)
    return x

class Net(nn.Module):
  #we implement 2 STN bolcks with 3 convolutional layers in this model
  def __init__(self,in_channels,in_wh,fc1_dim,num_classes,conv_params):
    super(Net,self).__init__()

    self.stn1=STN_block(in_channels,in_wh, conv_params['stn_ch1'][0],conv_params['stn_ch1'][1], fc_loc_out = 200, kernel_size = 5, stride = 1, padding = 2,index=1)
    self.conv1 = nn.Conv2d(in_channels, conv_params['out_channels'][0], conv_params['kernel_size'][0], conv_params['stride'][0], conv_params['padding'][0])
    self.bn1 = nn.BatchNorm2d(conv_params['out_channels'][0])

    self.conv2 = nn.Conv2d(conv_params['out_channels'][0], conv_params['out_channels'][1], conv_params['kernel_size'][1], conv_params['stride'][1], conv_params['padding'][1])
    self.stn2 = STN_block(conv_params['out_channels'][1], 8, conv_params['stn_ch2'][0], conv_params['stn_ch2'][1], fc_loc_out = 150, kernel_size = 3, stride = 1, padding = 2,index=2)
    self.bn2 = nn.BatchNorm2d(conv_params['out_channels'][1])

    self.conv3 = nn.Conv2d(conv_params['out_channels'][1], conv_params['out_channels'][2], conv_params['kernel_size'][2], conv_params['stride'][2], conv_params['padding'][2])
    self.bn3 = nn.BatchNorm2d(conv_params['out_channels'][2])

    self.fc_dim = conv_params['out_channels'][2] * 4 * 4

    self.fc1 = nn.Linear(self.fc_dim, fc1_dim)
    self.bn4 = nn.BatchNorm1d(fc1_dim)
    self.fc2 = nn.Linear(fc1_dim, num_classes)
    self.drop = nn.Dropout(p=0.5)

  def forward(self, x):
        #the first convolutional layer with the first STN layer
      x = torch.max_pool2d(F.relu(self.bn1(self.conv1(self.stn1(x)))), 2)
      #print('first STN layer finished',x.shape)
        #the second convolutional layer
      x = torch.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
      #print('second conv layer finished',x.shape)
        #the third convolutional layer with the second STN layer
      x = torch.max_pool2d(F.relu(self.bn3(self.conv3(self.stn2(x)))), 2)
      #print('third STN layer finished',x.shape)
      x = x.view(-1, self.fc_dim)
      x=self.fc1(x)
      #print(x.shape)
      x=self.bn4(x)
      #print(x.shape)
      x = F.relu(x)
      x = self.fc2(self.drop(x))
      return F.log_softmax(x, dim=1)