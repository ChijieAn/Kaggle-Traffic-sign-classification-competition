import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from Python_files.train_functions import (MyDataset,train,validation)
from STN_model import (STN_block,Net)
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
import matplotlib.pyplot as plt
import pickle
import pandas as pd

#fix some basic parameters
batch_size = 32
momentum = 0.9
lr = 0.01
epochs = 5
log_interval = 100

train_dataset = MyDataset(X_path="train/X.pt", y_path="train/y.pt")
val_dataset = MyDataset(X_path="validation/X.pt", y_path="validation/y.pt")

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

# Model Parameters:
in_channels=3
in_wh=32
fc1_dim=150
num_classes = 43

conv_params = {
     'out_channels':[150,200,300],
     'kernel_size':[7,4,4],
     'stride':[1,1,1],
     'padding':[2,2,2],
     'stn_ch1':[200, 300],
     'stn_ch2': [150, 150],
}

model = Net(in_channels, in_wh, fc1_dim, num_classes, conv_params)

#use swall model to improve the performance of model
swa_model = AveragedModel(model)
swa_model=swa_model.to('cuda')

#define some training parameters
epochs=300
swa_start=40
swa_lr=5e-5
weight_decay = 0.0001

#define the optimizer
optimizer = optim.Adam(model.parameters(), lr = 0.0001, betas = (0.9,0.999))

#Here I set a scheduler to adjust the learning rate.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)
swa_scheduler=SWALR(optimizer, swa_lr = swa_lr)

lst_train_loss=[]
lst_validation_loss=[]

for epoch in range(swa_start):
    lst_train_loss.append(train(epoch))
    lst_validation_loss.append(validation(model,scheduler))
    model_file = 'model_STN_10' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '.')

for epoch in range(swa_start,epochs):
     lst_train_loss.append(train(epoch))
     swa_model.update_parameters(model)
     torch.optim.swa_utils.update_bn(train_loader, swa_model,device='cuda')
     lst_validation_loss.append(validation(swa_model,swa_scheduler))
     model_file = 'model_STN_swa3' + str(epoch) + '.pth'
     torch.save(swa_model.state_dict(), model_file)
     print('\nSaved swa model to ' + model_file + '.')


#plot the training and validation loss
plt.plot([i for i in range(len(lst_train_loss))],lst_train_loss,label='training loss')
plt.plot([i for i in range(len(lst_validation_loss))],lst_validation_loss,label='validation_loss')
plt.xlabel('training epochs')
plt.ylabel('loss')

#create submission file to Kaggle
outfile = 'gtsrb_kaggle_swa_final_final.csv'

output_file = open(outfile, "w")
dataframe_dict = {"Filename" : [], "ClassId": []}

model_99=AveragedModel(Net(in_channels, in_wh, fc1_dim, num_classes, conv_params))
model_99.load_state_dict(torch.load('model_STN_swa3251.pth'))
model_99=model_99.to('cuda')

test_data = torch.load('testing/test.pt')
file_ids = pickle.load(open('testing/file_ids.pkl', 'rb'))
model_99.eval() # Don't forget to put your model on eval mode !

for i, data in enumerate(test_data):
    data = data.unsqueeze(0)
    data=data.to('cuda')
    output = model_99(data)
    pred = output.data.max(1, keepdim=True)[1].item()
    file_id = file_ids[i][0:5]
    dataframe_dict['Filename'].append(file_id)
    dataframe_dict['ClassId'].append(pred)

df = pd.DataFrame(data=dataframe_dict)
df.to_csv(outfile, index=False)
print("Written to csv file {}".format(outfile))

