import os
import time
import argparse
import torch
import math
import torch.nn as nn
from torch.nn import init
import pandas as pd

from torch.autograd import Variable
import numpy as np
import tqdm as tqdm
import torch.nn.functional as F
import  torchsnooper as torchsnooper
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,confusion_matrix
from torchvision import datasets, transforms
from torch.backends import cudnn
from bisect import bisect_right
import torch.utils.data as Data
from tqdm import tqdm
import os
import torch
torch.cuda.set_device(0)
n_gpu = torch.cuda.device_count()
print(n_gpu)
path=os.path.dirname(os.path.abspath("__file__"))
print(path)
pathlist = ['./oppotunity_sum/wisdm_new/X_train.npy',
            './oppotunity_sum/wisdm_new/y_train.npy',
            './oppotunity_sum/wisdm_new/X_test.npy',
            './oppotunity_sum/wisdm_new/y_test.npy']

a=np.load(pathlist[0])
b=np.load(pathlist[1])
print(a.shape,b.shape)
# # @torchsnooper.snoop()
def data_flat(data_y):
    data_y=np.argmax(data_y, axis=1)
    return data_y


def load_data(path_X,path_y,batchsize):
    train_x=np.load(path_X)
    train_x_shape = train_x.shape
    train_x = torch.from_numpy(
        np.reshape(train_x.astype(float), [train_x_shape[0], 1, train_x_shape[1], train_x_shape[2]])).cuda()


    train_y = data_flat(np.load(path_y))
    train_y = torch.from_numpy(train_y).cuda()

    # print(train_x.shape,train_y.shape,'ppppppppppppppppppp')
    torch_dataset=Data.TensorDataset(train_x,train_y)
    loader=Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0,
    )
    total=len(loader)
    # for _ in tqdm(range(total), desc='进行中', ncols=80,postfix="train_data"):
    #     pass
    return loader

def similarity_matrix(x):
    ''' Calculate adjusted cosine similarity matrix of size x.size(0) x x.size(0). '''
    if x.dim() == 4:
        if x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
            # print('this similarity matrix x shape',x.shape)
        else:
            x = x.view(x.size(0), -1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc ** 2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1, 0)).clamp(-1, 1)
    # print('this similarity matrix x shape\n', R.shape)
    return R

def quzheng_x(height,kernel_size,padding,stride,numlayer):
    list=[]
    for i in range(1,numlayer+1):
        feature=int((height-kernel_size+2*padding)/stride)+1
        height=feature
        list.append(feature)
    return list
def quzheng_s(height,kernel_size,padding,stride,numlayer):
    list=[]
    for i in range(1,numlayer+1):
        feature=math.ceil((height-kernel_size+2*padding)/stride)+1
        height=feature
        list.append(feature)
    return list

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        print('this is loss function!')

    def forward(self, output, label):
        loss_func = F.cross_entropy(output, label)
        return loss_func
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()

        # print(channel_in, channel_out,  kernel, stride, bias,'channel_in, channel_out, kernel, stride, bias')

        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(6,1),stride=(2,1),padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d((4, 1), stride=(1, 1),padding=(1,0)),
)



        self.encoder2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(6,1),stride=(2,1),padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d((4, 1), stride=(1, 1),padding=(1,0)),
)




        self.encoder3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=(6,2),stride=(2,1),padding=0),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d((4, 1), stride=(1, 1),padding=(1,0))
)



        self.linear = nn.Linear(16128, 6)
        # self.soft=nn.Softmax(1)


    def forward(self, x):
        # print(x.shape, 'x.shape')
        x = x.type(torch.cuda.FloatTensor)

        x = self.encoder1(x)
        # print(x.shape, 'encoder1 shape')

        x = self.encoder2(x)
        # print(x.shape, 'encoder2 shape')

        x = self.encoder3(x)
        # print(x.shape, 'encoder3 shape')


        x = x.contiguous().view(x.size(0), -1)

        x = self.linear(x)
        # x = self.soft(x)

        # x = F.softmax(x,dim=1)

        return x


def to_one_hot(y, n_dims=None):
    ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


# input_ch,output_ch,height,num_layers,num_hiden,num_classes

def train(train_loader, test_x_path, test_y_path,test_error,opti):

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Total_Number of params: {} |Trainable_num of params: {}'.format(total_num, trainable_num))
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        target_onehot = to_one_hot(batch_y)
        target_onehot = target_onehot.cuda()
        # print(batch_x.shape,target_onehot.shape,batch_y.shape,'batch_x,target_onehot.shape,batch_y.shape')
        opti.zero_grad()
        # check_parameters(model,2)

        output= model(batch_x)
        # print(output.shape,batch_y.shape,target_onehot.shape,'output.shape')

        loss = loss_func(output, batch_y.long())

        # check_parameters(model, 2)

        # loss_t.backward()
        loss.backward()
        opti.step()



            # if epoch%10==0:
            # print('局部更新')
            # check_parameters(model, 2)
        # check_parameters(model, 16)
        # params = list(model.named_parameters())
        # (name, param) = params[11]
        # print('___________________________________________________________________\n', name, param,
        #       '\n____________________________________________________________________')
        # train_output = torch.max(output, 1)[1].cuda()
        # taccuracy = (torch.sum(train_output == batch_y.long()).type(torch.FloatTensor) / batch_y.size(0)).cuda()
        # print(taccuracy,'train_accuracy')
    if epoch % 1 == 0:
        model.eval()


        test_x = np.load(test_x_path)
        test_x_shape = test_x.shape
        test_x = torch.from_numpy(np.reshape(test_x, [test_x_shape[0], 1, test_x_shape[1], test_x_shape[2]])).cuda()



        test_y = data_flat(np.load(test_y_path))
        test_y = torch.from_numpy(test_y).cuda()

        test_y_onehot = to_one_hot(test_y)
        test_y_onehot = test_y_onehot.cuda()

        # print(test_x.shape, test_y.shape, test_y_onehot.shape, 'test_x.shape,test_y.shape,target_y.shape')

        try:
            test_output = model(test_x)
            test_output_copy = test_output
            print(test_output.shape)
            test_output = data_flat(test_output.cpu().detach().numpy())
            # print(test_output.shape, 'test_output')

            test_output_f1 = np.asarray(pd.get_dummies(test_output))

            print(test_y_onehot.shape, test_output_f1.shape)
            acc = accuracy_score(test_y_onehot.cpu().numpy(), test_output_f1)
            f1 = f1_score(test_y_onehot.cpu().numpy(), test_output_f1, average='weighted')
            f2 = f1_score(test_y_onehot.cpu().numpy(), test_output_f1, average='micro')
            f3 = f1_score(test_y_onehot.cpu().numpy(), test_output_f1, average='macro')
            reca = recall_score(test_y_onehot.cpu().numpy(), test_output_f1, average='weighted')
            # print(confusion_y.tolist(), '\n', test_output_f1_con.tolist())
            print('Epoch: ', epoch, '| test accuracy: %.8f' % acc, '| test F1: %.8f' % f1, '| test recall: %.8f' % reca,
                  '| test micro: %.8f' % f2, '| test micro: %.8f' % f3)
            #
        except ValueError:
            print('error')
        else:
            pass
        test_output = torch.max(test_output_copy, 1)[1].cuda()
        # print(test_output.shape,'test_output.shape')
        accuracy = (torch.sum(test_output == test_y.long()).type(torch.FloatTensor) / test_y.size(0)).cuda()
        print('Epoch: ', epoch, '| test accuracy: %.8f' % accuracy)
        test_error.append((1 - accuracy.item()))
        model.train()
    # np.save('./matplotlib_picture/WISDM_error/bp_test.npy', test_error)

if __name__ == '__main__':

    model = cnn()
    model.cuda()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.8e-3)
    # schedual=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100,200,300],gamma=0.99)
    loss_func = nn.CrossEntropyLoss().cuda()
    train_loader = load_data(pathlist[0], pathlist[1], batchsize=200)
    test_error = []
    for epoch in range(500):
        train(train_loader, pathlist[2], pathlist[3],test_error,optimizer)



