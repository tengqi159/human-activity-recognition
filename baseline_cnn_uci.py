import os
import time
import argparse
import torch
import math
import torch.nn as nn
from torch.nn import init
import pandas as pd
from tensorboardX import SummaryWriter
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
import random
import torch
torch.cuda.set_device(0)
n_gpu = torch.cuda.device_count()
print(n_gpu)
path=os.path.dirname(os.path.abspath("__file__"))
print(path)
pathlist = ['./UCI/x_train.npy',
            './UCI/y_train.npy',
            './UCI/x_test.npy',
            './UCI/y_test.npy']


GLOBAL_SEED = 1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


GLOBAL_WORKER_ID = None


def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id



# this is UCIdataset. torch.Size([7352,128, 9]) torch.Size([7352]) windows size:128 channel:9 calss:6 overlap:50%



# # @torchsnooper.snoop()
def data_flat(data_y):
    data_y=np.argmax(data_y, axis=1)
    return data_y


def load_data(train_x_path, train_y_path, batchsize):
    train_x = np.load(train_x_path)
    train_x_shape = train_x.shape
    train_x = torch.from_numpy(
        np.reshape(train_x.astype(float), [train_x_shape[0],1, train_x_shape[1], train_x_shape[2]])).cuda()

    train_y = data_flat(np.load(train_y_path))
    train_y = torch.from_numpy(train_y).cuda()

    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n',
          train_x.shape, train_y.shape,
          '\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    torch_dataset = Data.TensorDataset(train_x, train_y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0,
        worker_init_fn=worker_init_fn
    )
    total = len(loader)
    # for _ in tqdm(range(total), desc='ongoing', ncols=80,postfix="train_data"):
    #     pass
    return loader

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()

        # print(channel_in, channel_out,  kernel, stride, bias,'channel_in, channel_out, kernel, stride, bias')

        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(6,1),stride=(3,1),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d((4, 1), stride=(1, 1),padding=(1,0)),
)



        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(6,1),stride=(3,1),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d((4, 1), stride=(1, 1),padding=(1,0)),
)




        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(6,1),stride=(3,1),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d((4, 1), stride=(1, 1),padding=(1,0))
)


        self.out = nn.Linear(15360, 6)
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

        x = self.out(x)
        # x = self.soft(x)

        # x = F.softmax(x,dim=1)

        return x


def to_one_hot(y, n_dims=None):
    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


def train(EPOCH,train_loader, test_x_path, test_y_path,train_error,test_error):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss_total_global = 0
    print('Total_Number of params: {} |Trainable_num of params: {}'.format(total_num, trainable_num))

    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        target_onehot = to_one_hot(batch_y)
        target_onehot = target_onehot.cuda()

        optimizer.zero_grad()
        output = model(batch_x)
        # print(output.shape,batch_y.shape,target_onehot.shape,'output.shape')

        loss = loss_func(output, batch_y.long())
        loss_total_global += loss.item() * batch_x.size(0)

        loss.backward()
        optimizer.step()

    train_output = torch.max(output, 1)[1].cuda()
    taccuracy = (torch.sum(train_output == batch_y.long()).type(torch.FloatTensor) / batch_y.size(0)).cuda()
    error = 1 - taccuracy.item()
    train_error.append(error)
    print('EPOCH',EPOCH,'train accuracy',taccuracy.item())

    # np.save('./matplotlib_picture/UCI_error/bp_train.npy', train_error)
    if epoch % 1 == 0:
        model.eval()

        test_x = np.load(test_x_path)
        test_x_shape = test_x.shape
        test_x = torch.from_numpy(np.reshape(test_x, [test_x_shape[0],1, test_x_shape[1], test_x_shape[2]])).cuda()

        test_y = data_flat(np.load(pathlist[3]))
        test_y = torch.from_numpy(test_y).cuda()

        test_y_onehot = to_one_hot(test_y)
        test_y_onehot = test_y_onehot.cuda()

        # print(test_x.shape, test_y.shape, test_y_onehot.shape, 'test_x.shape,test_y.shape,target_y.shape')
        try:
            test_output= model(test_x)
            test_output_copy=test_output
            # print(test_output.shape)
            test_output=data_flat(test_output.cpu().detach().numpy())
            # print(test_output.shape, 'test_output')

            test_output_f1 =np.asarray(pd.get_dummies(test_output))

            print(test_y_onehot.shape, test_output_f1.shape)
            acc = accuracy_score(test_y_onehot.cpu().numpy(), test_output_f1)
            f1 = f1_score(test_y_onehot.cpu().numpy(), test_output_f1, average='weighted')
            f2 = f1_score(test_y_onehot.cpu().numpy(), test_output_f1, average='micro')
            f3 = f1_score(test_y_onehot.cpu().numpy(), test_output_f1, average='macro')
            reca = recall_score(test_y_onehot.cpu().numpy(), test_output_f1,average='weighted')
            # print(confusion_y.tolist(), '\n', test_output_f1_con.tolist())
            print('Epoch: ', epoch, '| test accuracy: %.8f' % acc, '| test F1: %.8f' % f1, '| test recall: %.8f' % reca, '| test micro: %.8f' % f2, '| test micro: %.8f' % f3)
        #
            test_output = torch.max(test_output_copy, 1)[1].cuda()
            # print(test_output.shape,'test_output.shape')
            accuracy = (torch.sum(test_output == test_y.long()).type(torch.FloatTensor) / test_y.size(0)).cuda()
            # print('Epoch: ', epoch, '| test accuracy: %.8f' % accuracy)
            test_error.append((1-accuracy.item()))
            model.train()
        except ValueError:
            print('error')
        else:
            pass

    # np.save('./matplotlib_picture/UCI_error/bp_test.npy',test_error)
#
if __name__ == '__main__':

    model = cnn()
    model.cuda()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_func = nn.CrossEntropyLoss().cuda()
    train_loader = load_data(pathlist[0], pathlist[1], batchsize=550)
    # writer=SummaryWriter()
    train_error = []
    test_error = []
    for epoch in range(500):
        train(epoch,train_loader, pathlist[2], pathlist[3],train_error,test_error)
