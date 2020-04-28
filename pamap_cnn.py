import os
import time
import argparse
import torch
import math
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import init
import pandas as pd
import sklearn.metrics as sm
from torch.autograd import Variable
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,confusion_matrix
import numpy as np
import tqdm as tqdm
import torch.nn.functional as F
import  torchsnooper as torchsnooper
import torch.optim as optim
from torchvision import datasets, transforms
from torch.backends import cudnn
from bisect import bisect_right
import torch.utils.data as Data
from tqdm import tqdm
import os

torch.cuda.set_device(0)
n_gpu = torch.cuda.device_count()
print(n_gpu)
path=os.path.dirname(os.path.abspath("__file__"))
print(path)
# @torchsnooper.snoop()

pathlist = [r'./train_X_new.npy',
            r'./train_y_new.npy',
            r'./test_X.npy',
            r'./test_Y.npy']


# # @torchsnooper.snoop()
def data_flat(data_y):
    data_y=np.argmax(data_y, axis=1)
    return data_y

def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)
    plt.yticks(num_local, labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def load_data(path_X,path_y,batchsize):
    train_x=np.load(path_X)
    train_x_shape = train_x.shape
    train_x = torch.from_numpy(
        np.reshape(train_x.astype(float), [train_x_shape[0], 1, train_x_shape[1], train_x_shape[2]])).cuda()


    train_y = np.load(path_y)
    train_y = torch.from_numpy(train_y).cuda()

    torch_dataset=Data.TensorDataset(train_x,train_y)
    loader=Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0,
    )
    total=len(loader)
    # for _ in tqdm(range(total), desc='ongoing', ncols=80,postfix="train_data"):
    #     pass
    return loader


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

class conv_loss_block(nn.Module):
    def __init__(self, channel_in, channel_out, height_width, kernel, stride, bias,numlayer):
        super(conv_loss_block, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.height_width = height_width
        self.num_class = 12
        self.bias = bias
        self.dropout_p = 0.5
        self.batchnorm = True
        self.decode_ys=[]
        self.bns_decode_ys = []

        decode_t_list = [279552, 165376, 56832]
        # print(int(channel_out*height_width*0.5),'self.biasself.biasself.biasself.biasself.bias')
        print(channel_in, channel_out, height_width, kernel, stride, bias,'channel_in, channel_out, height_width, kernel, stride, bias')

        self.relu=nn.ReLU(inplace=True)
        self.encoder = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, (6,2), stride=stride, padding=0, bias=self.bias),
            nn.BatchNorm2d(channel_out),
            nn.LeakyReLU(inplace=True),

        )


        for i in range(3):
            decode_y = nn.Linear(decode_t_list[i], 12)
            setattr(self, 'decode_y%i' % i, decode_y)
            self._set_init(decode_y)
            self.decode_ys.append(decode_y)



        self.conv_loss = nn.Sequential(
            nn.Conv2d(channel_out, channel_out, kernel_size=(2,1), stride=(2,1), padding=(1,0), bias=False),
            # nn.BatchNorm2d(channel_out,momentum=0.5)
                                       )


        if self.batchnorm:
            self.bn = torch.nn.BatchNorm2d(channel_out,momentum=0.5)
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

        self.nonlin = nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(p=self.dropout_p, inplace=False)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)



        self.clear_stats()
    def _set_init(self, layer):
        init.normal_(layer.weight, mean=0., std=.1)
        init.constant_(layer.bias, 0.2)

    def clear_stats(self):
        self.loss_sim = 0.0
        self.loss_pred = 0.0
        self.correct = 0
        self.examples = 0

    def print_stats(self):
        stats = '{}, loss_sim={:.4f}, loss_pred={:.4f}, error={:.3f}%, num_examples={}\n'.format(
            self.encoder,
            self.loss_sim / self.examples,
            self.loss_pred / self.examples,
            100.0 * float(self.examples - self.correct) / self.examples,
            self.examples)
        return stats

    def set_learning_rate(self, lr):
        self.lr = lr
        # print('lr:', self.optimizer.param_groups[0]['lr'])
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def optim_zero_grad(self):
        self.optimizer.zero_grad()

    def optim_step(self):
        self.optimizer.step()

    def forward(self, x, y, y_onehot,loop,is_training):

        h = self.encoder(x)

        h_return = h
        h_return = self.dropout(h_return)


        h_loss = self.conv_loss(h)
        Rh = similarity_matrix(h_loss)

        #caculate unsupervised loss
        Rx=similarity_matrix(x).detach()
        loss_unsup=F.mse_loss(Rh,Rx)

        h_pool = h_return
        # print(h_pool.shape,'h_pool.view(h_pool.size(0)')

        y_hat_local = self.decode_ys[loop](h_pool.view(h_pool.size(0), -1))
        # print(y_hat_local.shape, y.shape,y_onehot.shape,'y_hat_local.shape, y.detach().shape')
        loss_pred = (1 - 0.99) * F.cross_entropy(y_hat_local, y.detach().long())

        Ry = similarity_matrix(y_onehot).detach()
        # print(Rh.dtype,Ry.dtype,'Rh.dtype,Ry.dtype')
        loss_sim = 0.99 * F.mse_loss(Rh, Ry)
        # print(loss_sim,'loss_simloss_simloss_simloss_simloss_simloss_simloss_simloss_sim')
        loss_sup = loss_pred+loss_sim

        loss = loss_sup*1+loss_unsup*0

        if is_training:
            loss.backward(retain_graph=False)
        if is_training:
            self.optimizer.step()
            self.optimizer.zero_grad()
            h_return.detach_()
        loss = loss.item()

        return h_return, loss


class convnet(nn.Module):
    def __init__(self, input_ch, output_ch, height, num_layers, num_hiden, num_classes,lr):
        super(convnet, self).__init__()
        self.num_hidden = num_hiden
        self.num_layers = num_layers
        self.height = height
        self.input_ch = input_ch
        self.output_ch = output_ch
        reduce_factor = 1
        self.bn=[]


        self.layers = nn.ModuleList(
            [conv_loss_block(self.input_ch, self.output_ch, self.height, kernel=(6,2), stride=(3,1), bias=False,numlayer=num_layers)])
        self.layers.extend(
            [conv_loss_block(128, 256, self.height, kernel=(6,2), stride=(3,1), bias=False,numlayer=num_layers),
             conv_loss_block(256, 384, self.height, kernel=(6,2), stride=(3,1), bias=False,numlayer=num_layers)
             ])

        self.layer_out = nn.Linear(56832, num_classes)
        self.layer_out.weight.data.zero_()

        bn = nn.BatchNorm2d(1, momentum=0.5)
        setattr(self, 'pre_bn' , bn)
        self.bn.append(bn)
        
    def parameters(self):
        return self.layer_out.parameters()

    def set_learning_rate(self, lr):
        for i, layer in enumerate(self.layers):
            layer.set_learning_rate(lr)

    def optim_step(self):
        for i, layer in enumerate(self.layers):
            layer.optim_step()

    def optim_zero_grad(self):
        for i, layer in enumerate(self.layers):
            layer.optim_zero_grad()

    def forward(self, x, y, y_onehot,is_training):

        total_loss = 0.0
        for i, layer in enumerate(self.layers):
            if i==0:
                x=x.float()
                x=self.bn[i](x)


            x, loss = layer(x, y, y_onehot,i,is_training)
            total_loss += loss


        x= x.contiguous().view(x.size(0), -1)
        x = self.layer_out(x)
        return x, total_loss


def to_one_hot(y, n_dims=None):
    ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

def plot_confusion(comfusion,class_data):
    plt.figure(figsize=(12,9))
    classes = class_data
    plt.imshow(comfusion, interpolation='nearest', cmap=plt.cm.Oranges) 
    plt.title('confusion_matrix', fontsize=12)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }
    plt.xticks(tick_marks, classes,rotation=315)
    plt.yticks(tick_marks, classes)
    plt.tick_params(labelsize=12)
    thresh = comfusion.max() / 2.
    # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]

    iters = np.reshape([[[i, j] for j in range(len(classes))] for i in range(len(classes))], (comfusion.size, 2))
    for i, j in iters:
        plt.text(j, i, format(comfusion[i, j]))  

    plt.ylabel('Real label',fontsize = 12)
    plt.xlabel('Prediction',fontsize = 12)
    plt.tight_layout()
    plt.savefig('./pamap2_confusion.eps')
    plt.show()

# input_ch,output_ch,height,num_layers,num_hiden,num_classes

def train(train_loader, test_x_path, test_y_path,test_error):

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Total_Number of params: {} |Trainable_num of params: {}'.format(total_num, trainable_num))
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        target_onehot = to_one_hot(batch_y)
        target_onehot = target_onehot.cuda()
        # print(batch_x.shape,target_onehot.shape,batch_y.shape,'batch_x,target_onehot.shape,batch_y.shape')

        # check_parameters(model,2)
        optimizer.zero_grad()
        output, _ = model(batch_x, batch_y, target_onehot, True)
        # print(output.shape,batch_y.shape,target_onehot.shape,'output.shape')

        loss = l(output, batch_y.long())

        # check_parameters(model, 2)

        # loss_t.backward()
        loss.backward()
        optimizer.step()

            
        # taccuracy = (torch.sum(train_output == batch_y.long()).type(torch.FloatTensor) / batch_y.size(0)).cuda()
        # print(taccuracy,'train_accuracy')
    if epoch % 1 == 0:
        model.eval()
        #
        test_x = np.load(test_x_path)
        test_x_shape = test_x.shape
        test_x = torch.from_numpy(np.reshape(test_x, [test_x_shape[0], 1, test_x_shape[1], test_x_shape[2]])).cuda()

        test_y = np.load(pathlist[3])
        test_y = torch.from_numpy(test_y).cuda()

        test_y_onehot = to_one_hot(test_y)
        test_y_onehot = test_y_onehot.cuda()

        # print(test_x.shape, test_y.shape, test_y_onehot.shape, 'test_x.shape,test_y.shape,target_y.shape')
        try:
            test_output,_=model(test_x,test_y,test_y_onehot,False )
            test_output_copy = test_output
            # print(test_output.shape)
            test_output = data_flat(test_output.cpu().detach().numpy())
            # print(test_output.shape, 'test_output')

            test_output_f1 = np.asarray(pd.get_dummies(test_output))

            print(test_y_onehot.shape, test_output_f1.shape)
            acc = accuracy_score(test_y_onehot.cpu().numpy(), test_output_f1)
            f1 = f1_score(test_y_onehot.cpu().numpy(), test_output_f1, average='weighted')
            f2 = f1_score(test_y_onehot.cpu().numpy(), test_output_f1, average='micro')
            f3 = f1_score(test_y_onehot.cpu().numpy(), test_output_f1, average='macro')
            reca = recall_score(test_y.cpu().numpy(), test_output, average='weighted')
            # if epoch%2==0:
            #     confusion = sm.confusion_matrix(test_y.cpu().numpy(), test_output)
            #     print('The confusion matrix is：', confusion, sep='\n')
            #     plot_confusion(confusion,
            #                    ['Lying', 'Sitting', 'Standing', 'Walking', 'Running', 'Cycling', 'Nordic walking',
            #                     'Ascending stairs', 'Descending stairs', 'Vacuum cleaning', 'Ironing', 'Rope jumping'])
            #
            # # print(confusion_y.tolist(), '\n', test_output_f1_con.tolist())
            print('Epoch: ', epoch, '| test accuracy: %.8f' % acc, '| test F1: %.8f' % f1, '| test recall: %.8f' % reca,
                  '| test micro: %.8f' % f2, '| test micro: %.8f' % f3)
            #
            test_output = torch.max(test_output_copy, 1)[1].cuda()
            accuracy = (torch.sum(test_output == test_y.long()).type(torch.FloatTensor) / test_y.size(0)).cuda()
            # print('Epoch: ', epoch, '| test accuracy: %.8f' % accuracy)
            # test_error.append((1 - accuracy.item()))

            model.train()
            
        except ValueError:
            test_output = torch.max(test_output_copy, 1)[1].cuda()
            accuracy = (torch.sum(test_output == test_y.long()).type(torch.FloatTensor) / test_y.size(0)).cuda()
            print('Epoch: ', epoch, '| test accuracy: %.8f' % accuracy)

            # confusion = sm.confusion_matrix(test_y.cpu().numpy(), test_y.cpu().numpy())
            # print('The confusion matrix is：', confusion, sep='\n')

        else:
            pass
if __name__ == '__main__':

    model = convnet(input_ch=1, output_ch=64 * 2, height=128 * 2, num_layers=3, num_hiden=100, num_classes=12,lr=1e-4)
    model.cuda()
    print(model)
    # optimizer = torch.optim.SGD(model.parameters(), lr=5e-4,momentum=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    l=nn.CrossEntropyLoss().cuda()

    train_loader = load_data(pathlist[0], pathlist[1], batchsize=200)
    test_error = []
    lr = [0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005]

    for epoch in range(500):
        if epoch <= 50:
            lr_dynamic = lr[0]
            model.set_learning_rate(lr_dynamic)
        elif 51 <= epoch <= 100:
            lr_dynamic = lr[1]
            model.set_learning_rate(lr_dynamic)
        elif 101 <= epoch <= 150:
            lr_dynamic = lr[2]
            model.set_learning_rate(lr_dynamic)
        elif 151 <= epoch <= 200:
            lr_dynamic = lr[3]
            model.set_learning_rate(lr_dynamic)
        elif 201 <= epoch <= 300:
            lr_dynamic = lr[4]
            model.set_learning_rate(lr_dynamic)
        elif 301 <= epoch :
            lr_dynamic = lr[5]
            model.set_learning_rate(lr_dynamic)
        train(train_loader, pathlist[2], pathlist[3],test_error)
