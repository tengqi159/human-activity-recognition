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
# import tqdm as tqdm
import torch.nn.functional as F
# import  torchsnooper as torchsnooper
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,confusion_matrix
# from torchvision import datasets, transforms
from torch.backends import cudnn
from bisect import bisect_right
import torch.utils.data as Data
# from tqdm import tqdm
import os
torch.cuda.set_device(0)
n_gpu = torch.cuda.device_count()
print(n_gpu)
path=os.path.dirname(os.path.abspath("__file__"))
print(path)
##################oppotunity_dataset_slide-windows with 8#########
# shape (25343, 64, 107) (25343,17) (5313, 64, 107) (5313,17)

train_x_path='./data_train_one.npy'
train_y_path_onehot='./label_train_onehot.npy'
test_x_path='./data_test_one.npy'
test_y_path_onehot='./label_test_onehot.npy'


# # @torchsnooper.snoop()
def data_flat(data_y):
    data_y=np.argmax(data_y, axis=1)
    return data_y


def load_data():
    # path = str(os.path.dirname(os.path.abspath("__file__")))
    train_x = np.load(train_x_path)
    train_x=torch.from_numpy(np.reshape(train_x,[25343,64,107,1])).cuda()

    train_y=data_flat(np.load(train_y_path_onehot))
    train_y=torch.from_numpy(train_y).cuda()
    # print(train_x.shape,train_y.shape,'ppppppppppppppppppp')
    torch_dataset=Data.TensorDataset(train_x,train_y)
    loader=Data.DataLoader(
        dataset=torch_dataset,
        batch_size=200,
        shuffle=True,
        num_workers=0,
    )
    total=len(loader)
    # for _ in tqdm(range(total), desc='ongoing', ncols=80,postfix="train_data"):
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

class conv_loss_block(nn.Module):
    def __init__(self, channel_in, channel_out, height_width, kernel, stride, bias,numlayer,lr):
        super(conv_loss_block, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.height_width = height_width
        self.num_class = 17
        self.bias = bias
        self.dropout_p = 0.5
        self.batchnorm = True
        self.decode_ys=[]
        self.bns_decode_ys = []

        decode_t_list=[6912,6912,5376,3072]

        #print(channel_in, channel_out, height_width, kernel, stride, bias,'channel_in, channel_out, height_width, kernel, stride, bias')

        self.relu=nn.ReLU(inplace=True)
        self.encoder = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, 3, stride=stride, padding=1, bias=self.bias),
            nn.BatchNorm2d(channel_out),
            nn.LeakyReLU(inplace=True),
        )

        for i in range(numlayer):
            print(quzheng_x(53, 2, 1, 2, numlayer)[i] * 128 * (i+1))
            decode_y = nn.Linear(decode_t_list[i], 17)
            setattr(self, 'decode_y%i' % i, decode_y)
            self._set_init(decode_y)
            self.decode_ys.append(decode_y)


        self.conv_loss = nn.Sequential(
            nn.Conv2d(channel_out, channel_out, kernel_size=(2,1), stride=(2,1), padding=(1,0), bias=False),
            nn.BatchNorm2d(channel_out,momentum=0.5)
                                       )


        if self.batchnorm:
            self.bn = torch.nn.BatchNorm2d(channel_out,momentum=0.5)
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

        self.nonlin = nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(p=self.dropout_p, inplace=False)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.clear_stats()
    def _set_init(self, layer):
        init.normal_(layer.weight, mean=0., std=.1)
        init.constant_(layer.bias, 0.2)
        
    def clear_stats(self):
        self.loss_sim = 0.0
        self.loss_pred = 0.0
        self.correct = 0
        self.examples = 0

    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def optim_zero_grad(self):
        self.optimizer.zero_grad()

    def optim_step(self):
        self.optimizer.step()

    def forward(self, x, y, y_onehot,loop,is_training):
        h = self.encoder(x)
        h_return = h
        h_shape=h.shape
        
        h_return = self.dropout(h_return)

        h_loss = self.conv_loss(h)
        Rh = similarity_matrix(h_loss)

        #caculate unsupervised loss
        Rx=similarity_matrix(x).detach()
        loss_unsup=F.mse_loss(Rh,Rx)



        # h_pool = self.avg_pool(h)
        h_pool = h
        # print(h_pool.shape,'h_pool.view(h_pool.size(0)')

        y_hat_local = self.decode_ys[loop](h_pool.view(h_pool.size(0), -1))
        # print(y_hat_local.shape, y.shape,y_onehot.shape,'y_hat_local.shape, y.detach().shape')

        loss_pred = (1 - 0.99) * F.cross_entropy(y_hat_local, y.detach().long())

        Ry = similarity_matrix(y_onehot).detach()

        loss_sim = 0.99 * F.mse_loss(Rh, Ry)
        # print(loss_sim,'loss_simloss_simloss_simloss_simloss_simloss_simloss_simloss_sim')
        loss_sup = loss_pred+loss_sim

        self.loss_pred += loss_pred.item() * h_pool.size(0)
        self.loss_sim += loss_sim.item() * h_pool.size(0)
        self.correct += y_hat_local.max(1)[1].eq(y.long()).cpu().sum()
        self.examples += h.size(0)

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
            [conv_loss_block(self.input_ch, self.output_ch, self.height, kernel=(3,1), stride=(2,1), bias=False,numlayer=num_layers,lr=lr)])
        self.layers.extend(
            [conv_loss_block(output_ch*i, output_ch * (i + 1), self.height, kernel=(3,1), stride=(2,1), bias=False,numlayer=num_layers,lr=lr) for i in range(1, num_layers)])

        
        self.layer_out = nn.Linear(5376, num_classes)
        self.layer_out.weight.data.zero_()


        bn = nn.BatchNorm2d(64)
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
        # x = x.view(x.size(0), -1)

        total_loss = 0.0
        for i, layer in enumerate(self.layers):
            if i==0:
                x = x.type(torch.cuda.FloatTensor)
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


def train(train_loader,train_error,test_error):
    model.train()

    for m in model.modules():
        if isinstance(m,conv_loss_block):
            m.clear_stats()
            
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Total_Number of params: {} |Trainable_num of params: {}'.format(total_num, trainable_num))
    
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        target_onehot = to_one_hot(batch_y)
        target_onehot = target_onehot.cuda()

        # print( batch_y.shape, target_onehot.shape, 'output.shape,batch_y.shape,target_onehot.shape')
        optimizer.zero_grad()
        model.optim_zero_grad()

        output, loss_local = model(batch_x, batch_y, target_onehot,True)


        loss_global = l(output, batch_y.long())
        loss_t = loss_global *1 + loss_local * 0
        loss_t *= (1 - 0.99)

        loss_t.backward(retain_graph=False)
        optimizer.step()
        
#     train_output = torch.max(output, 1)[1].cuda()
#     taccuracy = (torch.sum(train_output == batch_y.long()).type(torch.FloatTensor) / batch_y.size(0)).cuda()
#     error = 1 - taccuracy.item()
#     train_error.append(error)
#     print('EPOCH', epoch, 'train accuracy', taccuracy.item())

    # np.save('./matplotlib_picture/OPPO_error/predsim_train_batchsize1000.npy', train_error)
    if epoch % 1 == 0:
        model.eval()

        test_x = np.load(test_x_path)
        test_x_shape = test_x.shape
        test_x = torch.from_numpy(np.reshape(test_x, [test_x_shape[0], test_x_shape[1], test_x_shape[2],1])).cuda()

        test_y = data_flat(np.load(test_y_path_onehot))
        test_y = torch.from_numpy(test_y).cuda()

        test_y_onehot = to_one_hot(test_y)
        test_y_onehot = test_y_onehot.cuda()

        
        print(test_x.shape, test_y.shape, test_y_onehot.shape, 'test_x.shape,test_y.shape,target_y.shape')
        try:
            test_output, _ = model(test_x, test_y, test_y_onehot, False)
            test_output_copy=test_output
            print(test_output.shape)
            test_output=data_flat(test_output.cpu().detach().numpy())
            print(test_output.shape, 'test_output')

            test_output_f1 =np.asarray(pd.get_dummies(test_output))

            print(test_y_onehot.shape, test_output_f1.shape)
            acc = accuracy_score(test_y_onehot.cpu().numpy(), test_output_f1)
            f1 = f1_score(test_y_onehot.cpu().numpy(), test_output_f1, average='weighted')
            reca = recall_score(test_y_onehot.cpu().numpy(), test_output_f1,average='weighted')
            # print(confusion_y.tolist(), '\n', test_output_f1_con.tolist())
            print('Epoch: ', epoch, '| test accuracy: %.8f' % acc, '| test F1: %.8f' % f1, '| test recall: %.8f' % reca)
        except ValueError:
            print('error')
        else:
            pass
        test_output = torch.max(test_output_copy, 1)[1].cuda()
        # print(test_output.shape,'test_output.shape')
        accuracy = (torch.sum(test_output == test_y.long()).type(torch.FloatTensor) / test_y.size(0)).cuda()
        print('Epoch: ', epoch, '| test accuracy: %.8f' % accuracy)
        test_error.append((1 - accuracy.item()))
    # np.save('./matplotlib_picture/OPPO_error/sim_test.npy',test_error)
    
if __name__ == '__main__':
    model = convnet(64, 128, 128 * 2,3, 100, 17,0.01)
    model.cuda()
    print(model)
    train_error = []
    test_error = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    l=nn.CrossEntropyLoss().cuda()

    model.set_learning_rate(0.001)
    train_loader = load_data()

    for epoch in range(300):
        train(train_loader,train_error,test_error)

