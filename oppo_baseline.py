
import torch
import math
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn import init
import tqdm as tqdm
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,confusion_matrix
import os
torch.cuda.set_device(0)
n_gpu = torch.cuda.device_count()
print(n_gpu)
path=os.path.dirname(os.path.abspath("__file__"))
print(path)

# @torchsnooper.snoop()
def data_flat(data_y):
    data_y=np.argmax(data_y, axis=1)
    return data_y

path = os.path.dirname(os.path.dirname(__file__))
# print(path)

# model name 'baseline_cnn' never use
# model name 'baseline_cnn_local_UCI'  it is my method using UCI dataset.
# model name 'DCNN_UCI' the model is baseline using UCI dataset.
# model name 'CONVLSTM_UCI' the model is baseline using UCI dataset

# model name 'baseline_cnn_local_oppo'
# model name 'DCNN_oppo'
# model name 'CONVLSTM_oppo' it is my method using oppoyunity dataset.

# model name 'CONVLSTM_UNIMIB' the model is baseline using UCI dataset

model='baseline_cnn_local_UCI'



pathlist=['./oppotunity_sum/oppo_kun_norm_slidewindow8/data_train_one.npy',
                './oppotunity_sum/oppo_kun_norm_slidewindow8/label_train_onehot.npy',
                './oppotunity_sum/oppo_kun_norm_slidewindow8/data_test_one.npy',
                './oppotunity_sum/oppo_kun_norm_slidewindow8/label_test_onehot.npy']




def load_data(train_x_path, train_y_path, batchsize):
    train_x = np.load(train_x_path)
    train_x_shape = train_x.shape
    train_x = torch.from_numpy(
        np.reshape(train_x.astype(float), [train_x_shape[0], train_x_shape[1], train_x_shape[2],1])).cuda()

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
    )
    total = len(loader)
    # for _ in tqdm(range(total), desc='进行中', ncols=80,postfix="train_data"):
    #     pass
    return loader







def similarity_matrix(x):
    ''' Calculate adjusted cosine similarity matrix of size x.size(0) x x.size(0). '''
    if x.dim() == 4:
        if x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
        else:
            x = x.view(x.size(0), -1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc ** 2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1, 0)).clamp(-1, 1)
    return R
def quzheng_x(height, kernel_size, padding, stride, numlayer):
    list = []
    for i in range(1, numlayer + 1):
        feature = int((height - kernel_size + 2 * padding) / stride) + 1
        height = feature
        list.append(feature)
    return list


def quzheng_s(height, kernel_size, padding, stride, numlayer):
    list = []
    for i in range(1, numlayer + 1):
        feature = math.ceil((height - kernel_size + 2 * padding) / stride) + 1
        height = feature
        list.append(feature)
    return list


def check_parameters(defined_model, parameters_index):
    params = list(defined_model.named_parameters())
    (name, param) = params[parameters_index]
    print('___________________________________________________________________\n', name, param,
          '\n____________________________________________________________________')
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()

        # print(channel_in, channel_out,  kernel, stride, bias,'channel_in, channel_out, kernel, stride, bias')

        self.encoder1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3),padding=1,stride=(2,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

)


        self.encoder2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3),padding=1,stride=(2,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
)




        self.encoder3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=(3,3),padding=1,stride=(2,1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

)





        self.linear = nn.Linear(5376, 17)
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

def train(epoch,train_loader, test_x_path, test_y_path,test_error,train_error):
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
        output = model(batch_x)
        # print(output.shape,batch_y.shape,target_onehot.shape,'output.shape')

        loss = loss_func(output, batch_y.long())

        # check_parameters(model, 2)

        # loss_t.backward()
        loss.backward()
        optimizer.step()

    # train_output = torch.max(output, 1)[1].cuda()
    # taccuracy = (torch.sum(train_output == batch_y.long()).type(torch.FloatTensor) / batch_y.size(0)).cuda()
    # error = 1 - taccuracy.item()
    # train_error.append(error)
    # print('EPOCH', epoch, 'train accuracy', taccuracy.item())
    #
    # np.save('./matplotlib_picture/OPPO_error/bp_train_batchsize300.npy', train_error)
    if epoch % 1 == 0:
        model.eval()

        test_x = np.load(test_x_path)
        test_x_shape = test_x.shape
        test_x = torch.from_numpy(np.reshape(test_x, [test_x_shape[0], test_x_shape[1], test_x_shape[2],1])).cuda()

        test_y = data_flat(np.load(pathlist[3]))
        test_y = torch.from_numpy(test_y).cuda()

        test_y_onehot = to_one_hot(test_y)
        test_y_onehot = test_y_onehot.cuda()

        # print(test_x.shape, test_y.shape, test_y_onehot.shape, 'test_x.shape,test_y.shape,target_y.shape')

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
        print('Epoch: ', epoch, '| test accuracy: %.8f' % accuracy)
        test_error.append((1 - accuracy.item()))
    # np.save('./matplotlib_picture/OPPO_error/bp_test2.npy',test_error)

if __name__ == '__main__':

    model = cnn()
    model.cuda()
    print(model)
    train_error = []
    test_error = []
    optimizer = torch.optim.RMSprop(model.parameters(),lr=0.001)
    loss_func = nn.CrossEntropyLoss().cuda()
    train_loader = load_data(pathlist[0], pathlist[1], batchsize=200)

    for epoch in range(300):
        train(epoch,train_loader, pathlist[2], pathlist[3],test_error,train_error)

