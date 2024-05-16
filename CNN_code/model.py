# d2l version 0.17.6
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
#from torch.nn import functional as F
from torch.optim import lr_scheduler
from torchvision import models
import data
import sub_dataset

'''
# 定义卷积神经网络结构
def get_net(d):
    length, width = 0, 0
    if d == 1:
        length, width = 145, 121
    elif d == 2:
        length, width = 121, 121
    elif d == 3:
        length, width = 121, 145

    network = nn.Sequential(
        nn.Conv2d(1, 3, kernel_size=5, padding=2), nn.BatchNorm2d(3), nn.ReLU(),
        # nn.AvgPool2d(kernel_size=(5, 11), stride=(5, 11)),
        # nn.Conv2d(3, 6, kernel_size=3, padding=1), nn.Sigmoid(),
        nn.Flatten(),
        nn.Linear(3 * length * width, 960), nn.ReLU(),
        nn.Linear(960, 120), nn.ReLU(),
        nn.Linear(120, 2)
    )
    return network


net = nn.Sequential(nn.Conv2d(1, 16, kernel_size=7, padding=3), nn.BatchNorm2d(16), nn.ReLU(),
                    nn.AvgPool2d(kernel_size=3, stride=3),
                    #nn.Conv2d(8, 16, kernel_size=5, padding=2), nn.BatchNorm2d(16), nn.ReLU(),
                    #nn.AvgPool2d(kernel_size=4, stride=4),
                    nn.Conv2d(16, 64, kernel_size=5, padding=2), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=5, padding=2), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.AvgPool2d(kernel_size=4, stride=4),
                    nn.Conv2d(64, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                    #nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(256*10*10, 960), nn.ReLU(), nn.Dropout(0.25),
                    nn.Linear(960, 120), nn.ReLU(),  #nn.Dropout(0.1),
                    nn.Linear(120, 2))


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

Res_net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Dropout(0.5), nn.Linear(512, 2))
'''

'''
def evaluate_accuracy_gpu(network, data_iter, loss, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(network, nn.Module):
        network.eval()  # 设置为评估模式
        if not device:
            device = next(iter(network.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(3)
    with torch.no_grad():
        for X_, y_ in data_iter:
            X_ = X_.to(device)
            y_ = y_.to(device)
            metric.add(d2l.accuracy(network(X_), y_), loss(network(X_), y_), y_.numel())
    return metric[0] / metric[2], metric[1] / metric[2]
'''


# 衰减的学习率
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.9)


# 早停法
class EarlyStopping:

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_model(net_, train_iter_, test_iter_, num_epochs_, lr_, weight_decay_, device, scheduler=None):

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    early_stopping = EarlyStopping(patience=12, verbose=True)
    net_.apply(init_weights)
    print('training on', device)
    net_.to(device)
    optimizer = torch.optim.Adam(net_.parameters(), lr=lr_, weight_decay=weight_decay_)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = d2l.Timer(), len(train_iter_)
    for epoch in range(num_epochs_):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net_.train()
        for j, (X_, y_) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X_, y_ = X_.to(device), y_.to(device)
            y_hat = net_(X_)
            l = loss(y_hat, y_)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X_.shape[0], d2l.accuracy(y_hat, y_), X_.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        test_acc = d2l.evaluate_accuracy_gpu(net_, test_iter_)
        test_loss = 1 - test_acc
        early_stopping(test_loss, net_)
        if early_stopping.early_stop:
            print('early stopping')
            break

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                scheduler.step()
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scheduler(epoch)

        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs_ / timer.sum():.1f} examples/sec 'f'on {str(device)}')
    net_.load_state_dict(torch.load('checkpoint.pt'))
    return net_

# use resnet18 model by change the first layer and the last layer
resnet18 = models.resnet18()
resnet18.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 2))
resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# use resnet34 model by change the first layer and the last layer
resnet34 = models.resnet34()
resnet34.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet34.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 2))

num_test = 171


for p in range(3):
    num_epochs, lr, batch_size, weight_decay = 30, 0.001, 16, 1e-8
    # Scheduler = SquareRootScheduler(lr)
    Scheduler = None
    model = resnet34
    if p == 0:
        decision_matrix = torch.zeros([num_test, 121])
        for q in range(121):
            train_iter = data.get_train_iter(1, q, batch_size)
            test_iter = data.get_test_iter(1, q, batch_size)
            num_batchs = len(test_iter)
            net = train_model(model, train_iter, test_iter, num_epochs, lr, weight_decay, d2l.try_gpu(), Scheduler)
            for i, (X, y) in enumerate(test_iter):
                if i != num_batchs - 1:
                    decision_matrix[i * batch_size:(i + 1) * batch_size, q] = d2l.argmax(net(X), axis=1)
                elif i == num_batchs - 1:
                    decision_matrix[i * batch_size:num_test, q] = d2l.argmax(net(X), axis=1)
            print(p + 1, 'direction', q, 'position train down')
        np.savetxt('opt_Y1.csv', decision_matrix.numpy())
        print('opt_Y1.csv save down')
    elif p == 1:
        decision_matrix = torch.zeros([num_test, 145])
        for q in range(145):
            train_iter = data.get_train_iter(2, q, batch_size)
            test_iter = data.get_test_iter(2, q, batch_size)
            num_batchs = len(test_iter)
            net = train_model(model, train_iter, test_iter, num_epochs, lr, weight_decay, d2l.try_gpu(), Scheduler)
            for i, (X, y) in enumerate(test_iter):
                if i != num_batchs - 1:
                    decision_matrix[i * batch_size:(i + 1) * batch_size, q] = d2l.argmax(net(X), axis=1)
                elif i == num_batchs - 1:
                    decision_matrix[i * batch_size:num_test, q] = d2l.argmax(net(X), axis=1)
            print(p + 1, 'direction', q, 'position train down')
        np.savetxt('opt_Y2.csv', decision_matrix.numpy())
        print('opt_Y2.csv save down')
    elif p == 2:
        decision_matrix = torch.zeros([num_test, 121])
        for q in range(121):
            train_iter = data.get_train_iter(3, q, batch_size)
            test_iter = data.get_test_iter(3, q, batch_size)
            num_batchs = len(test_iter)
            net = train_model(model, train_iter, test_iter, num_epochs, lr, weight_decay, d2l.try_gpu(), Scheduler)
            for i, (X, y) in enumerate(test_iter):
                if i != num_batchs - 1:
                    decision_matrix[i * batch_size:(i + 1) * batch_size, q] = d2l.argmax(net(X), axis=1)
                elif i == num_batchs - 1:
                    decision_matrix[i * batch_size:num_test, q] = d2l.argmax(net(X), axis=1)
            print(p + 1, 'direction', q, 'position train down')
        np.savetxt('opt_Y3.csv', decision_matrix.numpy())
        print('opt_Y3.csv save down')
