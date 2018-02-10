import torch as t
from torch.autograd import Variable as V
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import prettytable as pt

# CUDA加速
CUDA = t.cuda.is_available()

# 设置随机数种子，为了在不同人电脑上运行时下面的输出一致
t.manual_seed(1000)


def get_fake_data(batch_size=8):
    ''' 产生随机数据：y = x*2 + 3，加上了一些噪声'''
    x = t.rand(batch_size, 1) * 20
    y = x * 2 + (1 + t.randn(batch_size, 1)) * 3
    return x, y


x, y = get_fake_data()
plt.scatter(x.squeeze().numpy(), y.squeeze().numpy())
plt.savefig('scatter.png')

# Init variable randomly
w = V(t.rand(1, 1), requires_grad=True)
b = V(t.zeros(1, 1), requires_grad=True)
lr = 0.001  # Learning rate

for ii in tqdm(range(20000)):
    x, y = get_fake_data()
    x, y = V(x), V(y)

    # forward
    if CUDA:
        y_pred = x.cuda().mm(w.cuda()) + b.cuda().expand_as(y.cuda())
        loss = 0.5 * (y_pred - y.cuda()) ** 2  # Mean Square Error
    else:
        y_pred = x.mm(w) + b.expand_as(y)
        loss = 0.5 * (y_pred - y) ** 2  # Mean Square Error

    loss = loss.sum()

    # backward
    loss.backward()

    # update parameters
    w.data.sub_(lr * w.grad.data)
    b.data.sub_(lr * b.grad.data)

    # clean gradients
    w.grad.data.zero_()
    b.grad.data.zero_()

    if ii % 1000 == 0:  # 畫圖
        plt.clf()
        x = t.arange(0, 20).view(-1, 1)
        y = x.mm(w.data) + b.data.expand_as(x)
        plt.plot(x.numpy(), y.numpy())  # predicted

        x2, y2 = get_fake_data(batch_size=20)
        plt.scatter(x2.numpy(), y2.numpy())  # true data

        plt.xlim(0, 20)
        plt.ylim(0, 41)
        plt.savefig('regression-%d.png' % ii)

table = pt.PrettyTable()
table.field_names = ['Weights', 'Bias']
table.add_row([w.data.squeeze()[0], b.data.squeeze()[0]])
print(table)
