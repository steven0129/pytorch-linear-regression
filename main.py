import torch as t
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')

from matplotlib import pyplot as plt

# 隨機種子
t.manual_seed(1000)


def get_fake_data(batch_size=8):
    x = t.rand(batch_size, 1) * 20
    y = x * 2 + (1 + t.randn(batch_size, 1)) * 3
    return x, y


x, y = get_fake_data()
plt.scatter(x.squeeze().numpy(), y.squeeze().numpy())
plt.savefig('scatter.png')

# Init variable randomly
w = t.rand(1, 1)
b = t.zeros(1, 1)
lr = 0.001  # Learning rate

for ii in tqdm(range(20000)):
    x, y = get_fake_data()

    # forward
    y_pred = x.mm(w) + b.expand_as(y)
    loss = 0.5 * (y_pred - y) ** 2  # Mean Square Error
    loss = loss.sum()

    # backward
    dloss = 1
    dy_pred = dloss * (y_pred - y)

    dw = x.t().mm(dy_pred)
    db = dy_pred.sum()

    # update parameters
    w.sub_(lr * dw)
    b.sub_(lr * db)

    if ii%1000 ==0: # 畫圖
        plt.clf()
        x = t.arange(0, 20).view(-1, 1)
        y = x.mm(w) + b.expand_as(x)
        plt.plot(x.numpy(), y.numpy()) # predicted
        
        x2, y2 = get_fake_data(batch_size=20) 
        plt.scatter(x2.numpy(), y2.numpy()) # true data
        
        plt.xlim(0, 20)
        plt.ylim(0, 41)
        plt.savefig('regression-%d.png' % ii)

print(w.squeeze()[0], b.squeeze()[0])