from utils import *
from module import *
import torch
from torchvision.utils import save_image
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')

setSeed(2022)
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--data', type=str, default='')
args = parser.parse_args()
device = torch.device('cpu') if args.cuda < 0 else torch.device('cuda', args.cuda)

beta = 10**7

if args.data == 'color':
    # color
    train_data, test_data = loadMnist(color=True)
    epochs = 1000
    verbose = 100
    batch_size = 2048
    lr = 1e-3

    n_chan = 3
    z_dim = 10
    s_dim = 3

    model = FairDisCoImage(z_dim=z_dim, s_dim=s_dim, n_chan=n_chan)
    if os.path.exists('./model/FairDisCoImage_color.pkl'):
        model.load('./model/FairDisCoImage_color.pkl')
        model.eval()
    else:
        model.fit(train_data=train_data, epochs=epochs, lr=lr,  batch_size=batch_size, verbose=verbose, beta=beta, device=device)
        torch.save(model.state_dict(), './model/FairDisCoImage_color.pkl')

    imgs1 = test_data.X[:12*8]
    imgs2 = model.decode(model.encode(imgs1), test_data.S[:12*8])
    imgs3 = model.decode(model.encode(imgs1), torch.zeros(12*8).long())
    imgs4 = model.decode(model.encode(imgs1), torch.zeros(12*8).long()+1)
    imgs5 = model.decode(model.encode(imgs1), torch.zeros(12*8).long()+2)

    save_image(imgs1, './res/color1.pdf', format='pdf', nrow=12)
    save_image(imgs2, './res/color2.pdf', format='pdf', nrow=12)
    save_image(imgs3, './res/color3.pdf', format='pdf', nrow=12)
    save_image(imgs4, './res/color4.pdf', format='pdf', nrow=12)
    save_image(imgs5, './res/color5.pdf', format='pdf', nrow=12)

elif args.data == 'number':
    # number
    train_data, test_data = loadMnist(color=False)
    epochs = 1000
    verbose = 100
    batch_size = 2048
    lr = 1e-3

    n_chan = 1
    z_dim = 10
    s_dim = 10

    model = FairDisCoImage(z_dim=z_dim, s_dim=s_dim, n_chan=n_chan)
    if os.path.exists('./model/FairDisCoImage_number.pkl'):
        model.load('./model/FairDisCoImage_number.pkl')
        model.eval()
    else:
        model.fit(train_data=train_data, epochs=epochs, lr=lr,  batch_size=batch_size, verbose=verbose, beta=beta, device=device)
        torch.save(model.state_dict(), './model/FairDisCoImage_number.pkl')

    imgs1 = test_data.X[20:30]
    X = torch.cat([imgs1]*10, dim=0)
    X = X.view(10,10,1,64,64).transpose(0,1).contiguous().view(-1,1,64,64)
    S = torch.LongTensor([range(10)]*10).view(-1)
    imgs2 = model.decode(model.encode(X), S)

    save_image(imgs1, './res/number1.pdf', format='pdf', nrow=1)
    save_image(imgs2, './res/number2.pdf', format='pdf', nrow=10)
