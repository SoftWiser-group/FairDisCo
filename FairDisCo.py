from utils import *
from module import *
import torch
import numpy as np
import argparse
import warnings
import os
warnings.filterwarnings('ignore')

setSeed(2022)
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=-1)
parser.add_argument('--data', type=str, default='')
args = parser.parse_args()
device = torch.device('cpu') if args.cuda < 0 else torch.device('cuda', args.cuda)

dataset = args.data

train_data, test_data, D = load_dataset(dataset)
S_train, S_test = train_data.S.numpy(), test_data.S.numpy()
Y_train, Y_test = train_data.Y.numpy(), test_data.Y.numpy()

batch_size = 2048
epochs = 1000
verbose = 100

lr = 1e-3
x_dim = train_data.X.shape[1]
s_dim = train_data.S.max().item()+1
h_dim = 64
z_dim = 8

logs = []
for lg_beta in range(11):
    beta = 10 ** lg_beta
    model = FairDisCo(x_dim, h_dim, z_dim, s_dim, D)

    if os.path.exists('./model/FairDisCo_{}_{}.pkl'.format(dataset, lg_beta)):
        model.load('./model/FairDisCo_{}_{}.pkl'.format(dataset, lg_beta))
        model.eval()
    else:
        model.fit(train_data=train_data, epochs=epochs, lr=lr, batch_size=batch_size, verbose=verbose, beta=beta, device=device)
        torch.save(model.state_dict(), './model/FairDisCo_{}_{}.pkl'.format(dataset, lg_beta))

    with torch.no_grad():
        n_iter = 10
        eval_res = np.zeros(len(eval_col_name))

        for _ in range(n_iter):
            Z_train = model.encode(train_data.X, train_data.S).numpy()
            Z_test = model.encode(test_data.X, test_data.S).numpy()
            eval_res += evaluate(Z_train, Z_test, S_train, S_test, Y_train, Y_test) / n_iter

        dis = model.calculate_dis(test_data.S, test_data.S.shape[0]).item() * beta
        logs.append([lg_beta, dis] + list(eval_res))

df = pd.DataFrame(logs, columns=['log_beta', 'dis'] + eval_col_name)
df.to_csv('./res/{}/FairDisCo.csv'.format(dataset), index=False)

print('FairDisCo {} finish'.format(dataset))
