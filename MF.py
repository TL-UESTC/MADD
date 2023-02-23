import numpy as np
import pandas as pd
import sys
import pymongo
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description='Matrix Factorization')
parser.add_argument('--domain', action='append', help='domain names needed MF')
parser.add_argument('--size', required=True, help='embedding size after MF', type=int)
parser.add_argument('--GPU', default='-1', help='CUDA device -1,0,1,2', type=int)
parser.add_argument('--batchsize', default=512, type=int)
parser.add_argument('--lr', default=0.001, help='learning rate', type=float)
parser.add_argument('--weight_decay', default=0.1, help='balancing factor in optimizer', type=float)
parser.add_argument('--Epoch', default=500, type=int)
parser.add_argument('--save', action='store_true', help='whether to save the pth file after model training')
args = parser.parse_args()

def triTupleGen(domain, label):
    client = pymongo.MongoClient('localhost', 27017)
    db = client['MADD']
    Collection = db[domain + 's']
    data = [[result[label[0]], result[label[1]], result[label[2]]] for result in Collection.find()]
    data = sorted(data, key=lambda x: (x[0], x[1]))
    return pd.DataFrame(data, columns=label)

class MFDataset(Dataset):
    def __init__(self, u_id, i_id, rating):
        self.u_id = u_id
        self.i_id = i_id
        self.rating = rating

    def __getitem__(self, index):
        return self.u_id[index], self.i_id[index], self.rating[index]

    def __len__(self):
        return len(self.rating)

class MatrixModel(nn.Module):
    def __init__(self, num_users, num_items, mean, embedding_size):
        super(MatrixModel, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)

        self.user_emb.weight.data.uniform_(0, 0.005)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(0, 0.005)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.mean = nn.Parameter(torch.FloatTensor([mean]), False)

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        return (U * I).sum(1) + b_u + b_i + self.mean

def train(model, x_train, y_train, loss_func):
    train_dataset = MFDataset(x_train[:, 0], x_train[:, 1], y_train)
    train_iter = DataLoader(train_dataset, batch_size=args.batchsize)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.float()
    for epoch in range(args.Epoch):
        model.train()
        total_loss, total_len = 0.0, 0
        for x_u, x_i, y in train_iter:
            if args.GPU != -1:
                x_u, x_i, y = x_u.cuda(args.GPU), x_i.cuda(args.GPU), y.cuda(args.GPU)
            y_pred = model(x_u, x_i)
            l = loss_func(y_pred, y).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            total_loss += l.item()
            total_len += len(y)
        sys.stdout.write('\rMF Train: {} / {}, loss={:.4f}'.format(epoch+1, args.Epoch, total_loss / total_len))
        sys.stdout.flush()
        if args.save and epoch == args.Epoch-1:
            np.save("./matrixdata/{}sMatrix_UsersEmbedding_k{}.npy".format(domain, args.size), np.array(model.user_emb.weight.data.cpu()))
            np.save("./matrixdata/{}sMatrix_{}sEmbedding_k{}.npy".format(domain, domain.title(), args.size), np.array(model.item_emb.weight.data.cpu()))

if __name__ == '__main__':
    labels = {'book' : ['uID', 'bID', 'rating'],
              'movie' : ['uID', 'mID', 'rating'],
              'music' : ['uID', 'muID', 'rating']}

    for domain in args.domain:
        df = triTupleGen(domain, labels[domain])
        xb, yb = df.iloc[:,:2], df.iloc[:, 2]
        xb = torch.tensor(xb.values, dtype=torch.int32)
        yb = torch.tensor(yb.values, dtype=torch.float32)
        mean_rating = df.iloc[:, 2].mean()
        num_users, num_items = df[labels[domain][0]].max()+1, df[labels[domain][1]].max()+1
        model = MatrixModel(num_users, num_items, mean_rating, args.size)
        loss = nn.MSELoss(reduction='sum')
        if args.GPU != -1: model, loss = model.cuda(args.GPU), loss.cuda(args.GPU)
        train(model, xb.numpy(), yb.numpy(), loss)