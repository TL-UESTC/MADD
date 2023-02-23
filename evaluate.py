import argparse
import numpy as np
from math import log2
import torch
import heapq
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
from model import MADD
from dataset import dualDataSet, splitTrainTest, getEmbeddings
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import functools

sns.set(rc={'figure.figsize':(18,10)})
palette = sns.color_palette("bright", 10)

class PR():
    def __init__(self, Iter, users, mode, GPU):
        self.Iter = Iter
        self.mode = mode
        self.users = users
        self.GPU = GPU
    def evaluate(self, model):
        prediction, recall = [], []
        golden, scores = [], []
        for u, i, u_embedding, i_embedding, rating in self.Iter:
            golden += list(rating.detach().numpy())
            u2_embedding = torch.tensor(np.array([self.users[xx] for xx in u.tolist()]))
            if self.GPU != -1:
                u_embedding, i_embedding, u2_embedding, rating = \
                    u_embedding.cuda(self.GPU), i_embedding.cuda(self.GPU), u2_embedding.cuda(self.GPU), rating.cuda(self.GPU)
            predict = model(u_embedding, i_embedding, u2_embedding, self.mode)[-1]
            scores += list(predict.detach().numpy().squeeze())
            n_rel = sum((true_r >= 0.5) for true_r in rating)
            n_rec_k = sum((est >= 0.5) for est in predict)
            n_rel_and_rec_k = sum(((true_r >= 0.5) and (est >= 0.5)) for true_r, est in zip(rating, predict))
            prediction.append(n_rel_and_rec_k / n_rec_k if n_rec_k != torch.tensor([0]) else torch.tensor([1]))
            recall.append(n_rel_and_rec_k / n_rel if n_rel != torch.tensor([0]) else torch.tensor([1]))
        MSE = mean_squared_error(golden, scores)
        MAE = mean_absolute_error(golden, scores)
        precision = np.mean([x.item() for x in prediction])
        recall = np.mean([x.item() for x in recall])
        return MSE, MAE, precision, recall

class HR_NDCG():
    def __init__(self, testNeg, users, items, user2, mode, topK, negNum, GPU):
        self.users, self.items, self.users2 = users, items, user2
        self.testNeg = testNeg
        self.mode = mode
        self.topK = topK
        self.negNum = negNum
        self.GPU = GPU

    def getNDCG(self, ranklist, targetItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == targetItem:
                return log2(2) / log2(i + 2)
        return 0
    def getHitRatio(self, ranklist, targetItem):
        for item in ranklist:
            if item == targetItem:
                return 1
        return 0

    def evaluate(self, model):
        HR, NDCG = [], []
        testUser, testItem = self.testNeg
        for i in range(len(testUser)):
            target = testItem[i][0]
            item_score_dict = {}
            uembedding = torch.tensor(np.array([self.users[testUser[i][0]] for _ in range(self.negNum+1)]))
            u2embedding = torch.tensor(np.array([self.users2[testUser[i][0]] for _ in range(self.negNum+1)]))
            iembedding = torch.tensor(np.array([self.items[testItem[i][j]] for j in range(self.negNum+1)]))
            if self.GPU != -1:
                uembedding = uembedding.cuda(self.GPU)
                u2embedding = u2embedding.cuda(self.GPU)
                iembedding = iembedding.cuda(self.GPU)
            predicts = model(uembedding, iembedding, u2embedding, self.mode)[-1].detach().cpu().numpy()
            for j in range(self.negNum+1):
                item_score_dict[testItem[i][j]] = predicts[j]
            ranklist = heapq.nlargest(self.topK, item_score_dict, key=item_score_dict.get)
            tmp_hr = self.getHitRatio(ranklist, target)
            tmp_NDCG = self.getNDCG(ranklist, target)
            HR.append(tmp_hr)
            NDCG.append(tmp_NDCG)
        return np.mean(HR), np.mean(NDCG)

class tSNE():
    def __init__(self, iterA, iterB, usersA, usersB):
        self.iterA = iterA
        self.iterB = iterB
        self.usersA = usersA
        self.usersB = usersB

    def get_data(self, Iter, users, model, mode):
        data, label = [], []
        for u, i, u_embedding, i_embedding, rating in Iter:
            u2_embedding = torch.tensor(np.array([users[xx] for xx in u.tolist()]))
            share_uembedding, _, pri_uembedding, *x = model(u_embedding, i_embedding, u2_embedding, mode)
            labels = torch.zeros(share_uembedding.shape[0])
            labelp = torch.ones(pri_uembedding.shape[0])
            data.append(share_uembedding)
            data.append(pri_uembedding)
            label.append(labels)
            label.append(labelp)
        data = functools.reduce(lambda x, y: torch.cat((x,y), dim=0), data)
        label = functools.reduce(lambda x, y: torch.cat((x,y), dim=0), label)
        return data.detach().numpy(), label.detach().numpy()

    def plot_embedding(self, datab, labelb, datam, labelm):
        colors = ['r', 'b', 'g', 'c']
        xb_min, xb_max = np.min(datab, 0), np.max(datab, 0)
        datab = (datab - xb_min) / (xb_max - xb_min)
        xm_min, xm_max = np.min(datam, 0), np.max(datam, 0)
        datam = (datam - xm_min) / (xm_max - xm_min)

        plt.figure(1)
        for i in range(datab.shape[0]):
            if int(labelb[i]) == 0:
                plt.subplot(133)
                plt.xlabel('(c)')
                plt.scatter(datab[i, 0], datab[i, 1], s=12, c=colors[int(labelb[i])])
                plt.xticks([])
                plt.yticks([])
                plt.grid(True)
            plt.subplot(131)
            plt.xlabel('(a)')
            plt.scatter(datab[i, 0], datab[i, 1], s=12, c=colors[int(labelb[i])])
            plt.xticks([])
            plt.yticks([])
            plt.grid(True)

        plt.figure(1)
        for i in range(datam.shape[0]):
            if int(labelm[i]) == 0:
                plt.subplot(133)
                plt.xlabel('(c)')
                plt.scatter(datam[i, 0], datam[i, 1], s=12, c=colors[int(labelm[i]) + 2])
                plt.xticks([])
                plt.yticks([])
                plt.grid(True)
            plt.subplot(132)
            plt.xlabel('(b)')
            plt.scatter(datam[i, 0], datam[i, 1], s=12, c=colors[int(labelm[i]) + 2])
            plt.xticks([])
            plt.yticks([])
            plt.grid(True)

    def plottsne(self, model):
        dataA, labelA = self.get_data(self.iterA, self.usersB, model, 0)
        dataB, labelB = self.get_data(self.iterB, self.usersA, model, 1)
        print('Computing t-SNE embedding')
        tsne = TSNE(n_components=2, learning_rate=300, init='pca')
        resultA, resultB = tsne.fit_transform(dataA), tsne.fit_transform(dataB)
        self.plot_embedding(resultA, labelA, resultB, labelB)
        plt.show()

def returnIter(train, test, size, batchsize):
    train_dataset = dualDataSet(train[:, 0], train[:, 1], train[:, 3:3+size*2], train[:, 3+size*2:], train[:, 2])
    train_iter = DataLoader(train_dataset, batch_size = batchsize, shuffle = True)
    test_dataset = dualDataSet(test[:, 0], test[:, 1], test[:, 3:3 + size*2], test[:, 3+size*2:], test[:, 2])
    test_iter = DataLoader(test_dataset, batch_size = batchsize, shuffle = True)
    return train_iter, test_iter

def printresults(*kargs):
    name = kargs[0]
    MSE, MAE, Precision, Recall = kargs[1]
    print('[Domain {}] MSE : {} | MAE : {} | Precision : {} | Recall : {}'.format(name, MSE, MAE, Precision, Recall))

def main():
    parser = argparse.ArgumentParser(description='Evaluate')
    parser.add_argument('--negNum', default=99, help='len of recommendataion list')
    parser.add_argument('--domainA', required=True, type=str)
    parser.add_argument('--domainB', required=True, type=str)
    parser.add_argument('--size', required=True, type=int, help='size of MF / size of Doc2Vec')
    parser.add_argument('--batchsize', default=512, type=int)
    parser.add_argument('--GPU', default=-1, help='CUDA device -1,0,1,2', type=int)
    parser.add_argument('--evapr', action='store_true')
    parser.add_argument('--evatsne', action='store_true')
    args = parser.parse_args()

    domainA, domainB, size = args.domainA.title(), args.domainB.title(), args.size
    dataA = np.load('./inputdata/{}{}_{}InputData_k{}.npy'.format(domainA, domainB, domainA, size))
    dataB = np.load('./inputdata/{}{}_{}InputData_k{}.npy'.format(domainA, domainB, domainB, size))
    dataA = dataA.astype(np.float32)
    dataB = dataB.astype(np.float32)
    dataA[:, 2] = dataA[:, 2] / dataA[:, 2].max()
    dataB[:, 2] = dataB[:, 2] / dataB[:, 2].max()
    trainA, testA = splitTrainTest(dataA)
    trainB, testB = splitTrainTest(dataB)
    usersA, itemsA = getEmbeddings(dataA, size)
    usersB, itemsB = getEmbeddings(dataB, size)
    train_iter_A, test_iter_A = returnIter(trainA, testA, size, args.batchsize)
    train_iter_B, test_iter_B = returnIter(trainB, testB, size, args.batchsize)

    model = MADD(args)
    if args.GPU == -1:
        model.load_state_dict(torch.load('./pthdata/MADD_{}{}_k{}.pth'.format(domainA, domainB, size), map_location='cpu'))
    else:
        model.load_state_dict(torch.load('./pthdata/MADD_{}{}_k{}.pth'.format(domainA, domainB, size)))

    if args.evapr:
        print('Train Data:')
        printresults(domainA, PR(train_iter_A, usersB, 0, args.GPU).evaluate(model))
        printresults(domainB, PR(train_iter_B, usersA, 1, args.GPU).evaluate(model))
        print('Test Data:')
        printresults(domainA, PR(test_iter_A, usersB, 0, args.GPU).evaluate(model))
        printresults(domainB, PR(test_iter_B, usersA, 1, args.GPU).evaluate(model))
    if args.evatsne:
        t = tSNE(train_iter_A, train_iter_B, usersA, usersB)
        t.plottsne(model)
if __name__ == '__main__':
    main()
