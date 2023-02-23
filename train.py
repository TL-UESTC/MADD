import argparse
import sys
import numpy as np
from model import MADD
from dataset import dualDataSet, splitTrainTest, getEmbeddings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from evaluate import PR, HR_NDCG

class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()
    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)
        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)
        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        return diff_loss

class SIMSE(nn.Module):
    def __init__(self):
        super(SIMSE, self).__init__()
    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)
        return simse

parser = argparse.ArgumentParser(description='MADD Model Training')
parser.add_argument('--alpha_weight', default=0.075, type=float, help='diffLossWeight')
parser.add_argument('--beta_weight', default=0.25, type=float, help='simLossWeight')
parser.add_argument('--gamma_weight', default=0.1, type=float, help='encoderLossWeight')
parser.add_argument('--eta_weight', default=0.25, type=float, help='sharedecoderLossWeight')
parser.add_argument('--theta_weight', default=1, type=float, help='predictLossWeight')
parser.add_argument('--size', required=True, type=int, help='size of MF / size of Doc2Vec')
parser.add_argument('--batchsize', default=1024, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--Epoch', default=100, type=int)
parser.add_argument('--active_domain_loss_step', default=-1, type=int)
parser.add_argument('--topK', default=10, type=int)
parser.add_argument('--negNum', default=99, type=int, help='len of recommendation list')
parser.add_argument('--GPU', default=-1, help='CUDA device -1,0,1,2', type=int)
parser.add_argument('--domainA', required=True, type=str)
parser.add_argument('--domainB', required=True, type=str)
parser.add_argument('--eva_hr_ndcg', action='store_true')
parser.add_argument('--eva_pr', action='store_true')
args = parser.parse_args()


def train(model, trainA, trainB):
    train_dataset_A = dualDataSet(trainA[:, 0], trainA[:, 1], trainA[:, 3:3+size*2], trainA[:, 3+size*2:], trainA[:, 2])
    train_iter_A = DataLoader(train_dataset_A, batch_size = args.batchsize, shuffle = True)
    train_dataset_B = dualDataSet(trainB[:, 0], trainB[:, 1], trainB[:, 3:3+size*2], trainB[:, 3+size*2:], trainB[:, 2])
    train_iter_B = DataLoader(train_dataset_B, batch_size = args.batchsize, shuffle = True)

    if args.eva_hr_ndcg:
        TestNeg1 = np.load('./testdata/{}{}_{}TestNeg.npy'.format(domainA, domainB, domainA))
        eva1 = HR_NDCG(TestNeg1, usersA, itemsA, usersB, 0, args.topK, args.negNum, args.GPU)

        TestNeg2 = np.load('./testdata/{}{}_{}TestNeg.npy'.format(domainA, domainB, domainB))
        eva2 = HR_NDCG(TestNeg2, usersB, itemsB, usersA, 1, args.topK, args.negNum, args.GPU)

    if args.eva_pr:
        pr1 = PR(train_iter_A, usersB, 0, args.GPU)
        pr2 = PR(train_iter_B, usersA, 1, args.GPU)

    len_dataloader = min(len(train_iter_A), len(train_iter_B))
    dann_epoch = np.floor(args.active_domain_loss_step / len_dataloader * 1.0)
    current_stepA = 0
    current_stepB = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_classification = nn.MSELoss()
    loss_recon = SIMSE()
    loss_diff = DiffLoss()
    loss_autoencoder = nn.MSELoss()
    loss_similarity = torch.nn.CrossEntropyLoss()
    if args.GPU != -1:
        loss_classification = nn.MSELoss().cuda(args.GPU)
        loss_recon = SIMSE().cuda(args.GPU)
        loss_diff = DiffLoss().cuda(args.GPU)
        loss_autoencoder = nn.MSELoss().cuda(args.GPU)
        loss_similarity = torch.nn.CrossEntropyLoss().cuda(args.GPU)
        model = model.cuda(args.GPU)
    # evaluate hr and ndcg
    hr_1, ndcg_1 = -1, -1
    hr_2, ndcg_2 = -1, -1

    for epoch in range(args.Epoch):
        model.train()
        k = 0
        batchs_A = iter(train_iter_A)
        batchs_B = iter(train_iter_B)
        while k < len(train_iter_B):
            if k < len(train_iter_A):
                u, i, u_embedding, i_embedding, rating = next(batchs_A)

                domainv_label = torch.zeros(len(rating)).long()
                u2_embedding = torch.tensor(np.array([usersB[xx] for xx in u.tolist()]))
                if args.GPU != -1:
                    u_embedding, i_embedding, u2_embedding, rating, domainv_label = \
                        u_embedding.cuda(args.GPU), i_embedding.cuda(args.GPU), u2_embedding.cuda(args.GPU), rating.cuda(args.GPU), domainv_label.cuda(args.GPU)
                loss, lossSim = 0, 0
                if current_stepA > args.active_domain_loss_step:
                    p = float(k + (epoch - dann_epoch) * len_dataloader / (args.Epoch - dann_epoch) / len_dataloader)
                    p = 2. / (1. + np.exp(-10 * p)) - 1
                    domain_label = model(u_embedding, i_embedding, u2_embedding, 0, p)[1]
                    lossSim = args.beta_weight * loss_similarity(domain_label, domainv_label)
                    loss += lossSim
                share_uembedding, domain_label, pria_uembedding, encode_iembedding, \
                            decode_iembedding, sharedecoder, predicta = model(u_embedding, i_embedding, u2_embedding, 0)
                lossDiff = args.alpha_weight * loss_diff(share_uembedding, pria_uembedding)
                lossRecon = args.eta_weight * loss_recon(sharedecoder, u_embedding)
                lossEncoder = args.gamma_weight * loss_autoencoder(i_embedding, decode_iembedding)

                lossClass = args.theta_weight * loss_classification(predicta.squeeze(), rating)
                loss = loss + lossDiff + lossRecon + lossEncoder + lossClass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                current_stepA += 1
                sys.stdout.write('\rDomain A : Epoch {} / {} | Batch {} / {} | lossDiff: {:.4f} | '
                                 'lossSim: {:.8f} | lossRecon: {:.8f} | lossEncoder: {:.8f} | lossClass: {:.4f}'.format(
                                epoch+1, args.Epoch, k, len(train_iter_A), lossDiff, lossSim, lossRecon, lossEncoder,lossClass))
                sys.stdout.flush()
                if args.eva_hr_ndcg:
                    hr, ndcg = eva1.evaluate(model)
                    hr_1, ndcg_1 = max(hr, hr_1), max(ndcg, ndcg_1)
                    print('\n[Evaluate HR & NDCG] : Domain {} Current HR: {:.4f}, Current NDCG: {:.4f}'.format(domainA, hr, ndcg))
                    print('[Evaluate HR & NDCG] : Domain {} Best HR: {:.4f}, Best NDCG: {:.4f}'.format(domainA, hr_1, ndcg_1))

            u, i, u_embedding, i_embedding, rating = next(batchs_B)

            domainv_label = torch.ones(len(rating)).long()
            u2_embedding = torch.tensor(np.array([usersA[xx] for xx in u.tolist()]))
            if args.GPU != -1:
                u_embedding, i_embedding, u2_embedding, rating, domainv_label = \
                    u_embedding.cuda(args.GPU), i_embedding.cuda(args.GPU), u2_embedding.cuda(args.GPU), rating.cuda(args.GPU), domainv_label.cuda(args.GPU)
            loss, lossSim = 0, 0
            if current_stepB > args.active_domain_loss_step:
                p = float(k + (epoch - dann_epoch) * len_dataloader / (args.Epoch - dann_epoch) / len_dataloader)
                p = 2. / (1. + np.exp(-10 * p)) - 1
                domain_label = model(u_embedding, i_embedding, u2_embedding, 1, p)[1]
                lossSim = args.beta_weight * loss_similarity(domain_label, domainv_label)
                loss += lossSim
            share_uembedding, domain_label, prib_uembedding, encode_iembedding, \
                        decode_iembedding, sharedecoder, predictb = model(u_embedding, i_embedding, u2_embedding, 1)

            lossDiff = args.alpha_weight * loss_diff(share_uembedding, prib_uembedding)
            lossRecon = args.eta_weight * loss_recon(sharedecoder, u_embedding)
            lossEncoder = args.gamma_weight * loss_autoencoder(i_embedding, decode_iembedding)
            lossClass = args.theta_weight * loss_classification(predictb.squeeze(), rating)
            loss = loss + lossDiff + lossRecon + lossEncoder + lossClass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_stepB += 1

            k += 1

            sys.stdout.write('\rDomain B : Epoch {} / {} | Batch {} / {} | lossDiff: {:.4f} | '
                             'lossSim: {:.8f} | lossRecon: {:.8f} | lossEncoder: {:.8f} | lossClass: {:.4f}'.format(
                            epoch+1, args.Epoch, k, len(train_iter_B), lossDiff, lossSim, lossRecon, lossEncoder, lossClass))
            sys.stdout.flush()

            if args.eva_hr_ndcg:
                hr, ndcg = eva2.evaluate(model)
                hr_2, ndcg_2 = max(hr, hr_2), max(ndcg, ndcg_2)
                print('\n[Evaluate HR & NDCG] : Domain {} Current HR: {:.4f}, Current NDCG: {:.4f}'.format(domainB, hr, ndcg))
                print('[Evaluate HR & NDCG] : Domain {} Best HR: {:.4f}, Best NDCG: {:.4f}'.format(domainB, hr_2, ndcg_2))
        if args.eva_pr:
            mse, mae, precision, recall = pr1.evaluate(model)
            print('[Evaluate MSE & MAE & Precision & Recall] : Domain {} MSE : {:.4f} MAE : {:.4f} '
                  'Precision : {:.4f} Recall : {:.4f}'.format(domainA, mse, mae, precision, recall))

            mse, mae, precision, recall = pr2.evaluate(model)
            print('[Evaluate MSE & MAE & Precision & Recall] : Domain {} MSE : {:.4f} MAE : {:.4f} '
                  'Precision : {:.4f} Recall : {:.4f}'.format(domainB, mse, mae, precision, recall))

    torch.save(model.state_dict(), './pthdata/MADD_{}{}_k{}.pth'.format(domainA, domainB, size))

if __name__ == '__main__':
    domainA, domainB, size = args.domainA.title(), args.domainB.title(), args.size
    dataA = np.load('./inputdata/{}{}_{}InputData_k{}.npy'.format(domainA, domainB, domainA, size))
    dataB = np.load('./inputdata/{}{}_{}InputData_k{}.npy'.format(domainA, domainB, domainB, size))
    dataA = dataA.astype(np.float32)
    dataB = dataB.astype(np.float32)
    dataA[:, 2] = dataA[:, 2] / dataA[:, 2].max()
    dataB[:, 2] = dataB[:, 2] / dataB[:, 2].max()

    usersA, itemsA = getEmbeddings(dataA, size)
    usersB, itemsB = getEmbeddings(dataB, size)
    trainA, testA = splitTrainTest(dataA)
    trainB, testB = splitTrainTest(dataB)

    model = MADD(args)
    train(model, trainA, trainB)