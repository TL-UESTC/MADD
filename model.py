import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p
        return output, None

class MADD(nn.Module):
    def __init__(self, args):
        super(MADD, self).__init__()
        self.size = [args.size*2, args.size, args.size//2, args.size//4]
        ###### private encoder A #####
        self.priA = nn.Sequential(
            nn.Linear(in_features=self.size[0], out_features=self.size[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[1], out_features=self.size[2]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[2], out_features=self.size[3]),
        )
        ####### private encoder B #####
        self.priB = nn.Sequential(
            nn.Linear(in_features=self.size[0], out_features=self.size[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[1], out_features=self.size[2]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[2], out_features=self.size[3]),
        )
        ####### shared encoder #######
        self.shared = nn.Sequential(
            nn.Linear(in_features=self.size[0], out_features=self.size[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[1], out_features=self.size[2]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[2], out_features=self.size[3]),
        )

        self.shared_encoder_pred_domain = nn.Sequential(
            nn.Linear(in_features=self.size[3], out_features=self.size[3]),
            nn.ReLU(True),
            nn.Linear(in_features=self.size[3], out_features=2),
        )

        self.shared_decoder = nn.Sequential(
            nn.Linear(in_features=self.size[2], out_features=self.size[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[1], out_features=self.size[0])
        )

        ####### domainA items autoencoder #######
        self.encoderA = nn.Sequential(
            nn.Linear(in_features=self.size[0], out_features=self.size[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[1], out_features=self.size[2]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[2], out_features=self.size[3]),
        )
        self.decoderA = nn.Sequential(
            nn.Linear(in_features=self.size[3], out_features=self.size[2]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[2], out_features=self.size[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[1], out_features=self.size[0]),
        )
        ####### domainB items autoencoder #######
        self.encoderB = nn.Sequential(
            nn.Linear(in_features=self.size[0], out_features=self.size[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[1], out_features=self.size[2]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[2], out_features=self.size[3]),
        )
        self.decoderB = nn.Sequential(
            nn.Linear(in_features=self.size[3], out_features=self.size[2]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[2], out_features=self.size[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[1], out_features=self.size[0]),
        )

        ####### domainA predict #######
        self.predict_user_A = nn.Sequential(
            nn.Linear(in_features=self.size[3], out_features=self.size[3]*2),
            nn.ReLU(),
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
        )

        self.predict_item_A = nn.Sequential(
            nn.Linear(in_features=self.size[3], out_features=self.size[3]*2),
            nn.ReLU(),
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
        )

        ####### domainB predict #######
        self.predict_user_B = nn.Sequential(
            nn.Linear(in_features=self.size[3], out_features=self.size[3]*2),
            nn.ReLU(),
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
        )

        self.predict_item_B = nn.Sequential(
            nn.Linear(in_features=self.size[3], out_features=self.size[3]*2),
            nn.ReLU(),
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
        )

        self.A2B = nn.Sequential(
            nn.Linear(in_features=self.size[3], out_features=self.size[3]),
        )
        self.B2A = nn.Sequential(
            nn.Linear(in_features=self.size[3], out_features=self.size[3]),
        )
        ####### domainA Attention #######
        self.attention1A = nn.Sequential(
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
            nn.ReLU()
        )
        self.atVec1A =  nn.Parameter(torch.randn(self.size[3]), True)
        self.atBias1A = nn.Parameter(torch.FloatTensor([0]), True)

        self.attention2A = nn.Sequential(
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
            nn.ReLU()
        )
        self.atVec2A = nn.Parameter(torch.randn(self.size[3]), True)
        self.atBias2A = nn.Parameter(torch.FloatTensor([0]), True)

        self.attention3A = nn.Sequential(
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
            nn.ReLU()
        )
        self.atVec3A = nn.Parameter(torch.randn(self.size[3]), True)
        self.atBias3A = nn.Parameter(torch.FloatTensor([0]), True)

        self.attention4A = nn.Sequential(
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
            nn.ReLU()
        )
        self.atVec4A = nn.Parameter(torch.randn(self.size[3]), True)
        self.atBias4A = nn.Parameter(torch.FloatTensor([0]), True)

        ####### domainB Attention #######
        self.attention1B = nn.Sequential(
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
            nn.ReLU()
        )
        self.atVec1B = nn.Parameter(torch.randn(self.size[3]), True)
        self.atBias1B = nn.Parameter(torch.FloatTensor([0]), True)

        self.attention2B = nn.Sequential(
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
            nn.ReLU()
        )
        self.atVec2B = nn.Parameter(torch.randn(self.size[3]), True)
        self.atBias2B = nn.Parameter(torch.FloatTensor([0]), True)

        self.attention3B = nn.Sequential(
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
            nn.ReLU()
        )
        self.atVec3B = nn.Parameter(torch.randn(self.size[3]), True)
        self.atBias3B = nn.Parameter(torch.FloatTensor([0]), True)

        self.attention4B = nn.Sequential(
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
            nn.ReLU()
        )
        self.atVec4B = nn.Parameter(torch.randn(self.size[3]), True)
        self.atBias4B = nn.Parameter(torch.FloatTensor([0]), True)

    def forward(self, u_embedding, i_embedding, u2_embedding, mode, p=0.0):
        results = []
        share_uembedding = self.shared(u_embedding)
        results.append(share_uembedding)
        reversed_shared_code = ReverseLayerF.apply(share_uembedding, p)
        domain_label = self.shared_encoder_pred_domain(reversed_shared_code)
        results.append(domain_label)

        if mode == 0:
            pria_uembedding = self.priA(u_embedding)
            results.append(pria_uembedding)
            encode_iembedding = self.encoderA(i_embedding)
            results.append(encode_iembedding)
            decode_iembedding = self.decoderA(encode_iembedding)
            results.append(decode_iembedding)
            sharedecoder = self.shared_decoder(torch.cat((pria_uembedding, share_uembedding), dim=1))
            results.append(sharedecoder)
            pribtoa = self.B2A(self.priB(u2_embedding))

            tmp_p = torch.cat((pria_uembedding, pribtoa), dim=1)
            Weight1 = torch.exp(torch.sum(self.attention1A(tmp_p).mul(self.atVec1A), dim=1) + self.atBias1A).unsqueeze(dim=1)
            Weight2 = torch.exp(torch.sum(self.attention2A(tmp_p).mul(self.atVec2A), dim=1) + self.atBias2A).unsqueeze(dim=1)
            pri_uembedding = Weight1*pria_uembedding + Weight2*pribtoa

            tmp_ps = torch.cat((pri_uembedding, share_uembedding), dim=1)
            Weight3 = torch.exp(torch.sum(self.attention3A(tmp_ps).mul(self.atVec3A), dim=1) + self.atBias3A).unsqueeze(dim=1)
            Weight4 = torch.exp(torch.sum(self.attention4A(tmp_ps).mul(self.atVec4A), dim=1) + self.atBias4A).unsqueeze(dim=1)
            domainu_embedding = Weight3*pri_uembedding + Weight4*share_uembedding

            predicta = torch.sum(self.predict_user_A(domainu_embedding).mul(self.predict_item_A(encode_iembedding)), dim=1)
            results.append(predicta)
        if mode == 1:
            prib_uembedding = self.priB(u_embedding)
            results.append(prib_uembedding)
            encode_iembedding = self.encoderB(i_embedding)
            results.append(encode_iembedding)
            decode_iembedding = self.decoderB(encode_iembedding)
            results.append(decode_iembedding)
            sharedecoder = self.shared_decoder(torch.cat((prib_uembedding, share_uembedding), dim=1))
            results.append(sharedecoder)
            priatob = self.A2B(self.priA(u2_embedding))

            tmp_p = torch.cat((prib_uembedding, priatob), dim=1)
            Weight1 = torch.exp(torch.sum(self.attention1B(tmp_p).mul(self.atVec1B), dim=1) + self.atBias1B).unsqueeze(dim=1)
            Weight2 = torch.exp(torch.sum(self.attention2B(tmp_p).mul(self.atVec2B), dim=1) + self.atBias2B).unsqueeze(dim=1)
            pri_uembedding = Weight1*prib_uembedding + Weight2*priatob

            tmp_ps = torch.cat((pri_uembedding, share_uembedding), dim=1)
            Weight3 = torch.exp(torch.sum(self.attention3B(tmp_ps).mul(self.atVec3B), dim=1) + self.atBias3B).unsqueeze(dim=1)
            Weight4 = torch.exp(torch.sum(self.attention4B(tmp_ps).mul(self.atVec4B), dim=1) + self.atBias4B).unsqueeze(dim=1)
            domainu_embedding = Weight3*pri_uembedding + Weight4*share_uembedding

            predictb = torch.sum(self.predict_user_B(domainu_embedding).mul(self.predict_item_B(encode_iembedding)), dim=1)
            results.append(predictb)
        return results