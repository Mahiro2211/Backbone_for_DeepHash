# Implementation of SoftTriple Loss
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init


class SoftTriple(nn.Module):
    def __init__(self, la, gamma, tau, margin, dim, cN, K,device):
        super(SoftTriple, self).__init__()
        self.la = la #0.1
        self.gamma = 1. / gamma
        self.tau = tau
        self.margin = margin # 1
        self.cN = cN # num_class
        self.K = K # 4
        self.device = device
        self.fc = Parameter(torch.Tensor(dim, cN * K)).to(device)
        self.weight = torch.zeros(cN * K, cN * K, dtype=torch.bool).to(device)
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i * K + j, i * K + j + 1:(i + 1) * K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target,device):
        target = target.float()
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc * self.gamma, dim=2)
        simClass = torch.sum(prob * simStruc, dim=2)
        marginM = torch.zeros(simClass.shape).to(self.device)
        marginM = target - marginM
        marginM[marginM > 0] = self.margin
        #marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la * (simClass - marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0 + 1e-5 - 2. * simCenter[self.weight])) / (self.cN * self.K * (self.K - 1.))
            return lossClassify + self.tau * reg
        else:
            return lossClassify


if __name__ == "__main__":
    feature = torch.rand(3, 16)
    # labels = torch.tensor(1, 2,).float()
    _labels = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()
    #_labels = F.pad(_labels, (0, 3, 0, 0), "constant", 0)
    _targets = torch.argmax(_labels, dim=1)
    # label=[1,2,3]
    cerition = SoftTriple(la=0.1, gamma=0.1, tau=0.2, margin=1, cN=3, K=10, dim=16)
    loss = cerition(feature, _labels)
    print(loss)


