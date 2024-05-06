import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class DTSHLoss(torch.nn.Module):
    def __init__(self):
        super(DTSHLoss, self).__init__()

    def forward(self, u, y ):

        inner_product = u @ u.t()
        s = y @ y.t() > 0
        count = 0

        loss1 = 0
        for row in range(s.shape[0]):
            # if has positive pairs and negative pairs
            if s[row].sum() != 0 and (~s[row]).sum() != 0:
                count += 1
                theta_positive = inner_product[row][s[row] == 1]
                theta_negative = inner_product[row][s[row] == 0]
                triple = (theta_positive.unsqueeze(1) - theta_negative.unsqueeze(0) - 5).clamp(min=-100,
                                                                                                             max=50)
                loss1 += -(triple - torch.log(1 + torch.exp(triple))).mean()

        if count != 0:
            loss1 = loss1 / count
        else:
            loss1 = 0

        loss2 = (u - u.sign()).pow(2).mean()

        return loss1 + loss2
class ProjectLayer(nn.Module):
    def __init__(self,nbit):
        super(ProjectLayer,self).__init__()
        self.linear_strc = nn.Sequential(nn.Linear(nbit, 512),
                                      nn.BatchNorm1d(512),
                                      nn.Linear(512, nbit),
                                      nn.BatchNorm1d(nbit),
                                      nn.Sigmoid() )
    def forward(self,x):
        return 1+0.5*(2*self.linear_strc(x)-1)
def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class NASA_loss(nn.Module):
    def __init__(self, t=3, alpha=10,device=None):
        super().__init__()
        self.margin = t
        self.scale = alpha
        self.device = device

    def forward(self, embeddings, embeddings_strc):
        dist_mat_ori = pdist(embeddings).view(-1)
        mean = dist_mat_ori.mean().detach()
        std = dist_mat_ori.std().detach()
        self.eta = torch.tensor([mean, torch.max(0.2 * mean, mean - self.margin * std)])

        mm, nn = embeddings.shape[0], embeddings.shape[1]
        adapt_dist_mat = torch.zeros([mm, mm])
        for i in range(0, embeddings.shape[0]):
            dis = (embeddings[i] - embeddings).pow(2).clamp(min=1e-12)
            tmp = torch.mul(embeddings_strc[i].expand(mm, nn), dis)
            adapt_dist_mat[i] = torch.sqrt(torch.sum(tmp, 1))

        dis_mat_noanchor = adapt_dist_mat[
            ~torch.eye(adapt_dist_mat.shape[0], dtype=torch.bool, device=adapt_dist_mat.device, )]
        pos_group = (dis_mat_noanchor[None].to(self.device) - self.eta[:, None].to(self.device)).abs()[0]
        neg_group = (dis_mat_noanchor[None].to(self.device) - self.eta[:, None].to(self.device)).abs()[1]
        c = torch.exp(-self.scale * pos_group) / (
                    torch.exp(-self.scale * pos_group) + torch.exp(-self.scale * neg_group))
        self_ranking_loss = (c * pos_group + (1 - c) * neg_group).mean()

        return self_ranking_loss


class Triplet(nn.Module):
    def __init__(self, margin=0.2, sampler=None, reduce=True, size_average=True):
        super().__init__()
        self.margin = margin
        self.sampler = sampler
        self.sampler.dist_func = lambda e: pdist(e, squared=(p == 2))
        self.reduce = reduce
        self.size_average = size_average

    def forward(self, x, y):
        a_id, p_id, n_id = self.sampler(x, y)
        loss_metric = F.triplet_margin_loss(x[a_id], x[p_id], x[n_id], margin=self.margin, reduction="none")
        return loss_metric.mean()

class DTSHLoss(torch.nn.Module):
    def __init__(self,):
        super(DTSHLoss, self).__init__()

    def forward(self, u, y,feat2=None):

        inner_product = u @ u.t() if feat2==None else u @ feat2.t()
        s = y.float() @ y.float().t() > 0
        count = 0

        loss1 = 0
        for row in range(s.shape[0]):
            # if has positive pairs and negative pairs
            if s[row].sum() != 0 and (~s[row]).sum() != 0:
                count += 1
                theta_positive = inner_product[row][s[row] == 1]
                theta_negative = inner_product[row][s[row] == 0]
                triple = (theta_positive.unsqueeze(1) - theta_negative.unsqueeze(0) - 0.5).clamp(
                    min=-100,
                    max=50)
                loss1 += -(triple - torch.log(1 + torch.exp(triple))).mean()

        if count != 0:
            loss1 = loss1 / count
        else:
            loss1 = 0

        loss2 = 0.1 * (u - u.sign()).pow(2).mean()

        return loss1 + loss2

class DPSHLoss(torch.nn.Module):
    def __init__(self, num_train,nclass, bit,device):
        super(DPSHLoss, self).__init__()


    def forward(self, u, y, feat2=None):

        s = (y @ y.t() > 0).float()
        inner_product = u @ u.t() * 0.5 if feat2 == None else u @ feat2.t() * 0.5

        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(
            min=0) - s * inner_product

        likelihood_loss = likelihood_loss.mean()

        quantization_loss = 0.1 * (u - u.sign()).pow(2).mean()

        return likelihood_loss + quantization_loss