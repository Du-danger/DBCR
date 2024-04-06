import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from utils import *
import torch.nn.functional as F
from torch_scatter import scatter_sum

class DBCR(nn.Module):
    def __init__(self, n_u, n_i, adj_norm, edgE_vndex, args):
        super(DBCR,self).__init__()
        self.n_u = n_u
        self.n_i = n_i
        self.n_b = args.n_b
        self.adj_norm = adj_norm
        self.d = d = args.d

        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))
        self.E_v_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))
        self.E_b = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_b , d)))
        self.E_u = None
        self.E_v = None

        self.l = args.l
        self.dropout = args.dropout
        self.temp = args.temp
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.lambda_3 = args.lambda_3
        self.is_norm = args.is_norm
        self.is_mean = args.is_mean
        self.edgE_vndex = edgE_vndex
        self.num_edges = edgE_vndex.shape[1]
        self.device = args.device
    
    def cal_pcl_loss(self, uids, iids, G_u_norm, G_i_norm):
        E_u_norm = self.E_u
        E_v_norm = self.E_v
        neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_v_norm.T / self.temp).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp, -5.0, 5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_v_norm[iids]).sum(1) / self.temp, -5.0, 5.0)).mean()
        loss_cl = -pos_score + neg_score
        return loss_cl
    

    def forward(self, uids, iids, pos, neg):
        E_u_list = [None] * (self.l+1)
        E_v_list = [None] * (self.l+1)
        E_u_list[0] = self.E_u_0
        E_v_list[0] = self.E_v_0

        Z_u_list = [None] * (self.l+1)
        Z_i_list = [None] * (self.l+1)
        Z_u_list[0] = self.E_u_0
        Z_i_list[0] = self.E_v_0

        # representation learing on the original view
        adj_norm = self.adj_norm.to(self.device)
        for layer in range(1,self.l+1):
            E_u_list[layer] = (torch.spmm(sparse_dropout(adj_norm, self.dropout), E_v_list[layer-1]))
            E_v_list[layer] = (torch.spmm(sparse_dropout(adj_norm, self.dropout).transpose(0,1), E_u_list[layer-1]))

        self.E_u = sum(E_u_list)
        self.E_v = sum(E_v_list)

        # preference view learning
        src, dst = self.edgE_vndex[0], self.edgE_vndex[1]
        x_u, x_i = self.E_u[src], self.E_v[dst]
        edge_logits = torch.mul(x_u, x_i).sum(1)
        Ag = torch.sigmoid(edge_logits).squeeze()
        batch_aug_edge_weight = Ag
        
        # representation learing on the preference view
        weight = batch_aug_edge_weight.detach().cpu()
        aug_adj = new_graph(torch.tensor(self.edgE_vndex), weight, self.n_u, self.n_i)
        adj_norm = self.adj_norm
        aug_adj = aug_adj * adj_norm
        aug_adj = aug_adj.to(self.device)
        for layer in range(1,self.l+1):
            Z_u_list[layer] = (torch.spmm(sparse_dropout(aug_adj, self.dropout), Z_i_list[layer-1]))
            Z_i_list[layer] = (torch.spmm(sparse_dropout(aug_adj, self.dropout).transpose(0,1), Z_u_list[layer-1]))
        
        # bpr loss
        u_emb = self.E_u[uids]
        pos_emb = self.E_v[pos]
        neg_emb = self.E_v[neg]
        pos_scores = (u_emb * pos_emb).sum(-1)
        neg_scores = (u_emb * neg_emb).sum(-1)
        loss_bpr = -(pos_scores - neg_scores).sigmoid().log().mean()

        # pcl loss
        if self.lambda_1 == 0.0:
            loss_pcl = torch.tensor([0.0]).to(self.device)
        else:
            Z_u_norm = sum(Z_u_list)
            Z_i_norm = sum(Z_i_list)
            loss_pcl = self.cal_pcl_loss(uids, iids, Z_u_norm, Z_i_norm)

        # bcl loss
        if self.lambda_2 == 0.0:
            loss_bcl = torch.tensor([0.0]).to(self.device)
        else:
            weight_b = pos_scores.sigmoid()
            if self.is_norm:
                pos_scores_min = torch.min(pos_scores)
                pos_scores_max = torch.max(pos_scores)
                weight_b = (pos_scores - pos_scores_min) / (pos_scores_max - pos_scores_min + 1e-9)
            
            relations = (weight_b * self.n_b).to(torch.int64)
            relations = torch.where(relations>=self.n_b, self.n_b-1, relations)
            relations = torch.where(relations<0, 0, relations)
            x_u, x_i = self.E_u[uids], self.E_v[pos]
            edge_logits = torch.sigmoid(torch.mul(x_u, x_i))
            
            # behavior learning
            E_b = self.E_b
            if self.is_mean:
                ones = torch.ones_like(relations)
                counts = scatter_sum(ones, relations, dim=0)
                sum_T = scatter_sum(edge_logits, relations, dim=0)
                counts = counts.repeat(self.d, 1).t()
                E_b = sum_T / (counts + 1e-9)
            xb = E_b[relations]

            batch, n_b = edge_logits.shape[0], self.n_b
            relations_list = list(relations.cpu().numpy())
            index = [i for i in range(batch)]
            mask = torch.ones(batch, n_b).to(self.device)
            mask[index, relations_list] = 0

            neg_bcl = ((edge_logits @ E_b.T) * mask).mean(-1).mean()
            pos_bcl = (edge_logits * xb).sum(-1).mean()
            loss_bcl = neg_bcl - pos_bcl

        # reg loss
        loss_reg = 0
        for param in self.parameters():
            loss_reg += param.norm(2).square()
        loss_reg *= self.lambda_3
        
        # total loss
        loss = loss_bpr + self.lambda_1 * loss_pcl + self.lambda_2 * loss_bcl + loss_reg
        return loss, loss_bpr, self.lambda_1 * loss_pcl, self.lambda_2 * loss_bcl
    
    def predict(self, uids):
        preds = self.E_u[uids] @ self.E_v.T
        return preds
