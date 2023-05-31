import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from info_nce import InfoNCE
from utils import ddp_all_gather
from .base import BaseLoss, gather_and_scale_wrapper


class newNCE(BaseLoss):

    def __init__(self, batch_size, loss_term_weight=1.0):
        super(newNCE, self).__init__(loss_term_weight)
        self.batch_size_ = batch_size   #a list eg:[8,16]
        self.new_infonce_loss_git = InfoNCE(negative_mode='unpaired')
        # self.new_infonce_loss_git = InfoNCE()
        self.negative_label = True
        self.loss_info = {}

    # @gather_and_scale_wrapper
    def forward(self, embeddings, embeddings_da, labels):
        labels = ddp_all_gather(labels)
        embeddings = ddp_all_gather(embeddings)
        embeddings_da = ddp_all_gather(embeddings_da)
        import pdb;pdb.set_trace()
        # import pdb; pdb.set_trace()
        # embeddings: [n, p, c], label: [n]
        embeddings, embeddings_da = self.normalize(embeddings, embeddings_da)
        bs = embeddings.shape[0]
        embeddings = embeddings.reshape(bs, -1).float()
        embeddings_da = embeddings_da.reshape(bs, -1).float()
        embeddings = torch.cat([embeddings, embeddings_da], dim=0)
        labels = torch.cat([labels, labels], dim=0)

        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
        dist = dist.unsqueeze(0)
        positive_logit, negative_logits = self.Convert2Triplets(labels, ref_label, dist)
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros((len(logits)), dtype=torch.long, device=embeddings.device)

        temperature=0.1
        new_infonce_loss = F.cross_entropy(logits / temperature, labels, reduction='mean')

        self.loss_info['scalar/new/infonce_loss'] = new_infonce_loss.mean(dim=0)
        return new_infonce_loss, self.loss_info

    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        dist = torch.einsum('nc,kc->nk', x, y)
        # x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        # y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        # inner = x.matmul(y.transpose(-1, -2))  # [p, n_x, n_y]
        # dist = x2 + y2 - 2 * inner
        # dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).byte()  # [n_r, n_c]
        diffenc = matches ^ 1  # [n_r, n_c]
        mask = matches.unsqueeze(2) * diffenc.unsqueeze(1)
        a_idx, p_idx, n_idx = torch.where(mask)

        bs = matches.shape[0]
        rest = 256

        n_idx = n_idx.reshape(bs, -1)[:, :rest].reshape(-1)
        a_idx = a_idx.reshape(bs, -1)[:, :rest].reshape(-1)
        p_idx = p_idx.reshape(bs, -1)[:, :rest].reshape(-1)

        # import pdb; pdb.set_trace()
        n_idx = n_idx.reshape(bs, -1)
        n_idx = n_idx.unsqueeze(1).repeat(1, bs, 1).view(-1, bs).view(-1)

        a2_idx = a_idx.unsqueeze(1).repeat(1, bs).view(-1)
        ap_dist = dist[:, a_idx, p_idx].transpose(1, 0)
        an_dist = dist[:, a2_idx, n_idx].transpose(1, 0).reshape(-1, bs)


        return ap_dist, an_dist



