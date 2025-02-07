import torch
from torch import nn
from src.models.lightgcn import LightGCN

class CLCRec(LightGCN):
    def __init__(self, config: dict, dataset):
        super(CLCRec, self).__init__(config, dataset)
        self.tau = self.config['tau']  
        self.contrastive_weight = self.config['contrastive_weight']  

    def contrastive_loss(self, user_emb, pos_emb, neg_emb):
        user_emb = nn.functional.normalize(user_emb, dim=1)
        pos_emb = nn.functional.normalize(pos_emb, dim=1)
        neg_emb = nn.functional.normalize(neg_emb, dim=1)

        pos_similarity = torch.sum(user_emb * pos_emb, dim=1) / self.tau

        neg_similarity = torch.matmul(user_emb, neg_emb.T) / self.tau

        logits = torch.cat([pos_similarity.unsqueeze(1), neg_similarity], dim=1)

        labels = torch.zeros(user_emb.size(0), dtype=torch.long).to(user_emb.device)

        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

    def clc_loss(self, users, pos_items, neg_items):
        users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0 = self.getEmbedding(users.long(), pos_items.long(), neg_items.long())

        bpr_loss, reg_loss = super().bpr_loss(users, pos_items, neg_items)

        contrastive_loss = self.contrastive_loss(users_emb, pos_emb, neg_emb)

        total_loss = bpr_loss + self.contrastive_weight * contrastive_loss

        return total_loss, reg_loss

    def forward(self, users, items):
        return super().forward(users, items)
