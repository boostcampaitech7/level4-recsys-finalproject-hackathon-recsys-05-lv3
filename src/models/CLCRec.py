import torch
from torch import nn
from src.models.lightgcn import LightGCN

class CLCRec(LightGCN):
    def __init__(self, config: dict, dataset):
        super(CLCRec, self).__init__(config, dataset)
        self.tau = self.config['tau']  # Contrastive temperature parameter
        self.contrastive_weight = self.config['contrastive_weight']  # Weight for contrastive loss

    def contrastive_loss(self, user_emb, pos_emb, neg_emb):
        """
        Computes the contrastive loss.
        Args:
            user_emb: User embeddings (batch_size x latent_dim)
            pos_emb: Positive item embeddings (batch_size x latent_dim)
            neg_emb: Negative item embeddings (batch_size x latent_dim)
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        user_emb = nn.functional.normalize(user_emb, dim=1)
        pos_emb = nn.functional.normalize(pos_emb, dim=1)
        neg_emb = nn.functional.normalize(neg_emb, dim=1)

        # Positive similarity
        pos_similarity = torch.sum(user_emb * pos_emb, dim=1) / self.tau

        # Negative similarity
        neg_similarity = torch.matmul(user_emb, neg_emb.T) / self.tau

        # Compute logits
        logits = torch.cat([pos_similarity.unsqueeze(1), neg_similarity], dim=1)

        # Labels: positive samples are at index 0
        labels = torch.zeros(user_emb.size(0), dtype=torch.long).to(user_emb.device)

        # Contrastive loss using CrossEntropy
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

    def clc_loss(self, users, pos_items, neg_items):
        """
        Combines BPR loss and contrastive loss.
        Args:
            users: Batch of user indices
            pos_items: Batch of positive item indices
            neg_items: Batch of negative item indices
        Returns:
            Total loss and regularization loss
        """
        # Get embeddings for users and items
        users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0 = self.getEmbedding(users.long(), pos_items.long(), neg_items.long())

        # BPR Loss (from LightGCN)
        bpr_loss, reg_loss = super().bpr_loss(users, pos_items, neg_items)

        # Contrastive Loss
        contrastive_loss = self.contrastive_loss(users_emb, pos_emb, neg_emb)

        # Combine losses with weights
        total_loss = bpr_loss + self.contrastive_weight * contrastive_loss

        return total_loss, reg_loss

    def forward(self, users, items):
        """
        Forward pass for prediction.
        Args:
            users: User indices
            items: Item indices
        Returns:
            Predicted scores for user-item pairs.
        """
        return super().forward(users, items)
