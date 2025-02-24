import torch
import torch.optim as optim
import torch.nn.functional as F
import sys


class BPRLoss:
    def __init__(self, recmodel, args):
        self.config = args.optimizer
        self.model = recmodel
        self.weight_decay = self.config.args["weight_decay"]
        self.lr = self.config.args["lr"]
        optimizer_class = getattr(optim, self.config.type)
        self.opt = optimizer_class(self.model.parameters(), lr=self.lr)
        # self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

        self.args = args

    def predict(self, users, pos, neg):

        if self.args.model == "CLCRec":
            loss, reg_loss = self.model.clc_loss(users, pos, neg)
        else:
            loss, reg_loss = self.model.bpr_loss(users, pos, neg)

            cold_items = set(cold_items['newItemId'])
            loss, reg_loss = self.model.bpr_loss_cold_weight(users, pos, neg, cold_items, alpha=1)

        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()

class BPRLossWithReg:
    def __init__(self, recmodel, args):
        self.config = args.optimizer
        self.model = recmodel
        self.weight_decay = self.config.args['weight_decay'] 
        self.reg_weight = self.config.args['reg_weight']
        self.lr = self.config.args['lr']

        optimizer_class = getattr(optim, self.config.type)
        self.opt = optimizer_class(recmodel.parameters(), lr=self.lr)

        self.similarity_matrix = self._compute_similarity_matrix(self.model.embedding_item.weight).detach().cpu()

    def predict(self, users, pos, neg):
        """
        BPR Loss + Graph Regularization Loss 적용.
        """
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay

        graph_reg_loss = self._graph_regularization_loss(self.model)
        graph_reg_loss = graph_reg_loss * self.reg_weight

        total_loss = loss + reg_loss + graph_reg_loss

        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        return total_loss.cpu().item()

    def _graph_regularization_loss(self, model, batch_size=128):
        """
        Graph Regularization Loss 계산 (메모리 최적화 버전).
        """
        item_embeddings = model.embedding_item.weight  
        num_items = item_embeddings.shape[0]
        reg_loss = torch.tensor(0.0, device="cuda")  

        for i in range(0, num_items, batch_size):
            batch_embeddings = item_embeddings[i : i + batch_size].to("cuda")  
            diff = torch.cdist(batch_embeddings, item_embeddings.to("cuda"), p=2).pow(2)  
            
            similarity_batch = self.similarity_matrix[i : i + batch_size].to("cuda")
            reg_loss += torch.sum(similarity_batch * diff)

        return reg_loss


    def _compute_similarity_matrix(self, embeddings, threshold=0.1):
        """
        Cosine Similarity 기반으로 아이템 간 유사도 행렬 계산 (Sparse 적용).
        """
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.t())

        similarity_matrix = torch.where(similarity_matrix > threshold, similarity_matrix, torch.tensor(0.0, device=similarity_matrix.device))

        return similarity_matrix


class BPRLoss_with_coldweight:
    def __init__(self, recmodel, args):
        self.config = args.optimizer
        self.model = recmodel
        self.weight_decay = self.config.args["weight_decay"]
        self.lr = self.config.args["lr"]
        optimizer_class = getattr(optim, self.config.type)
        self.opt = optimizer_class(self.model.parameters(), lr=self.lr)

        self.args = args

    def predict(self, users, pos, neg):
        cold_items = set(self.model.dataset.coldItem)
        loss, reg_loss = self.bpr_loss_cold_weight(users, pos, neg, cold_items, alpha=1)

        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()
    
    def bpr_loss_cold_weight(self, users, pos, neg, cold_items, alpha=0.5):
        """
        cold_items에 속한 pos item에 대해 추가 가중치 alpha를 부여여
        alpha=0.5라면, cold pos일 때 loss를 1.5배(기본 1 + alpha)로 만듦
        """
        users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0 = self.model.getEmbedding(users.long(), pos.long(), neg.long())
        
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                        posEmb0.norm(2).pow(2)  +
                        negEmb0.norm(2).pow(2))/float(len(users))

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)  # (batch,)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)  # (batch,)

        bpr_raw = torch.nn.functional.softplus(neg_scores - pos_scores)  # (batch,)

        pos_mask = torch.tensor(
            [1.0 if p.item() in cold_items else 0.0 for p in pos],
            dtype=torch.float,
            device=pos.device
        )

        pos_factor = 1.0 + alpha * pos_mask
        bpr_weighted = bpr_raw * pos_factor

        loss = torch.mean(bpr_weighted)

        return loss, reg_loss