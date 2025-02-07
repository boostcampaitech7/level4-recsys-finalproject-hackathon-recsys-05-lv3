import torch
import torch.optim as optim
import torch.nn.functional as F
import sys

class BPRLoss:
    def __init__(self,
                 recmodel,
                 args):
        self.config = args.optimizer
        self.model = recmodel
        self.weight_decay = self.config.args['weight_decay']
        self.lr = self.config.args['lr']
        optimizer_class  = getattr(optim,self.config.type)
        self.opt = optimizer_class(self.model.parameters(), lr=self.lr)
        # self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

        self.args = args

    def predict(self, users, pos, neg):
        
        if self.args.model == "CLCRec":
            loss, reg_loss = self.model.clc_loss(users, pos, neg)
        else:
            loss, reg_loss = self.model.bpr_loss(users, pos, neg)
            
        reg_loss = reg_loss*self.weight_decay
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
        self.reg_weight = self.config.args.get('reg_weight', 0.000001)  # Graph Regularization 가중치 추가
        self.lr = self.config.args['lr']

        optimizer_class = getattr(optim, self.config.type)  # 가변적인 Optimizer 사용
        self.opt = optimizer_class(recmodel.parameters(), lr=self.lr)

    def predict(self, users, pos, neg):
        """
        BPR Loss + Graph Regularization Loss 적용.
        """
        # 기존 BPR Loss 계산
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay  # 기존 정규화

        # Graph Regularization Loss 계산
        graph_reg_loss = self.graph_regularization_loss(self.model)
        graph_reg_loss = graph_reg_loss * self.reg_weight  # 가중치 적용

        # 최종 Loss 계산
        total_loss = loss + reg_loss + graph_reg_loss

        # 역전파 및 최적화
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()
        
        # 디버깅용
        # print(f"pos: {pos[:5]}, neg: {neg[:5]}")
        # print(f"users_emb[:5]: {users_emb[:5]}")
        # print(f"pos_emb[:5]: {pos_emb[:5]}")
        # print(f"neg_emb[:5]: {neg_emb[:5]}")
        # sys.stdout.flush()
        print(f"🔹 total_loss: {total_loss.item()}, loss: {loss.item()}, reg_loss: {reg_loss.item()}, graph_reg_loss: {graph_reg_loss.item()}")
        sys.stdout.flush()


        return total_loss.cpu().item()

    def graph_regularization_loss(self, model):
        """
        Graph Regularization Loss 계산.
        L_reg = Σ w_ij ||E_i - E_j||^2
        """
        item_embeddings = model.embedding_item.weight  # 아이템 임베딩 가져오기
        similarity_matrix = self.compute_similarity_matrix(item_embeddings)  # 유사도 행렬 계산
        # 디버깅: 크기 확인
        # print(f"item_embeddings.shape: {item_embeddings.shape}")  # 예상: (3533, 64)
        # print(f"similarity_matrix.shape: {similarity_matrix.shape}")  # 예상: (3533, 3533)
        # sys.stdout.flush()
        # Debug: 유사도 행렬 값 확인
        print(f"similarity_matrix min: {similarity_matrix.min().item()}, max: {similarity_matrix.max().item()}")
        print(f"item_embeddings min: {item_embeddings.min().item()}, max: {item_embeddings.max().item()}")
        sys.stdout.flush()

        # 크기 조정 후 계산
        reg_loss = torch.sum(similarity_matrix.unsqueeze(-1) * (item_embeddings.unsqueeze(1) - item_embeddings.unsqueeze(0)).pow(2))
        # reg_loss = torch.sum(similarity_matrix * (item_embeddings.unsqueeze(1) - item_embeddings.unsqueeze(0)).pow(2))

        # Debug: reg_loss 값 확인
        print(f"graph_reg_loss (before scaling): {reg_loss.item()}")
        sys.stdout.flush()
        
        return reg_loss

    def compute_similarity_matrix(self, embeddings):
        """
        Cosine Similarity 기반으로 아이템 간 유사도 행렬 계산.
        """
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)  # 정규화
        similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.t())  # Cosine Similarity 계산

        # 디버깅: 음수 값 방지 (Cosine Similarity는 -1 ~ 1 범위이므로, 0 이상으로 설정)
        similarity_matrix = torch.clamp(similarity_matrix, min=0)

        return similarity_matrix