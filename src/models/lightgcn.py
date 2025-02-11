import torch
from src.data.dataloader import BasicDataset
from torch import nn


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class LightGCN(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(LightGCN, self).__init__()

        self.config = config.model_args
        self.dataset: BasicDataset = dataset
        self.__init_weight() 

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["n_layers"]
        self.keep_prob = self.config["keep_prob"]
        self.A_split = self.config["A_split"]
        self.ssl_batch_size = self.config["ssl_batch_size"]
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )
        if self.config["pretrain"] == 0:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            print("use NORMAL distribution initilizer")
            
            # 추가: pretrain이 0이지만, meta embedding을 적용할 경우
            if self.config['use_meta_embedding'] == True:
                item_emb_path = self.config["item_emb_path"]

                if item_emb_path:
                    print(f"🔹 Loading item meta data embeddings from {item_emb_path} (pretrain=0)")
                    item_embeddings = np.load(item_emb_path) 
                    self.embedding_item.weight.data.copy_(torch.tensor(item_embeddings, dtype=torch.float32))
                else:
                    print("⚠ Warning: item_emb_path is not set, using default random embeddings.")

        else:
            # pretrain=1일 경우 기존 방식 유지
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            
            if self.config['use_meta_embedding'] == False:
                print("🔹 Using default pre-trained item embeddings")
                self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            else:
                item_emb_path = self.config["item_emb_path"]
                if item_emb_path:
                    print(f"🔹 Loading item meta data embeddings from {item_emb_path}")
                    item_embeddings = np.load(item_emb_path)
                    self.embedding_item.weight.data.copy_(torch.tensor(item_embeddings, dtype=torch.float32))
                else:
                    print("⚠ Warning: item_emb_path is not set, using default random embeddings.")

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"LightGCN is already to go (dropout:{self.config['dropout']})")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __degree_aware_edge_dropout(self, graph, keep_prob, threshold=10, max_drop_prob=0.5, min_drop_prob=0.2):
        """
        아이템의 차수(Degree)에 따라 Edge Drop 확률을 다르게 적용하는 함수.
        - graph: 원본 Sparse Graph
        - keep_prob: 전체 평균 Edge 유지 비율
        - threshold: 특정 차수 이하인 경우 Edge Drop을 하지 않음.
        - max_drop_prob: 최대 Drop 확률 (기본값 50%)
        - min_drop_prob: 최소 Drop 확률 (기본값 10%)
        """
        device = graph.device
        # 아이템 차수(Degree) 계산
        item_degrees = torch.bincount(graph.indices()[1]).to(
            device
        )  # 아이템(1) 차수 계산
        max_degree = item_degrees.max()

        drop_base = min_drop_prob + (max_drop_prob - min_drop_prob) * (1 - (item_degrees / max_degree))
        item_drop_probs = torch.where(item_degrees <= threshold, 0.0, drop_base)  # 차수가 threshold 이하인 경우 Drop 확률 0%

        # 엣지마다 Drop 확률 적용
        edge_mask = (
            torch.rand(graph._nnz(), device=device)
            > item_drop_probs[graph.indices()[1]]
        )
        # 새로운 그래프 생성 (Edge Drop이 반영됨)
        new_graph = torch.sparse.FloatTensor(
            graph.indices()[:, edge_mask], graph.values()[edge_mask], graph.size()
        )

        return new_graph

    def __dropout(self, keep_prob):
        if self.config["use_ssl"]:
            # Degree-aware Edge Dropout 적용
            if self.A_split:
                graph = []
                for g in self.Graph:
                    graph.append(self.__degree_aware_edge_dropout(g, keep_prob))
            else:
                graph = self.__degree_aware_edge_dropout(self.Graph, keep_prob)
        else:
            if self.A_split:
                graph = []
                for g in self.Graph:
                    graph.append(self.__dropout_x(g, keep_prob))
            else:
                graph = self.__dropout_x(self.Graph, keep_prob)

        return graph

    def computer(self, dropped=False):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config["dropout"] or dropped:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):

        if self.config["use_ssl"]:
            # Contrastive Learning을 위한 두 개의 증강된 뷰 생성
            users_1, items_1 = self.computer(dropped=True)
            users_2, items_2 = self.computer(dropped=True)

            users_emb = users_1[users]
            pos_emb = items_1[pos]
            neg_emb = items_1[neg]
            userEmb0 = users_emb.detach()
            posEmb0 = pos_emb.detach()
            negEmb0 = neg_emb.detach()
            # Contrastive Loss 계산
            loss_ssl = self.__contrastive_loss(
                users_1, users_2, batch_size=self.ssl_batch_size
            ) + self.__contrastive_loss(items_1, items_2, self.ssl_batch_size)

        else:
            (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = (
                self.getEmbedding(users.long(), pos.long(), neg.long())
            )
            loss_ssl = 0  # 기존 방식에서는 Contrastive Loss 없음

        reg_loss = (
            (1 / 2)
            * (
                userEmb0.norm(2).pow(2)
                + posEmb0.norm(2).pow(2)
                + negEmb0.norm(2).pow(2)
            )
            / float(len(users))
        )

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss + self.config["ssl_lambda"] * loss_ssl, reg_loss

    def __contrastive_loss(self, z1, z2, temperature=0.5, batch_size=1024):
        """
        Contrastive Loss (InfoNCE Loss)
        - z1, z2: 두 개의 서로 다른 증강된 그래프에서 얻은 노드 임베딩
        - temperature: Contrastive Learning에서 사용하는 온도(temperature) 파라미터
        """
        f = lambda x: torch.exp(x / temperature)
        # 같은 노드의 서로 다른 뷰 간 유사도 (Positive Pair)
        pos_sim = torch.sum(torch.mul(z1, z2), dim=1)
        # Negative Pair를 메모리 효율적으로 계산
        num_nodes = z1.shape[0]
        total_loss = 0.0
        for i in range(0, num_nodes, batch_size):
            z1_batch = z1[i : i + batch_size]
            z2_batch = z2[i : i + batch_size]
            between_sim = torch.matmul(z1_batch, z2_batch.T)
            neg_loss = torch.logsumexp(between_sim, dim=1)  # OOM 방지

            total_loss += torch.sum(torch.log(pos_sim[i:i+batch_size]) - neg_loss)

        return -total_loss / num_nodes

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
