import numpy as np
from src.data.dataloader import BasicDataset
from sklearn.metrics import roc_auc_score


def recall_at_k(test_data, r, k):
    """
    Recall@K
    """
    right_pred = r[:, :k].sum(1)  
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])  
    recall = np.sum(right_pred / recall_n)  
    return recall


def precision_at_k(test_data, r, k):
    """
    Precision@K
    """
    right_pred = r[:, :k].sum(1)  
    precis_n = k  
    precision = np.sum(right_pred) / (precis_n * len(test_data))  
    return precision


def ndcg_at_k(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def auc(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def hit_rate_at_k(test_data, r, k):
    """
    Hit Rate@K: 추천된 아이템 중 사용자가 본 아이템이 하나라도 있는지 확인
    Args:
        test_data: List of ground truth items for each user
        r: Binary matrix (1 if item is in test_data, 0 otherwise)
        k: Top-K 값
    
    Returns:
        hit_rate: Hit Rate@K 값
    """
    hits = (r[:, :k].sum(1) > 0).astype(int)  # 추천 리스트에 정답이 하나라도 있으면 1, 없으면 0
    hit_rate = np.mean(hits)  # 전체 사용자 평균
    return hit_rate


def mrr_at_k(test_data, r, k):
    """
    Mean Reciprocal Rank (MRR@K)
    """
    pred_data = r[:, :k]
    ranks = np.arange(1, k+1)  # 단순한 순위 값 (1, 2, 3, ... K)
    
    # 첫 번째로 맞춘 정답의 Reciprocal Rank 값만 유지
    rr = pred_data / ranks  # log 대신 1/rank 사용!
    
    # 사용자별 MRR을 계산 (첫 번째로 맞춘 정답만 고려)
    rr = rr.max(axis=1)  # 가장 높은 Reciprocal Rank 값만 선택
    return np.mean(rr)  # 전체 평균 MRR 반환


def get_label(test_data, predictions):
    labels = np.zeros_like(predictions, dtype=np.float32)

    for i, ground_truth in enumerate(test_data):
        labels[i] = np.isin(predictions[i], ground_truth).astype(np.float32)

    return labels

