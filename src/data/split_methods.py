import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from typing import List, Tuple


def train_test_split_strategy(
    grouped: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[List[Tuple[int, List[int]]], List[Tuple[int, List[int]]]]:
    """기본 Train-Test 분할 방식"""
    train_data = []
    test_data = []

    for row in grouped.itertuples():
        user_id = row.newUserId
        item_list = row.newItemId
        if len(item_list) <= 1:
            train_data.append((user_id, item_list))
        else:
            train_part, test_part = train_test_split(
                item_list, test_size=test_size, random_state=random_state
            )
            train_data.append((user_id, train_part))
            test_data.append((user_id, test_part))

    return train_data, test_data


def leave_one_out(grouped: pd.DataFrame) -> Tuple[
    List[Tuple[int, List[int]]],
    List[Tuple[int, List[int]]],
]:
    """Leave-One-Out 방식"""
    train_data = []
    test_data = []
    cold_idx = []

    for idx, row in grouped.iterrows():
        user_id = row["newUserId"]
        item_list = row["newItemId"]

        if len(item_list) >= 3:
            if 3 <= len(item_list) <= 15:
                cold_idx.append(user_id)
            train_part = item_list[:-1]
            test_part = [item_list[-1]]
            train_data.append((user_id, train_part))
            test_data.append((user_id, test_part))
        else:
            train_data.append((user_id, item_list))

    return train_data, test_data, cold_idx


# timestamp 추가 필요


def k_fold(
    grouped: pd.DataFrame, k: int = 5
) -> Tuple[List[Tuple[int, List[int]]], List[Tuple[int, List[int]]]]:
    """K-Fold 방식으로 데이터 분할"""
    train_data = []
    test_data = []

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for row in grouped.itertuples():
        user_id = row.newUserId
        item_list = row.newItemId

        if len(item_list) < k:
            train_data.append((user_id, item_list))
            continue

        for train_idx, test_idx in kf.split(item_list):
            train_part = [item_list[i] for i in train_idx]
            test_part = [item_list[i] for i in test_idx]
            train_data.append((user_id, train_part))
            test_data.append((user_id, test_part))

    return train_data, test_data
