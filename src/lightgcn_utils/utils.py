import os

import numpy as np
import torch
from time import time

from src.data.dataloader import BasicDataset


def Negative_Sampling(dataset, neg_ratio = 1):
    dataset : BasicDataset
    allPos = dataset.allPos
    start = time()
    if dataset.neg_sampling_strategy == "popular":
        S = Sampling_by_popular(dataset)
    elif dataset.neg_sampling_strategy == "cold":
        S = Sampling_by_cold(dataset)
    elif dataset.neg_sampling_strategy == "random":
        try:
            from cppimport import imp_from_filepath
            import sys
            sys.path.append('src/data')
            sampling = imp_from_filepath('src/data/sampling.cpp')
            sampling.seed(42)
            S = sampling.sample_negative(dataset.n_users, dataset.m_items,dataset.trainDataSize, allPos, neg_ratio)
            print("Cpp extension loaded")
        except:
            print("Cpp extension not loaded")
            S = Sampling_randomly(dataset)
    else:
        raise ValueError(f"Invalid neg_sampling strategy: {dataset.neg_sampling_strategy}")
    
    return S


def Sampling_randomly(dataset):
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)


def Sampling_by_popular(dataset):
    """ 인기 아이템 기반 Negative Sampling """
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    popular_set = set(dataset.popular_items)
    
    samples = []
    for i, user in enumerate(users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue

        positem = np.random.choice(posForUser)
        posForUser_set = set(posForUser)
        candidates = list(popular_set - posForUser_set)

        negitem = np.random.choice(candidates) if candidates else np.random.choice(list(popular_set))

        samples.append([user, positem, negitem])
    
    return np.array(samples)


def Sampling_by_cold(dataset, prob_cold=0.3):
    """
    콜드 아이템을 네거티브로 뽑을 확률(prob_cold)를 높이는 샘플러
    """
    cold_items = set(dataset.coldItem)

    total_start = time()
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    for user in users:
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        # Positive 아이템 하나 뽑기
        positem = np.random.choice(posForUser)

        # Negative 아이템 뽑기 (콜드 아이템에 가중치)
        while True:
            if np.random.rand() < prob_cold:
                # 콜드 아이템 중 랜덤
                negitem = np.random.choice(list(cold_items))
            else:
                # 전체 아이템 중 랜덤
                negitem = np.random.randint(0, dataset.m_items)
            
            # 만약 그 아이템이 user의 Pos 리스트에 있다면 계속 다시 뽑기
            if negitem not in posForUser:
                break
        S.append([user, positem, negitem])
    return np.array(S)


# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName(args):

    file = f'{args.model}-{args.model_experiment_name}.pth.tar'   

    return os.path.join(args.FILE_PATH,file)
    
def minibatch(args,*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', args.dataloader['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)

