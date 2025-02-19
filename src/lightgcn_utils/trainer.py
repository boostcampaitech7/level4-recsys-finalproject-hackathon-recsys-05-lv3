from src.lightgcn_utils import utils
import numpy as np
import torch
from src.models import lightgcn
import multiprocessing
import src.lightgcn_utils.metrics as metric_module 


METRIC_NAMES = {
    'precision': 'precision_at_k',
    'recall': 'recall_at_k',
    'ndcg': 'ndcg_at_k',
    'mrr': 'mrr_at_k',
    'hr': 'hit_rate_at_k'
}

class Trainer :
    def __init__(self, args, dataset, model, loss, w) :
        self.args = args
        self.dataset = dataset
        self.model = model
        self.loss = loss
        self.w = w
        self.popular_items = self.args.popular_items if hasattr(self.args, "popular_items") else []
        self.neg_ratio = self.args.dataloader.neg_ratio
        
    def train(self) : 
        self.model.train()
        
        with utils.timer(name="Sample"):
            S = utils.Negative_Sampling(self.dataset)
        print("Negative Sampling Complete")
        users = torch.Tensor(S[:, 0]).long()
        posItems = torch.Tensor(S[:, 1]).long()
        negItems = torch.Tensor(S[:, 2]).long()

        users = users.to(self.args.device)
        posItems = posItems.to(self.args.device)
        negItems = negItems.to(self.args.device)

        users, posItems, negItems = utils.shuffle(users, posItems, negItems)
        total_batch = len(users) // self.args.dataloader['bpr_batch_size'] + 1
        aver_loss = 0.
        
        for (batch_i,
            (batch_users,
            batch_pos,
            batch_neg)) in enumerate(utils.minibatch(self.args,users,
                                                    posItems,
                                                    negItems,
                                                    batch_size=self.args.dataloader['bpr_batch_size'])):
            if batch_i % self.args.train.show_interval== 0:
                print(f'{batch_i} / {total_batch}')
            # 역전파
            cri = self.loss.predict(batch_users, batch_pos, batch_neg)
            aver_loss += cri

            if self.args.tensorboard:
                self.w.add_scalar(f'BPRLoss/BPR', cri, self.args.train.epochs * int(len(users) / self.args.dataloader['bpr_batch_size']) 
                                  + batch_i)
                
        aver_loss = aver_loss / total_batch
        time_info = utils.timer.dict()
        utils.timer.zero()

        return f"loss{aver_loss:.3f}-{time_info}", aver_loss
    


    def _test_one_batch(self,X):
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        r = metric_module.get_label(groundTrue, sorted_items)
        metrics_result = {metric: [] for metric in self.args.metrics}  # 딕셔너리로 초기화
        for k in self.args.topks:
            for metric in self.args.metrics :
                metric_fn = getattr(metric_module,METRIC_NAMES[metric])
                metrics_result[metric].append(metric_fn(groundTrue, r, k))
            
        return {metric: np.array(values) for metric, values in metrics_result.items()}
                
    def test(self):
        u_batch_size = self.args.dataloader['test_batch_size']
        testDict  = self.dataset.testDict
        # eval mode with no dropout
        self.model = self.model.eval()
        max_K = max(self.args.topks)
        if self.args.model_args['multicore'] == 1:
            pool = multiprocessing.Pool(self.args.CORES)

        metrics = self.args.metrics 
        results = {metric: np.zeros(len(self.args.topks)) for metric in metrics}

        with torch.no_grad():
            users = list(testDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            # auc_record = []
            # ratings = []
            total_batch = len(users) // u_batch_size + 1
            for batch_users in utils.minibatch(self.args,users, batch_size=u_batch_size):
                allPos = self.dataset.getUserPosItems(batch_users)
                groundTrue = [testDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(self.args.device)

                rating = self.model.getUsersRating(batch_users_gpu)
                #rating = rating.cpu()
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1<<10)
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                # aucs = [ 
                #         utils.AUC(rating[i],
                #                   dataset, 
                #                   test_data) for i, test_data in enumerate(groundTrue)
                #     ]
                # auc_record.extend(aucs)
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
            assert total_batch == len(users_list)
            X = zip(rating_list, groundTrue_list)
            if self.args.model_args['multicore'] == 1:
                pre_results = pool.map(self._test_one_batch, X)
            else:
                pre_results = []
                for x in X:
                    pre_results.append(self._test_one_batch(x))
            scale = float(u_batch_size/len(users))
            for result in pre_results:
                for metric in metrics:
                    results[metric] += result[metric]
            for metric in metrics:
                results[metric] /= float(len(users))

            # results['auc'] = np.mean(auc_record)
            if self.args.tensorboard:
                for metric in metrics:
                    self.w.add_scalars(f"Test/{metric.capitalize()}@{self.args.topks}",
                                {str(self.args.topks[i]): results[metric][i] for i in range(len(self.args.topks))},
                                self.args.train.epochs)
            if self.args.model_args['multicore'] == 1:
                pool.close()
            print(results)
            return results
        
    def test_cold(self):
            u_batch_size = self.args.dataloader['test_cold_batch_size']
            testDict  = self.dataset.coldDict

            # eval mode with no dropout
            self.model = self.model.eval()
            max_K = max(self.args.topks)
            if self.args.model_args['multicore'] == 1:
                pool = multiprocessing.Pool(self.args.CORES)

            metrics = self.args.metrics 
            results = {metric: np.zeros(len(self.args.topks)) for metric in metrics}

            with torch.no_grad():
                users = list(self.dataset.coldDict.keys())
                try:
                    assert u_batch_size <= len(users) / 10
                except AssertionError:
                    print(f"test_cold_batch_size is too big for this dataset, try a small one {len(users) // 10}")
                users_list = []
                rating_list = []
                groundTrue_list = []
                # auc_record = []
                # ratings = []

                if len(users) % u_batch_size == 0:
                    total_batch = len(users) // u_batch_size 
                else:
                    total_batch = len(users) // u_batch_size + 1

                for batch_users in utils.minibatch(self.args,users, batch_size=u_batch_size):
                    allPos = self.dataset.getUserPosItems(batch_users)
                    groundTrue = [testDict[u] for u in batch_users]
                    batch_users_gpu = torch.Tensor(batch_users).long()
                    batch_users_gpu = batch_users_gpu.to(self.args.device)

                    rating = self.model.getUsersRating(batch_users_gpu)
                    #rating = rating.cpu()
                    exclude_index = []
                    exclude_items = []
                    for range_i, items in enumerate(allPos):
                        exclude_index.extend([range_i] * len(items))
                        exclude_items.extend(items)
                    rating[exclude_index, exclude_items] = -(1<<10)
                    _, rating_K = torch.topk(rating, k=max_K)
                    rating = rating.cpu().numpy()
                    # aucs = [ 
                    #         utils.AUC(rating[i],
                    #                   dataset, 
                    #                   test_data) for i, test_data in enumerate(groundTrue)
                    #     ]
                    # auc_record.extend(aucs)
                    del rating
                    users_list.append(batch_users)
                    rating_list.append(rating_K.cpu())
                    groundTrue_list.append(groundTrue)
                assert total_batch == len(users_list)
                X = zip(rating_list, groundTrue_list)

                if self.args.model_args['multicore'] == 1:
                    pre_results = pool.map(self._test_one_batch, X)
                else:
                    pre_results = []
                    for x in X:
                        pre_results.append(self._test_one_batch(x))
                    
                scale = float(u_batch_size/len(users))
                for result in pre_results:
                    for metric in metrics:
                        results[metric] += result[metric]
                for metric in metrics:
                    results[metric] /= float(len(users))

                # results['auc'] = np.mean(auc_record)
                if self.args.tensorboard:
                    for metric in metrics:
                        self.w.add_scalars(f"Test/{metric.capitalize()}@{self.args.topks}",
                                    {str(self.args.topks[i]): results[metric][i] for i in range(len(self.args.topks))},
                                    self.args.train.epochs)
                if self.args.model_args['multicore'] == 1:
                    pool.close()
                print(results)
                return results


class Inference :
    def __init__(self, args, dataset, model) :
        self.args = args
        self.dataset = dataset
        self.model = model

    def run_inference(self):
        u_batch_size = self.args.dataloader['test_batch_size']
        testDict  = self.dataset.testDict
        # eval mode with no dropout
        self.model = self.model.eval()
        max_K = max(self.args.topks)

        with torch.no_grad():
            users = list(testDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"inference_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []

            for batch_users in utils.minibatch(self.args,users, batch_size=u_batch_size):
                allPos = self.dataset.getUserPosItems(batch_users)
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(self.args.device)

                rating = self.model.getUsersRating(batch_users_gpu)
                #rating = rating.cpu()
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1<<10)
                _, rating_K = torch.topk(rating, k=max_K)

                users_list.extend(batch_users)
                rating_list.append(rating_K.cpu())
        
        return np.array(users_list).reshape(-1,1), np.array(torch.cat(rating_list, dim=0))