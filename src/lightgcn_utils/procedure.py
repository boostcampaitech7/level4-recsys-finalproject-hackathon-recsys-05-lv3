from src.lightgcn_utils import utils
import numpy as np
import torch
from src.models import lightgcn
import multiprocessing

import src.lightgcn_utils.metrics as metric_module 



METRIC_NAMES = {
    'precision': 'Precision_atK',
    'recall': 'Recall_atK',
    'ndcg': 'NDCG_atK'
}



CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(args,dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with utils.timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(args.device)
    posItems = posItems.to(args.device)
    negItems = negItems.to(args.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // args.dataloader['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(args,users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=args.dataloader['bpr_batch_size'])):
        if batch_i % 50==0:
            print(f'{batch_i} / {total_batch}')
        # 역전파
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if args.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / args.dataloader['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = utils.timer.dict()
    utils.timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}", aver_loss
    

# 수정해야함 
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = metric_module.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in [20]:
        # for metric in args.metrics :
        #     metric_fn = getattr(metric_module,METRIC_NAMES[metric])().to(args.device)
            

        ret = metric_module.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(metric_module.NDCG_atK(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(args,dataset, Recmodel, epoch, w=None):
    u_batch_size = args.dataloader['test_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: lightgcn.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(args.topks)
    if args.model_args['multicore'] == 1:
        pool = multiprocessing.Pool(CORES)

    results = {}

    for name in args.metrics :
        results[name] = np.zeros(len(args.topks))

    print(results)
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
        for batch_users in utils.minibatch(args,users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(args.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
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
        if args.model_args['multicore'] == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if args.tensorboard:
            w.add_scalars(f'Test/Recall@{args.topks}',
                          {str(args.topks[i]): results['recall'][i] for i in range(len(args.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{args.topks}',
                          {str(args.topks[i]): results['precision'][i] for i in range(len(args.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{args.topks}',
                          {str(args.topks[i]): results['ndcg'][i] for i in range(len(args.topks))}, epoch)
        if args.model_args['multicore'] == 1:
            pool.close()
        print(results)
        return results
