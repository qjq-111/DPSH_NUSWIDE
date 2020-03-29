import torch
import time
from functools import wraps
import numpy as np


def timing(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        print(f'total time = {time.time() - start:.4f}')
        return ret
    return wrapper


@timing
def compute_result(dataloader, net, use_gpu):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in dataloader:
        clses.append(cls)
        if use_gpu:
            img = img.cuda()
        bs.append(net(img).data.cpu())
    return torch.sign(torch.cat(bs)), torch.cat(clses)


def Logtrick(x, use_gpu):
    if use_gpu:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, torch.FloatTensor([0.]).cuda())
    else:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, torch.FloatTensor([0.]))
    return lt


# 计算mAP指标
@timing
def compute_mAP(trn_binary, trn_label, tst_binary, tst_label, whole=True, k=0):
    for x in trn_binary, trn_label, tst_binary, tst_label:
        x.long()
    topk = trn_binary.size(0) if whole else k
    mAP = 0
    # 检索项个数
    num_query = tst_label.size(0)
    NS = (torch.arange(topk) + 1).float()
    for iter in range(num_query):
        query_label = tst_label[iter].unsqueeze(0)
        query_binary = tst_binary[iter]
        result = query_label.mm(trn_label.t()).squeeze(0) > 0
        _, query_index = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        result = result[query_index[:topk]].float()
        mAP += float(torch.sum(result*(torch.cumsum(result, dim=0)/torch.sum(result))/NS).item())
    mAP /= num_query
    return mAP


# 选择gpu
def choose_gpu(i_gpu):
    torch.cuda.device(i_gpu).__enter__()


# 设定随机种子
def feed_random_seed(seed=np.random.randint(1, 1000)):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

'''
if __name__ == '__main__':
    qB = np.array([[1, -1, 1, 1], [-1, 1, -1, -1], [1, -1, -1, -1]])
    tst_binary = torch.from_numpy(qB)
    rB = np.array([
        [1, -1, -1, -1],
        [-1, 1, 1, -1],
        [1, 1, 1, -1],
        [-1, -1, 1, 1],
        [1, 1, -1, -1],
        [1, 1, 1, -1],
        [-1, 1, -1, -1]])
    trn_binary = torch.from_numpy(rB)
    queryL = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 1],
    ], dtype=np.int64)
    tst_label = torch.from_numpy(queryL)
    retrievalL = np.array([
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
    ], dtype=np.int64)
    trn_label = torch.from_numpy(retrievalL)

    topk = 5
    map = compute_mAP(trn_binary, trn_label, tst_binary, tst_label)
    topkmap = compute_mAP(trn_binary, trn_label, tst_binary, tst_label, False, topk)
    print(map)
    print(topkmap)

'''



