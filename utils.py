import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)

def spmm(sp, emb, device):
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs =  emb[cols] * torch.unsqueeze(sp.values(),dim=1)
    result = torch.zeros((sp.shape[0],emb.shape[1])).to(device)
    result.index_add_(0, rows, col_segs)
    return result

def new_graph(edge_index, weight, n, m):
    edge_index = edge_index.numpy()
    indices = torch.from_numpy(
        np.vstack((edge_index[0], edge_index[1])).astype(np.int64))
    values = weight
    shape = torch.Size((n,m))
    return torch.sparse.FloatTensor(indices, values, shape)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def Recall_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    return recall


def NDCGatK_r(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def test_one_batch(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    recall, ndcg = [], []
    for k in topks:
        recall.append(Recall_ATk(groundTrue, r, k))
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'ndcg': np.array(ndcg)}


def eval_PyTorch(model, data_generator, Ks=[20, 40]):
    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}
    valid_users = list(data_generator.valid_set.keys())
    u_batch_size = data_generator.valid_batch
    n_valid_users = len(valid_users)
    n_user_batchs = n_valid_users // u_batch_size + 1

    batch_rating_list = []
    ground_truth_list = []
    count = 0
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = valid_users[start: end]
        rate_batch = model.predict(user_batch)

        count += rate_batch.shape[0]

        exclude_index = []
        exclude_items = []
        ground_truth = []

        for i in range(len(user_batch)):
            train_items = list(data_generator.train_items[user_batch[i]])
            exclude_index.extend([i] * len(train_items))
            exclude_items.extend(train_items)
            ground_truth.append(list(data_generator.valid_set[user_batch[i]]))
        
        rate_batch[exclude_index, exclude_items] = -(1 << 20)
        _, rate_batch_k = torch.topk(rate_batch, k=max(Ks))
        batch_rating_list.append(rate_batch_k.cpu())
        ground_truth_list.append(ground_truth)

    X = zip(batch_rating_list, ground_truth_list)
    batch_results = []
    for x in X:
        batch_results.append(test_one_batch(x, Ks))
    for batch_result in batch_results:
        result['recall'] += batch_result['recall'] / n_valid_users
        result['ndcg'] += batch_result['ndcg'] / n_valid_users

    assert count == n_valid_users

    return result

def test_PyTorch(model, data_generator, Ks=[20, 40]):
    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}
    test_users = list(data_generator.test_set.keys())
    u_batch_size = data_generator.test_batch
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    batch_rating_list = []
    ground_truth_list = []
    count = 0
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        rate_batch = model.predict(user_batch)

        count += rate_batch.shape[0]

        exclude_index = []
        exclude_items = []
        ground_truth = []

        for i in range(len(user_batch)):
            train_items = list(data_generator.train_items[user_batch[i]])
            exclude_index.extend([i] * len(train_items))
            exclude_items.extend(train_items)
            ground_truth.append(list(data_generator.test_set[user_batch[i]]))
        
        rate_batch[exclude_index, exclude_items] = -(1 << 20)
        _, rate_batch_k = torch.topk(rate_batch, k=max(Ks))
        batch_rating_list.append(rate_batch_k.cpu())
        ground_truth_list.append(ground_truth)

    X = zip(batch_rating_list, ground_truth_list)
    batch_results = []
    for x in X:
        batch_results.append(test_one_batch(x, Ks))
    for batch_result in batch_results:
        result['recall'] += batch_result['recall'] / n_test_users
        result['ndcg'] += batch_result['ndcg'] / n_test_users

    assert count == n_test_users

    return result