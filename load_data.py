import pickle
import numpy as np
from time import time
from tqdm import tqdm
from utils import *
import scipy.sparse as sp
import random

class Data(object):
    def __init__(self, args):
        self.device = args.device
        self.path = args.data_path + args.dataset
        self.batch_size = args.batch_size
        self.test_batch = args.batch_size
        self.valid_batch = args.batch_size

        train_file = self.path + '/train.pkl'
        valid_file = self.path + '/valid.pkl'
        test_file = self.path + '/test.pkl'
        with open(train_file, 'rb') as f:
            train_mat = pickle.load(f)
        with open(valid_file, 'rb') as f:
            valid_mat = pickle.load(f)
        with open(test_file, 'rb') as f:
            test_mat = pickle.load(f)

        # get number of users and items
        self.n_users, self.n_items = train_mat.shape[0], train_mat.shape[1]
        self.n_train, self.n_valid, self.n_test = len(train_mat.row), len(valid_mat.row), len(test_mat.row)
        self.print_statistics()

        # train data
        self.rows = train_mat.row
        self.cols = train_mat.col
        self.R = train_mat.todok()
        self.train_items, self.test_set, self.valid_set = {}, {}, {}
        for i in range(len(self.rows)):
            uid = self.rows[i]
            iid = self.cols[i]
            if uid not in self.train_items:
                self.train_items[uid] = [iid]
            else:
                self.train_items[uid].append(iid)
        
        # valid data
        valid_uid, valid_iid = valid_mat.row, valid_mat.col
        for i in range(len(valid_uid)):
            uid = valid_uid[i]
            iid = valid_iid[i]
            if uid not in self.valid_set:
                self.valid_set[uid] = [iid]
            else:
                self.valid_set[uid].append(iid)

        # test data
        test_uid, test_iid = test_mat.row, test_mat.col
        for i in range(len(test_uid)):
            uid = test_uid[i]
            iid = test_iid[i]
            if uid not in self.test_set:
                self.test_set[uid] = [iid]
            else:
                self.test_set[uid].append(iid)

        # edge_index 
        self.edge_index = np.array(train_mat.nonzero())

        # normalizing the adj matrix
        rowD = np.array(train_mat.sum(1)).squeeze()
        colD = np.array(train_mat.sum(0)).squeeze()
        for i in range(len(train_mat.data)):
            train_mat.data[i] = train_mat.data[i] / pow(rowD[train_mat.row[i]]*colD[train_mat.col[i]], 0.5)

        self.adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train_mat).coalesce()
        
    def uniform_sample(self):
        user_pos = [(u, i) for u,i in zip(self.rows, self.cols)]
        random.shuffle(user_pos)
        train_data = []
        # for i, up in tqdm(enumerate(user_pos), desc='Sampling Data', total=len(user_pos)):
        for i, up in enumerate(user_pos):
            user = up[0]
            pos_item = up[1]
            while True:
                neg_item = np.random.randint(0, self.n_items)
                if (user, neg_item) not in self.R:
                    break
            train_data.append([user, pos_item, neg_item])
        self.train_data = np.array(train_data)
        return len(self.train_data)

    def mini_batch(self, batch_idx):
        st = batch_idx * self.batch_size
        ed = min((batch_idx + 1) * self.batch_size, len(self.train_data))
        batch_data = self.train_data[st: ed]
        users = batch_data[:, 0]
        pos_items = batch_data[:, 1]
        neg_items = batch_data[:, 2]
        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train)/(self.n_users * self.n_items)))

    def get_statistics(self):
        sta = ""
        sta += 'n_users=%d, n_items=%d\t' % (self.n_users, self.n_items)
        sta += 'n_interactions=%d\t' % (self.n_train + self.n_test)
        sta += 'n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train)/(self.n_users * self.n_items))
        return sta
