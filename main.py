import torch.optim as optim
import torch
import random
import datetime
import os
from parser import parse_args
from utils import *
from load_data import *
from model import *
from tqdm import tqdm
from time import time
from copy import deepcopy

args = parse_args()

if args.gpu >= 0 and torch.cuda.is_available():
    args.device = "cuda:{}".format(args.gpu)
else:
    args.device = "cpu"

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print(args)

if __name__ == '__main__':
    data_generator = Data(args)
    n, m = data_generator.n_users, data_generator.n_items

    adj_norm = data_generator.adj_norm
    edge_index = data_generator.edge_index

    model = DBCR(n, m, adj_norm, edge_index, args).to(args.device)
    optimizer = optim.Adam(model.parameters(), weight_decay=0, lr=args.lr)

    print("Start Training")
    stopping_step = 0
    last_state_dict = None
    max_recall = 0.0
    for epoch in range(args.epoch):
        epoch_loss = 0
        epoch_loss_bpr = 0
        epoch_loss_pcl = 0
        epoch_loss_bcl = 0

        n_samples = data_generator.uniform_sample()
        n_batch = int(np.ceil(n_samples / args.batch_size))

        model.train()
        for idx in tqdm(range(n_batch)):
            optimizer.zero_grad()

            user, pos, neg = data_generator.mini_batch(idx)
            item = np.concatenate([pos, neg], axis=0)

            loss, loss_bpr, loss_pcl, loss_bcl = model(user, item, pos, neg)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.cpu().item()
            epoch_loss_bpr += loss_bpr.cpu().item()
            epoch_loss_pcl += loss_pcl.cpu().item()
            epoch_loss_bcl += loss_bcl.cpu().item()

        epoch_loss = epoch_loss/n_batch
        epoch_loss_bpr = epoch_loss_bpr/n_batch
        epoch_loss_pcl = epoch_loss_pcl/n_batch
        epoch_loss_bcl = epoch_loss_bcl/n_batch

        print('Epoch:', epoch, 'Loss:', epoch_loss, 'loss_bpr:', epoch_loss_bpr, 'loss_pcl:', epoch_loss_pcl, \
            'loss_bcl:', epoch_loss_bcl)
        if epoch % 5 == 0 or epoch == args.epoch - 1:  # valid every 10 epochs
            model.eval()
            valid_ret = eval_PyTorch(model, data_generator)
            perf_str = 'Epoch: %2d, valid-recall=[%.4f, %.4f], valid-ndcg=[%.4f, %.4f]' % \
                    (epoch, valid_ret['recall'][0], valid_ret['recall'][1], valid_ret['ndcg'][0], valid_ret['ndcg'][1])
            print(perf_str)
            if valid_ret['recall'][0] > max_recall:
                max_recall = valid_ret['recall'][0]
                torch.save(model.state_dict(), f'./ckpts/{args.dataset}_best_model.pth')
    
    best_model = DBCR(n, m, adj_norm, edge_index, args).to(args.device)
    best_model.load_state_dict(torch.load(f'./ckpts/{args.dataset}_best_model.pth'))
    best_model.forward(user, item, pos, neg)
    test_ret = test_PyTorch(best_model, data_generator)
    perf_str = 'Fianl Testing: test-recall=[%.4f, %.4f], test-ndcg=[%.4f, %.4f]' % \
            (test_ret['recall'][0], test_ret['recall'][1], test_ret['ndcg'][0], test_ret['ndcg'][1])
    print(perf_str)