from cgi import test
import torch 
from torch import dtype, float32, nn
from utils import *
from model import GCN, GCN_2

epoch = 100
latency_limit = 5
errbound = 0.1
logger = get_logger('log/exp20220331.log')

adj, feature, train_y, adj_t, feature_t, test_y = load_data()
train_len = len(adj)
test_len = len(adj_t)

#net = GCN(7, 7)
net = GCN_2([7, 20, 20])
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), 1e-3)
logger.info('net:GCN_2(7, 20, 20)\t criterion:L1Loss()\t optimizer:Adam(1e-3)')
logger.info('avrbound:{}\t latencylimit:{}'.format(errbound, latency_limit))
logger.info('clear to train')
for ep in range(epoch):
    
    losssum = 0
    accusum = 0
    accusum_t = 0
    losssum_t = 0
    missaccu_t = 0
    missaccu = 0
    for i in range(train_len):
        A = torch.tensor(adj[i], dtype=torch.float32)
        X = torch.tensor(feature[i], dtype=torch.float32)
        Y = net.forward(A, X)
        #print(Y)
        label = torch.tensor([train_y[i] * 1000], dtype=float32)
        loss = criterion(label, Y)
        accusum += 1 - abs(Y - label) / label
        if label < latency_limit and Y > latency_limit:
            missaccu += 1
        if Y < label * (1 - errbound) or Y > label * (1 + errbound):
            losssum += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #print('epoch', ep, 'avrboundaccu = ', 1 - losssum / train_len, 'accu = ', accusum.item() / train_len, 'missaccu = ', missaccu / train_len)
    logger.info('Epoch:[{}/{}]\t avrbaccu={:.6f}\t acc={:.6f}\t missaccu={:.6f}'.format(ep, epoch , 1 - losssum / train_len, accusum.item() / train_len, missaccu / train_len ))
    #log.writelines('epoch', ep, 'avrboundaccu = ', 1 - losssum / train_len, 'accu = ', accusum.item() / train_len, 'missaccu = ', missaccu / train_len)
    #print('testing...')
    for i in range(test_len):
        A = torch.tensor(adj_t[i], dtype=torch.float32)
        X = torch.tensor(feature_t[i], dtype=torch.float32)
        Y = net.forward(A, X)
        label = torch.tensor([test_y[i] * 1000], dtype=float32)
        if label < latency_limit and Y > latency_limit:
            missaccu_t += 1
        if Y < label * (1 - errbound) or Y > label * (1 + errbound):
            losssum_t += 1
        #if i == 15:
        #    print('pre = ', Y, ', label = ', label)
        accusum_t += 1 - abs(Y - label) / label
    #print('testepoch', ep, ', avrboundaccu = ', 1 - losssum_t / test_len, ', accu = ', accusum_t.item() / test_len)
    logger.info('Test:\t avrbaccu={:.6f}\t acc={:.6f}\t missaccu={:.6f}'.format(1 - losssum_t / test_len, accusum_t.item() / test_len, missaccu_t / test_len ))
    #print('missed_accu=', missaccu_t / test_len)



    

