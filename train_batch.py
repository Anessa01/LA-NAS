from cgi import test
import torch 
import argparse
from torch import dtype, float32, nn
import torch.utils.data as Data
from utils import *
from model import GCN, GCN_2
import time


parser = argparse.ArgumentParser("LA-NAS")
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='init learning rate')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--train_portion', type=float, default=0.9, help='portion of training data')
parser.add_argument('--latency_limit', type=float, default=5, help='laytency limit for missed accuracy test')
parser.add_argument('--err_bound', type=float, default=0.1, help='error bound for bounded accuracy test')
parser.add_argument('--layers', type=int, default=2, help='number of convolutional layers in network')
parser.add_argument('--param', type=list, default=[7, 20, 20], help='initial parameter size of network')
parser.add_argument('--seed', type=int, default=1, help='torch manual seed')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))

MODELS = {1: GCN, 2: GCN_2}
torch.manual_seed(args.seed)

def main():
    epoch = args.epochs
    latency_limit = args.latency_limit
    errbound = args.err_bound
    logger = get_logger('log/' + args.save + 'log')

    adj, feature, train_y, adj_t, feature_t, test_y = load_data_b()
    train_len = len(adj)
    test_len = len(adj_t)
    adj = torch.tensor(adj, dtype=float32)
    feature = torch.tensor(feature, dtype=float32)
    train_y = torch.tensor(train_y, dtype=float32)

    torch_trainset = Data.TensorDataset(adj, feature, train_y)
    loader = Data.DataLoader(
        dataset=torch_trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = MODELS[args.layers]

    #net = GCN(7, 7)
    net = model(args.param)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), args.learning_rate)
    logger.info('net:{}:{}\t criterion:L1Loss()\t optimizer:Adam({})'.format(model, args.param, args.learning_rate))
    logger.info('avrbound:{}\t latencylimit:{}'.format(errbound, latency_limit))
    logger.info('clear to train')
    for ep in range(epoch):
    
        losssum = 0
        accusum = 0
        accusum_t = 0
        losssum_t = 0
        missaccu_t = 0
        missaccu = 0

        for i, data in enumerate(loader):
            A, X, Label = data
            Y = net.forward(A, X)
            loss = criterion(Label, Y)
            accusum += args.batch_size - sum(abs(Y - Label) / Label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info('Epoch:[{}/{}]\t acc={:.6f}\t'.format(ep, epoch , accusum.item() / train_len))
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
        logger.info('Test:\t avrbaccu={:.6f}\t acc={:.6f}\t missaccu={:.6f}'.format(1 - losssum_t / test_len, accusum_t.item() / test_len, missaccu_t / test_len ))



    
if __name__ == '__main__':
    main()
