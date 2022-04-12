from cgi import test
import torch 
import argparse
from torch import dtype, float32, ne, nn
import torch.utils.data as Data
from utils import *
from model import GCN, GCN_2, GCN_3, GCN_3_bn
import time


parser = argparse.ArgumentParser("LA-NAS")
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='init learning rate')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=1000, help='num of training epochs')
parser.add_argument('--load_path', type=str, default=None, help='load model path')
parser.add_argument('--model_path', type=str, default='Saved/', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--train_portion', type=float, default=0.9, help='portion of training data')
parser.add_argument('--latency_limit', type=float, default=5, help='laytency limit for missed accuracy test')
parser.add_argument('--err_bound', type=float, default=0.1, help='error bound for bounded accuracy test')
parser.add_argument('--models', type=int, default=4, help='number of convolutional layers in network')
parser.add_argument('--param', type=list, default=[7, 3, 40, 30, 9], help='initial parameter size of network')
parser.add_argument('--seed', type=int, default=1, help='torch manual seed')
args = parser.parse_args()

args.save = 'train-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
args.model_path = args.model_path + 'model-{}'.format(time.strftime("%Y%m%d-%H%M%S"))

MODELS = {1: GCN, 2: GCN_2, 3: GCN_3, 4: GCN_3_bn}
torch.manual_seed(args.seed)

def main():

    use_gpu = torch.cuda.is_available()

    epoch = args.epochs
    latency_limit = args.latency_limit
    errbound = args.err_bound
    logger = get_logger('log/' + args.save + 'log')
    adj, feature, train_y, adj_t, feature_t, test_y = load_data_b()
    train_len = len(adj)
    test_len = len(adj_t)

    if use_gpu:
    
        adj = torch.tensor(adj, dtype=float32).cuda()
        feature = torch.unsqueeze(torch.tensor(feature, dtype=float32), dim=1).cuda()
        train_y = 1000 * torch.tensor(train_y, dtype=float32).cuda()
        adj_t = torch.tensor(adj_t, dtype=float32).cuda()
        feature_t = torch.unsqueeze(torch.tensor(feature_t, dtype=float32), dim=1).cuda()
        test_y = 1000 * torch.tensor(test_y, dtype=float32).cuda()

    torch_trainset = Data.TensorDataset(adj, feature, train_y)
    torch_testset = Data.TensorDataset(adj_t, feature_t, test_y)
    loader = Data.DataLoader(
        dataset=torch_trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    tester = Data.DataLoader(dataset=torch_testset)

    model = MODELS[args.models]

    if args.load_path:
        net = torch.load(args.load_path)
    else:
        net = model(args.param)
    torch.save(net, args.model_path)
    if use_gpu:
        net.cuda()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), args.learning_rate)
    logger.info('net:{}:{}\t criterion:L1Loss()\t optimizer:Adam({})'.format(model, args.param, args.learning_rate))
    logger.info('avrbound:{}\t latencylimit:{}'.format(errbound, latency_limit))
    logger.info('clear to train')
    for ep in range(epoch):
    
        losssum = 0
        errsum = 0
        errsum_t = 0
        losssum_t = 0
        missaccu_t = 0
        missaccu = 0

        # train_phase
        net.train()
        for i, data in enumerate(loader):
            
            A, X, Label = data
            Y = net.forward(A, X)
            loss = criterion(Label, Y)
            errsum += sum(abs((Y - Label) / Label)).item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info('Epoch:[{}/{}]\t acc={:.6f}\t'.format(ep, epoch , 1 -  errsum / train_len))

        # test phase
        net.eval()
        for i, data in enumerate(tester):
            A, X, Label = data
            Y = net.forward(A, X)
            errsum_t += sum(abs((Y - Label) / Label)).item()
            if Label < latency_limit and Y > latency_limit:
                missaccu_t += 1
            if Y < Label * (1 - errbound) or Y > Label * (1 + errbound):
                losssum_t += 1
        avrbaccu = 1 - losssum_t / test_len
        if avrbaccu > 0.9:
            torch.save(net, args.model_path)
            logger.info('model saved!')
        logger.info('Test:\t avrbaccu={:.6f}\t acc={:.6f}\t missaccu={:.6f}'.format(avrbaccu, 1 - errsum_t / test_len, missaccu_t / test_len ))

    logger.info('Train finished')

    
if __name__ == '__main__':
    main()
