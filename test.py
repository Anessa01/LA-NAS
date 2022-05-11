from cgi import test
import torch 
import argparse
from torch import dtype, float32, ne, nn
import torch.utils.data as Data
from transfer_utils import *
from utils import *
from model import GCN, GCN_2, GCN_3, GCN_3_bn
import time


parser = argparse.ArgumentParser("LA-NAS")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--load_path', type=str, default='Saved/model-20220510-094714', help='load model path')
parser.add_argument('--latency_limit', type=float, default=6, help='laytency limit for missed accuracy test')
parser.add_argument('--err_bound', type=float, default=0.15, help='error bound for bounded accuracy test')
parser.add_argument('--seed', type=int, default=1, help='torch manual seed')
args = parser.parse_args()
torch.manual_seed(args.seed)

def main():

    use_gpu = torch.cuda.is_available()

    latency_limit = args.latency_limit
    errbound = args.err_bound
    #adj, feature, train_y, adj_t, feature_t, test_y = load_data_b(sep=0.95)
    adj_t, feature_t, test_y = load_data_101()
    test_len = len(adj_t)

    if use_gpu:
    
        adj_t = torch.tensor(adj_t, dtype=float32).cuda()
        feature_t = torch.unsqueeze(torch.tensor(feature_t, dtype=float32), dim=1).cuda()
        test_y = 1000 * torch.tensor(test_y, dtype=float32).cuda()

    torch_testset = Data.TensorDataset(adj_t, feature_t, test_y)
    tester = Data.DataLoader(dataset=torch_testset)

    net = torch.load(args.load_path)
    if use_gpu:
        net.cuda()
    criterion = nn.L1Loss()
    errsum = 0
    errsum_t = 0
    losssum_t = 0
    missaccu_t = 0
    # test phase
    net.eval()
    latlim = 0.1
    errb = 0.01
    while latlim <= latency_limit or errb <= errbound:
        for i, data in enumerate(tester):
            A, X, Label = data
            Y = net.forward(A, X)
            errsum_t += sum(abs((Y - Label) / Label)).item()
            if Label < latlim and Y > latlim:
                missaccu_t += 1
            if Y < Label * (1 - errb) or Y > Label * (1 + errb):
                losssum_t += 1
        avrbaccu = 1 - losssum_t / test_len
        print('Test:\t latlim={:.2f}\t errb={:.2f}\t avrbaccu={:.6f}\t acc={:.6f}\t missaccu={:.6f}'.format(latlim, errb, avrbaccu, 1 - errsum_t / test_len, missaccu_t / test_len ))
        latlim += 0.1
        errb += 0.01
        errsum_t = 0
        losssum_t = 0
        missaccu_t = 0


    print('Train finished')

    
if __name__ == '__main__':
    main()
