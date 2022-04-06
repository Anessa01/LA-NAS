import torch
from torch import nn

class graph_convolution_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(graph_convolution_layer, self).__init__()
        self.params = nn.Parameter(torch.randn(input_dim, output_dim, dtype=torch.float32))
    def forward(self, A, H):
        pre_mul = torch.mm(H, self.params)
        output = torch.mm(A, pre_mul)
        return output



class graph_convolution_layer_channeled(nn.Module):
    def __init__(self, channel, input_dim, output_dim):
        super(graph_convolution_layer_channeled, self).__init__()
        self.params = nn.Parameter(torch.randn(channel, input_dim, output_dim, dtype=torch.float32))
    def forward(self, A, H):
        asz = A.shape
        hsz = H.shape
        pre_mul = torch.matmul(torch.unsqueeze(A, dim=1), H)     #add 1 to the sekond dimension in order to do broadcasted matmul
        pre_mul = torch.unsqueeze(pre_mul, dim=2)
        output = torch.matmul(pre_mul, self.params).view(asz[0], -1, asz[1], self.params.shape[2])
        return output
        
