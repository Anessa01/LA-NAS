import torch
from torch import nn

class graph_convolution_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(graph_convolution_layer, self).__init__()
        self.params = nn.Parameter(torch.randn(input_dim, output_dim, dtype=torch.float32))
    def forward(self, A, H):
        if len(A.shape) == 2:   # for batchsz = 1
            pre_mul = torch.mm(H, self.params)
            output = torch.mm(A, pre_mul)
        else:
            output = []
            for i in range(A.shape[0]):
                pre_mul = torch.mm(H[i], self.params)
                output.append(torch.mm(A[i], pre_mul))
            output = torch.stack([output[i] for i in range(A.shape[0])])
        return output

layer = graph_convolution_layer(6, 6)
print(layer)
