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

layer = graph_convolution_layer(6, 6)
print(layer)
