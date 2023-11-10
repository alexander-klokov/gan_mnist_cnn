import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_true = torch.FloatTensor([1.0]).to(device)
label_false = torch.FloatTensor([0.0]).to(device)

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)

class PrintLayerSize(nn.Module):
    def __init__(self, label = '-'):
        super(PrintLayerSize, self).__init__()
        self.label = label
    
    def forward(self, x):
        print(self.label, x.size())
        return x

def generate_random_seed(size):
    random_data = torch.randn(size) # normal
    return random_data

def generate_random_image(size):
    random_data = torch.rand(size)
    return random_data
