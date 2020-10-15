import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, inputsize, h1size, h2size, outputsize):
        super(MLP, self).__init__()
        self.model = nn.Sequential(nn.Linear(inputsize, h1size),
                                   nn.ReLU(),
                                   nn.Linear(h1size, h2size),
                                   nn.ReLU(),
                                   nn.Linear(h2size, h2size),
                                   nn.ReLU(),
                                   # nn.Linear(h2size, h2size),
                                   # nn.ReLU(),
                                   # nn.Linear(h2size, h2size),
                                   # nn.ReLU(),
                                   nn.Linear(h2size, outputsize))

    def forward(self, x):
        return self.model(x)
