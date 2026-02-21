from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, n_classes: int = 10):
        super().__init__()
        self.enc = encoder
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.lin = nn.Linear(self.feature_dim, self.n_classes)

    def forward(self, x):
        return self.lin(self.enc(x))


class Linear(nn.Module):
    def __init__(self,  feature_dim=3072, n_classes: int = 10):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.lin = nn.Linear(self.feature_dim, self.n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.lin(x)