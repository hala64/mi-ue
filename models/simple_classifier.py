from torch import nn
from torch.nn import functional as F

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

class two_NN(nn.Module):
    def __init__(self,  feature_dim=3072, n_classes: int = 10):
        super().__init__()
        self.linear1 = nn.Linear(feature_dim, feature_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        x = x.view(len(x), -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class three_NN(nn.Module):
    def __init__(self, feature_dim=3072, n_classes: int = 10):
        super().__init__()
        self.linear1 = nn.Linear(feature_dim, feature_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(feature_dim, int(feature_dim / 2))
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(int(feature_dim / 2), n_classes)

    def forward(self, x):
        x = x.view(len(x), -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


class LeNet5(nn.Module):

    def __init__(self, num_classes=10, grayscale=False):
        super(LeNet5, self).__init__()

        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(

            nn.Conv2d(in_channels, 6 * in_channels, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6 * in_channels, 16 * in_channels, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5 * in_channels, 120 * in_channels),
            nn.Tanh(),
            nn.Linear(120 * in_channels, 84 * in_channels),
            nn.Tanh(),
            nn.Linear(84 * in_channels, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(len(x), -1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits#, probas