import torch as T
import torch.optim as optim
import torch.nn.modules as nn
import torch.nn.functional as F


class DDPGActor(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden, lr, device):
        super(DDPGActor, self).__init__()
        self.device = device

        self.input = nn.Linear(n_states, n_hidden)
        self.l1 = nn.Linear(n_hidden, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)

        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, x):
        if not isinstance(x, T.Tensor):
            x = T.Tensor(x).unsqueeze(0).to(self.device)

        x = F.relu(self.input(x))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.tanh(self.out(x))

        return x


class DDPGCritic(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden, lr, device):
        super(DDPGCritic, self).__init__()
        self.device = device

        self.input = nn.Linear(n_states, n_hidden)
        self.l1 = nn.Linear(n_hidden + n_actions, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, 1)

        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, x, y):
        x = F.relu(self.input(x))
        x = T.cat([x, y], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.out(x)

        return x
