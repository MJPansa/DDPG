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
        self.out = nn.Linear(n_hidden, n_actions)

        self.dropout = nn.Dropout(0.5)

        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, x):
        if not isinstance(x, T.Tensor):
            x = T.Tensor(x).unsqueeze(0).to(self.device)

        x = self.dropout(F.relu(self.input(x)))
        x = self.dropout(F.relu(self.l1(x)))
        x = self.dropout(F.tanh(self.out(x)))

        return x


class DDPGCritic(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden, lr, device):
        super(DDPGCritic, self).__init__()
        self.device = device

        self.input = nn.Linear(n_states + n_actions, n_hidden)
        self.l1 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, 1)

        self.dropout = nn.Dropout(0.5)

        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, x, y):
        x = self.dropout(self.input(T.cat([x, y], dim=1)))
        x = self.dropout(F.relu(self.l1(x)))
        x = self.out(x)


        return x
