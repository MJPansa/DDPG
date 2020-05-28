from collections import deque
import torch as T
import numpy as np


class DDPGExperienceBuffer:
    def __init__(self, size, bs, threshold, device):
        self.size = size
        self.bs = bs
        self.threshold_v = threshold
        self.device = device

        self.states = deque(maxlen=self.size)
        self.actions = deque(maxlen=self.size)
        self.rewards = deque(maxlen=self.size)
        self.dones = deque(maxlen=self.size)
        self.next_states = deque(maxlen=self.size)

    def add(self, state, action, reward, done, next_state):

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def draw(self):
        random = np.random.permutation(np.arange(len(self)))[:self.bs]
        return [T.stack([x[i] for i in random]).to(self.device) for x in [self.states, self.actions, self.rewards, self.dones, self.next_states]]

    @property
    def threshold(self):
        return True if (self.threshold_v < len(self.states) / self.size) else False

    def __len__(self):
        return len(self.states)
