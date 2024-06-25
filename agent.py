import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import gymnasium as gym
import random
import math

import environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

node_data_size = 10
embedding_size = 32


env_name = environment.register_env(node_data_size, embedding_size, "none")
env = gym.make(env_name)

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, node_data_size, embedding_size, hidden_size):
        super(DQN, self).__init__()
        self.gru_head_1 = nn.GRUCell(input_size=node_data_size, hidden_size=hidden_size)
        self.gru_head_2 = nn.GRUCell(input_size=node_data_size, hidden_size=hidden_size)

        self.hidden_layer = nn.Linear(256 + embedding_size, hidden_size)

        self.out_head_1 = nn.Linear(hidden_size, node_data_size)
        self.out_head_1 = nn.Linear(hidden_size, node_data_size)

    def forward(self, edges, embedding):
        h1 = torch.zeros(128, device=device)
        h2 = torch.zeros(128, device=device)

        for edge in edges:
            h1 = self.gru_head_1(edge[0], h1)

        for edge in edges:
            h2 = self.gru_head_2(edge[1], h2)

        h = torch.cat((h1, h2, embedding), 1)
        h = F.relu(self.hidden_layer(h))

        out_node_1 = F.sigmoid(self.out_head_1(h))
        out_node_2 = F.sigmoid(self.out_head_2(h))

        return out_node_1, out_node_2


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


state, info = env.reset()


policy_net = DQN(node_data_size, embedding_size, 64).to(device)
target_net = DQN(node_data_size, embedding_size, 64).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        )
