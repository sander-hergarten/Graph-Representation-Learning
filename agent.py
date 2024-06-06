import torch
import gymnasium as gym

import environment


env = gym.make(environment.GraphLearnerEnv(node_data_size, embedding_size, render_mode))


