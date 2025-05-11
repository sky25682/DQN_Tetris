
import math
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


Transition = namedtuple('Transition',('state','action','next_state','reward'))


class DQN(nn.Module):
    def __init__(self, device, n_observations, n_actions=1):
        super(DQN,self).__init__()
        self.device = device
        self.layer = nn.Sequential(
            nn.Linear(n_observations,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,n_actions)
        ).to(self.device)
        
        
    
    def forward(self,x):
        #X = x.to(device)
        return self.layer(x) # 1차원으로 바꾸기

class ReplayMemory(object):
    def __init__(self,capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory,batch_size)
    def __len__(self):
        return len(self.memory)
    #def pop(self):
    #    self.memroy.pop()





