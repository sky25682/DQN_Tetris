
import math
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


Transition = namedtuple('Transition',('state','action','next_state','reward'))


# class DQN(nn.Module):
#     def __init__(self, device, n_observations, n_actions=1):
#         super(DQN,self).__init__()
#         self.device = device
#         self.layer = nn.Sequential(
#             nn.Conv2d(n_observations,32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32,64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
            
#             nn.Linear(64,n_actions)
#         ).to(self.device)
        
        
    
#     def forward(self,x):
#         #X = x.to(device)
#         return self.layer(x) # 1차원으로 바꾸기

class DQN(nn.Module):
    def __init__(self, device, n_observations, n_actions=1):
        super(DQN, self).__init__()
        self.device = device

        self.conv = nn.Sequential(
            nn.Conv2d(n_observations, 32, kernel_size=3, padding=1),  # 입력 채널: n_observations (보통 1)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 출력 크기 절반으로 줄임
        )

        # Conv 결과가 (64, 10, 5)라고 가정 (입력이 1x20x10인 경우)
        self.flatten_dim = 64 * 10 * 5

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)            # (B, C=1, 20, 10)
        x = self.conv(x)                 # Conv layer 통과
        x = x.view(x.size(0), -1)        # flatten
        x = self.fc(x)                   # Fully connected
        return x                         # Q(s, a)

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





