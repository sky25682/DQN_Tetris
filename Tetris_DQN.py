# -*- coding: utf-8 -*-
import pygame
import random
import copy

colors = [
    (0, 0, 0),
    (120, 37, 179),
    (100, 179, 179),
    (80, 34, 22),
    (80, 134, 22),
    (180, 34, 22),
    (180, 34, 122),
]


class Figure:
    x = 0
    y = 0

    figures = [
            [[0, 4, 8, 12], [0, 1, 2, 3]],
            [[4, 5, 9, 10], [1, 5, 4, 8]],
            [[5, 6, 8, 9], [0, 4, 5, 9]],
            [[0, 1, 4, 8], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
            [[0, 1, 5, 9], [4, 5, 6, 8], [0, 4, 8, 9], [2, 4, 5, 6]],
            [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [0, 4, 5, 8]],
            [[0, 1, 4, 5]],
            ]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = random.randint(0, len(self.figures) - 1)
        self.color = 1#random.randint(1, len(colors) - 1)
        self.rotation = 0

    def image(self):
        return self.figures[self.type][self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.figures[self.type])
    def len_rotate(self):
         return len(self.figures[self.type])

class Tetris:
    def __init__(self, height, width):
        self.level = 2
        self.score = 0
        self.state = "start"
        self.field = []
        self.height = 0
        self.width = 0
        self.x = 100
        self.y = 60
        self.zoom = 20
        self.figure = None
    
        self.height = height
        self.width = width
        self.score = 0
        self.state = "start"
        self.field = []
        for i in range(self.height):
            new_line = []
            for j in range(self.width):
                new_line.append(0)
            self.field.append(new_line)

    def new_figure(self,x=3,y=0):
        self.figure = Figure(x, y)
        if self.intersects():
            self.state = "gameover"
            

    def intersects(self):
        intersection = False
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    if i + self.figure.y > self.height - 1 or \
                            j + self.figure.x > self.width - 1 or \
                            j + self.figure.x < 0 or \
                            self.field[i + self.figure.y][j + self.figure.x] > 0:
                        intersection = True # true이면 겹쳤다는 것 -> stop
        return intersection

    def break_lines(self):
        lines = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines += 40 # 부신 라인 개수당 40점
                for i1 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[i1][j] = self.field[i1 - 1][j]
        self.score += lines  
        
    def DQN_break_lines(self):
        lines = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines += 40 # 부신 라인 개수당 40점
                for i1 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[i1][j] = self.field[i1 - 1][j]
        #self.score += lines  

    def go_space(self):
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        self.freeze()

    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            self.freeze()

    def freeze(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
        self.score += 1 
        self.break_lines()
        self.new_figure() 
        if self.intersects():
            self.state = "gameover"

    def DQN_freeze(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
        self.score += 1 
        self.break_lines()
        
    def DQN_no_freeze(self):
        
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
        self.DQN_break_lines()
        #self.score += 1 
        #self.break_lines()
        #self.new_figure() 
        #if self.intersects():
        #    self.state = "gameover"
    
    
    def go_side(self, dx):
        old_x = self.figure.x
        self.figure.x += dx
        if self.intersects():
            self.figure.x = old_x

    def rotate(self):
        old_rotation = self.figure.rotation
        self.figure.rotate()
        if self.intersects():
            self.figure.rotation = old_rotation
            return True # 에러나서 회전 못했다.
        return False

    def only_data_input(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.field[i + self.figure.y][j + self.figure.x] = self.figure.color

    def get_board(self): # 
        return copy.deepcopy(self.field)
        
    
    def get_score(self):
        return self.score
        
    def reset_game(self): # __init__ 다 가져옴  반환ㄱ밧은 get_state
        self.level = 2
        self.score = 0
        self.state = "start"
        self.field = []
        self.x = 100
        self.y = 60
        self.zoom = 20
        self.figure = None
    
        self.height = self.height
        self.width = self.width
        self.score = 0
        self.state = "start"
        self.field = []
        for i in range(self.height):
            new_line = []
            for j in range(self.width):
                new_line.append(0)
            self.field.append(new_line)
        self.new_figure()
        return self.get_state_prop()
        #return self.get_board()
            
    def get_next_state(self):
        #정보 저장
        states = {}
        t_state = self.state 
        field = self.get_board()
        figure_x = self.figure.x
        figure_y = self.figure.y
        figure_rotation = self.figure.rotation
        
        
        for rotate in range(self.figure.len_rotate()):
            for x in range(10):
                self.field = copy.deepcopy(field) # 보드 초기화
                #self.score = score # 점수 초기화

                self.figure.x = figure_x 
                self.figure.y = figure_y 
                self.figure.rotation = figure_rotation 
                        

                self.figure.x = x # x좌표 설정
                for _ in range(rotate): #회전을 위한 for문이다. 
                    if self.rotate(): #회전 실제 변경
                        continue #회전 안된다면 그냥 넘어가기로 함
                    
                if not self.intersects():#안 부딫쳤을때
                    #self.go_space() 
                    while not self.intersects():
                        self.figure.y += 1
                    self.figure.y -= 1 # space 누르기
                    self.DQN_no_freeze() # 피스 넣기

                    
                    
                    states[(x, rotate)] =  self.get_state_prop() #()가 키이고 = 뒤에 있는게 값이다.
                    #for debug
        
                    #print(f'x : {x} , rotate : {rotate}')
                    #print(self.field,'\n\n\n\n')
        
        self.field = copy.deepcopy(field) # 원상복구 
        self.state = t_state
        self.figure.x = figure_x 
        self.figure.y = figure_y 
        self.figure.rotation = figure_rotation 
        return states
    def get_state_prop(self):
        checkline = self.check_lines()
        hole = self.check_hole()
        sun_heights, max_heights = self.check_height()
        diff = self.diff_line()
        return [checkline,hole,sun_heights,diff]

    
    def diff_line(self):
        diff = 0
        lists = []
        for i in range(self.width):
            
            for j in range(self.height):
                if self.field[j][i] != 0:
                    lists.append(self.height - j)
                    break

        for i in range(len(lists)-1):
            diff += abs(lists[i] -lists[i+1])
        return diff
        
    def check_hole(self):#홀 수 
        hole = 0
        for i in range(self.width):
            check = 0
            hp = 0.02
            for j in range(0, self.height):
                if self.field[j][i] != 0 and check == 0:
                    check = 1
                elif self.field[j][i] == 0 and check == 1:
                    hole += 1*hp
                hp += 0.02
        return hole
    def check_height(self): #높이 
        height = 0
        lists = []
        for i in range(self.width):
            
            for j in range(self.height):
                if self.field[j][i] != 0:
                    lists.append(self.height - j)
                    #lists.append(self.height)
                    break
        if lists:
            max_list = max(lists)
        else:
            max_list = 0
        return sum(lists), max_list
    def play_game(self, best_action):# x, y 가 주어진다. 
        x_location, rotation = best_action
        self.figure.x = x_location
        self.figure.rotation = rotation
        
        while not self.intersects():
            self.figure.y += 1
            
        self.figure.y -= 1 # space 누르기
        self.DQN_freeze() # no new figure
        score = 1 + (self.check_lines() ** 2) * 50
        #score += self.check_height()
       # sun_heights, max_heights = self.check_height()
        
        #diff = self.diff_line()
        ##if diff > 50:
        #score -= diff * 0.02
        #holes = self.check_hole()
        #score -= holes
        #score += self.check_linear()
        #if self.check_x() < 5:
        #    score-= 10 -self.check_x()
        return score, self.state
        
    def check_x(self):
        lines = 0
        for i in range(10):
            if self.field[19][i] !=0:
                lines +=1
        return lines
        
    def check_linear(self):
        lines = 0
        for i in range(self.height-1,-1):
            zeros = 0
            for j in range(self.width-1):
                if self.field[i][j] != 0:
                    if self.field[i][j+1] != 0:
                        lines += 1
        return lines
        
    def check_lines(self):
        lines = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines += 1
        return lines
    def check_sc(self):
        count = 0
        ref = 0
        for i in range( self.height):
            ref += 0.02
            for j in range(self.width):
                if self.field[i][j] != 0:
                    count += ref
            
        return count
   
# """
# 0  1  2  3
# 4  5  6  7
# 8  9  10 11
# 12 13 14 15
# """
                    

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions=1):
        super(DQN,self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_observations,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,n_actions)
        ).to(device)
        
        
    
    def forward(self,x):
        #X = x.to(device)
        return self.layer(x) # 1차원으로 바꾸기




Transition = namedtuple('Transition',('state','action','next_state','reward'))

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


def show(game):
    screen.fill(WHITE)

    for i in range(game.height):
        for j in range(game.width):
            pygame.draw.rect(screen, GRAY, [game.x + game.zoom * j, game.y + game.zoom * i, game.zoom, game.zoom], 1)
            if game.field[i][j] > 0:
                pygame.draw.rect(screen, colors[game.field[i][j]],
                                 [game.x + game.zoom * j + 1, game.y + game.zoom * i + 1, game.zoom - 2, game.zoom - 1])

   

    font = pygame.font.SysFont('Calibri', 25, True, False)
    font1 = pygame.font.SysFont('Calibri', 65, True, False)
    text = font.render("Score: " + str(game.score), True, BLACK)
    text_game_over = font1.render("Game Over", True, (255, 125, 0))
    text_game_over1 = font1.render("Press ESC", True, (255, 215, 0))

    screen.blit(text, [0, 0])
    if game.state == "gameover":
        screen.blit(text_game_over, [20, 200])
        screen.blit(text_game_over1, [25, 265])

    pygame.display.flip()




  

import numpy as np
from tqdm import tqdm
import time
from IPython.display import display, clear_output
import sys
import os
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAU = 0.005
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
LR = 1e-4
MODEL_SAVE_PATH = "tetris_dqn_model.pth"
"""
내 state는 보드 자체이니까 e-greedy를 사용하지 못한다. 
그래서 e-greedy 없이 한 번 해보려고 한다. 
"""
steps_done = 0
def select_action(states): 
    max_val  = None
    best_state = None
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        
        states = torch.tensor(list(states),dtype=torch.float).to(device) #딕셔너리라 리스트로 바꾼 뒤 텐서
           
        value = policy_net(states)
        #print(value)
        
        #if not max_val or max_val < value.item():
        #    max_val = value
        #    best_state = state
        best_state = value.argmax().item()
        return list(states[best_state])
    else:
        return random.choice(list(states))


# Initialize the game engine
pygame.init()

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

size = (400, 500)
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Tetris")


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transition = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transition))
    #print(batch.next_state)
    rewards = torch.tensor(tuple(map(lambda s: s ,batch.reward)), dtype=torch.float, requires_grad=True).to(device)
    
    cur_states = torch.tensor(tuple(map(lambda s: s,batch.state)), dtype=torch.float, requires_grad=True).to(device)
    cur_q_value = torch.cat([policy_net(cur_states)])
    #cur_q_value = policy_net(cur_states[0])
    

    next_states = torch.tensor(tuple(map(lambda s: s,batch.next_state)), dtype=torch.float).to(device)
    
    next_q_value = target_net(next_states).max(1)[0]
    #next_q_value = target_net(next_states)  # 각 상태에 대한 최대 Q값 선택
    
    max_Q_value = rewards + GAMMA * next_q_value
    # Huber 손실 계산
    criterion = nn.MSELoss()
    loss = criterion(cur_q_value, max_Q_value)
    #if steps % 50 == 0:
    #    print(f"loss : {loss}")
    
    # 모델 최적화
    optimizer.zero_grad()
    #loss.requires_grad_(True)
    loss.backward()
    # 변화도 클리핑 바꿔치기
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# 모델 저장 함수
def save_model(model, optimizer, episode, path=MODEL_SAVE_PATH):
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"모델이 저장되었습니다: {path}")

# 모델 불러오기 함수
def load_model(model, optimizer, path=MODEL_SAVE_PATH, parse=False):
    if os.path.exists(path) and parse:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"모델이 불러와졌습니다: {path}")
        
    else:
        print(f"저장된 모델을 찾을 수 없습니다: {path}")
       

env = Tetris(20,10)
#state = env.get_board()
#state = torch.cat(list(map(torch.tensor,state))).to(device)
#obervation = len(state)
obervation = 4
action=1
policy_net = DQN(obervation, action).to(device)
target_net = DQN(obervation, action).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


load_module_real_= False
load_model(policy_net, optimizer, MODEL_SAVE_PATH, load_module_real_)



steps = 0
x=[]
y=[]


if torch.cuda.is_available():
    num_episodes = 10000
else:
    num_episodes= 50
# 테스트용
#num_episodes=10

#for i_ep in tqdm(range(num_episodes)): # 게임 * epi
for i_ep in range(num_episodes):
    cur_state = env.reset_game() #맨 처음 보드
    reward = env.get_score()
    while (1): # 한 게임을 의미한다.
        
        next_state = env.get_next_state() # 단순히 state만 반환

        
        
        if not next_state:
            break
            
        try:
            best_state = select_action(next_state.values())
        except:
           # print(next_state)
           # print(next_state.values())
            sys.exit()
            
        best_action = None
        for action, state1 in next_state.items():
            if state1 == best_state:
                best_action = action
                break
                
        if best_action == None:
            break
            
        #한 단계 수행
        reward, done = env.play_game(best_action)
        #print(reward)
        #if done == 'gameover':
            #memory.pop()
        #    break
        #메모리 시시
        memory.push(cur_state, best_action, next_state[best_action] ,reward)
        
        
        #다음 스텝
        cur_state = next_state[best_action]
        
        steps += 1
        #time.sleep(0.5)

        #최적화 단계
        optimize_model()
        
        #가중치 소프트 업데이트 
        if i_ep % 10 == 0:
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

        #그림 그기기
        show(env)
        env.new_figure() # 여기서 죽는지 안 죽는지 체크한다. 
        
        if done == 'gameover':
            break
            
    #print(f"Episode :{i_ep} score :{env.get_score()}")
    x.append(i_ep)
    y.append(env.get_score())
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='green', label="Scores")
    plt.xlim(0, max(10, i_ep + 1))  # x축 자동 확장
    plt.ylim(0, max(50, max(y) + 10))  # y축 자동 확장
    plt.title("Dynamic Scatter Plot")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    
    # 그래프 출력 및 이전 그래프 제거
    clear_output(wait=True)
    display(plt.gcf())
    plt.close()  # 플롯 객체 닫기

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='green', label="Scores")
plt.xlim(0, max(10, len(x)))  # x축 자동 확장
plt.ylim(0, max(50, max(y) + 10))  # y축 자동 확장
plt.title("Final Scatter Plot")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
display(plt.gcf())

pygame.quit()

save_model(policy_net, optimizer, num_episodes, MODEL_SAVE_PATH)


