# -*- coding: utf-8 -*-
import math
import pygame
import torch
import torch.optim as optim
import torch.nn as nn
import random
from tqdm import tqdm
from IPython.display import display, clear_output
import sys
import os
import matplotlib.pyplot as plt
import DQN
import tetris
from collections import namedtuple, deque
import numpy as np
# """
# 0  1  2  3
# 4  5  6  7
# 8  9  10 11
# 12 13 14 15
# """
               
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUMBER_OF_EPISODE = 10
TAU = 0.005
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
LR = 1e-4
MODEL_SAVE_PATH = "./tetris_cnn_model.pth"
colors = [
    (0, 0, 0),
    (120, 37, 179),
    (100, 179, 179),
    (80, 34, 22),
    (80, 134, 22),
    (180, 34, 22),
    (180, 34, 122),
]
Transition = namedtuple('Transition',('state','action','next_state','reward'))




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


def select_action(states): 
    max_val  = None
    best_state = None
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        
        states = torch.tensor(list(states),dtype=torch.float).unsqueeze(1).to(device) #딕셔너리라 리스트로 바꾼 뒤 텐서
        
        value = policy_net(states)
        best_state = value.argmax().item()
        return states[best_state].squeeze().int().tolist()
    else:
        return random.choice(list(states))





def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transition = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transition))
    #print(batch.next_state)
    rewards = torch.tensor(tuple(map(lambda s: s ,batch.reward)), dtype=torch.float, requires_grad=True).to(device)
    
    cur_states = torch.tensor(tuple(map(lambda s: s,batch.state)), dtype=torch.float, requires_grad=True).unsqueeze(1).to(device)
    cur_q_value = torch.cat([policy_net(cur_states)])
    #cur_q_value = policy_net(cur_states[0])
    

    next_states = torch.tensor(tuple(map(lambda s: s,batch.next_state)), dtype=torch.float).unsqueeze(1).to(device)
   
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
    print(f"model save: {path}")

# 모델 불러오기 함수
def load_model(model, optimizer, path=MODEL_SAVE_PATH):

    if os.path.exists(path):
        
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"model load: {path}")
        
    else:
        print(f"not found : {path}")
       


if __name__=='__main__':
    # Initialize the game engine
    pygame.init()
    # Define some colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)
    size = (400, 500)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Tetris")
    
    """
    내 state는 보드 자체이니까 e-greedy를 사용하지 못한다. 
    그래서 e-greedy 없이 한 번 해보려고 한다. 
    """
    steps_done = 0
    env = tetris.Tetris(20,10)
    #state = env.get_board()
    #state = torch.cat(list(map(torch.tensor,state))).to(device)
    #obervation = len(state)
    obervation = 1
    action=1
    policy_net = DQN.DQN(device, obervation, action).to(device)
    target_net = DQN.DQN(device, obervation, action).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = DQN.ReplayMemory(10000)

    load_model(policy_net, optimizer, MODEL_SAVE_PATH)
    steps = 0
    x=[]
    y=[]

    plt.ion() 
    fig, ax = plt.subplots()  
    if torch.cuda.is_available():
        num_episodes = NUMBER_OF_EPISODE
    else:
        num_episodes= 50

    #for i_ep in tqdm(range(num_episodes)): # 게임 * epi
    for i_ep in range(num_episodes):
        cur_state = env.reset_game() #맨 처음 보드
        reward = env.get_score()
        while (1): # 한 게임을 의미한다.
            next_state = env.get_next_state() # 단순히 state만 반환
            if not next_state:
                print("not state")
                break
            try:
                best_state = select_action(next_state.values()) # ai가 좋은 state인지 판단
            except:
                print("error")
                sys.exit()

            best_action = None
            for action, temp_state in next_state.items(): #action은 x, rotation 
                if temp_state == best_state:
                    best_action = action
                    break
                    
            if best_action == None:
                print("no best action22")
                break
            
            reward = env.play_game(best_action)
           #그림 그기기
            show(env)
            env.new_figure() # 여기서 죽는지 안 죽는지 체크한다. 
            if env.state == "gameover":
                reward = -30
                memory.push(cur_state, best_action, next_state[best_action] ,reward)
                show(env)
                break
            else:
                memory.push(cur_state, best_action, next_state[best_action] ,reward)
                #다음 스텝
                cur_state = next_state[best_action]
                steps += 1

                #최적화 단계
                optimize_model()
                
                #가중치 소프트 업데이트 
                if i_ep % 10 == 0:
                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                    target_net.load_state_dict(target_net_state_dict)

        #print(f"Episode :{i_ep} score :{env.get_score()}")
        
        x.append(i_ep)
        y.append(env.get_score())
        ax.scatter(x, y, s=20,color='green', label="Scores")
        ax.set_xlim(0, max(10, i_ep + 1))  # x축 자동 확장
        ax.set_ylim(0, max(50, max(y) + 10))  # y축 자동 확장
        ax.set_title("Dynamic Scatter Plot")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Score")
        
        plt.pause(0.0001)
    
    save_model(policy_net, optimizer, num_episodes, MODEL_SAVE_PATH)
    
    plt.close()
    plt.ioff()     # 인터랙티브 모드 끄기
    plt.show()    
    pygame.quit()


