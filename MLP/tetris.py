import random
import copy




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
        self.color = 1
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
        scores = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                scores += 10 # 부신 라인 개수당 10점
                for i1 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[i1][j] = self.field[i1 - 1][j]
        self.score += scores  
        

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
#이건 gameover만 안 하고 freeze함
    def DQN_freeze(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
        self.score += 1 
        self.break_lines()
        
    # def DQN_no_freeze(self):
        
    #     for i in range(4):
    #         for j in range(4):
    #             if i * 4 + j in self.figure.image():
    #                 self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
    #     self.DQN_break_lines()
    #     #self.score += 1 
    #     #self.break_lines()
    #     #self.new_figure() 
    #     #if self.intersects():
    #     #    self.state = "gameover"
    
    
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
        state_prop = {}
        t_state = self.state 
        field = self.get_board()
        figure_x = self.figure.x
        figure_y = self.figure.y
        figure_rotation = self.figure.rotation
        score = self.score
        
        for num_rotate in range(self.figure.len_rotate()):
            for x in range(10):
                self.field = copy.deepcopy(field) # 보드 초기화
                self.figure.x = x # x좌표 설정
                self.figure.y = figure_y
                self.figure.rotation = num_rotate 
                if self.intersects():
                    continue
                while not self.intersects():
                    self.figure.y += 1
                self.figure.y -= 1
                self.DQN_freeze() # 피스 넣기
                state_prop[(x, num_rotate)] =  self.get_state_prop() 
         
        self.field = copy.deepcopy(field) # 원상복구 
        self.state = t_state
        self.figure.x = figure_x 
        self.figure.y = figure_y 
        self.figure.rotation = figure_rotation 
        self.score = score
        
        return state_prop
    
    
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
            for j in range(0, self.height):
                if self.field[j][i] != 0 and check == 0:
                    check = 1
                elif self.field[j][i] == 0 and check == 1:
                    hole += 1
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
        score = 1 + (self.check_lines() * 30) 
    
        return score
        
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
   