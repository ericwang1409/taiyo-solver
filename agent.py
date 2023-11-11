import torch
import numpy as np
import random
from collections import deque
from taiyogameAi import TaiyoGameAi
from model import Linear_QNet, QTrainer
from helper import plot, plot_differences
import pygame

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.95 # discount
        self.memory = deque(maxlen=MAX_MEMORY)
        self.num_actions = 10
        self.model = Linear_QNet(196, [256]*4, self.num_actions)
        print(self.model)
        # self.model.load("model/model2.pth")
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self,game):
        balls = game.balls
        state = -1*torch.ones(196)
        i = 0
        for ball in balls:
            state[i:i+3] = torch.tensor([ball.body.position[0],ball.body.position[1],ball.radius])
            i+=3
        state[-1] = game.current_ball.radius

        return state

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        # print(state)
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, self.num_actions)
            # final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            # final_move[move] = 1

        return 484 + int(move*((800-484)/self.num_actions))

    def train_long_term_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_term_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
def get_height(state):
    # Reshape the state tensor to 65 rows and 3 columns (for x, y, r)
    reshaped_state = state[:-1].view(65, 3)  # The last element is the radius of an additional ball, so we exclude it
    # Filter out the (-1, -1, -1) triples, which we assume indicate unused slots
    valid_rows = reshaped_state[~(reshaped_state == -1).all(1)]
    # Calculate the y position minus the radius for each ball, only for valid rows
    y_minus_r = valid_rows[:, 1] - valid_rows[:, 2]  # Index 1 is y position, Index 2 is radius
    # Return the smallest value (which corresponds to the lowest point, hence the highest height as lower is higher here)
    try:
        highest_point = torch.min(y_minus_r, dim=0)[0]
    except IndexError:
        print("Index error because first state is all -1's\n", state)
        screen = pygame.display.set_mode((1280, 720))
        HEIGHT = screen.get_height()
        highest_point = (HEIGHT + HEIGHT // 1.4) // 2

    return highest_point


def train():
    plot_scores = []
    plot_mean_scores = []
    height_differences=[]

    total_score = 0
    # reward_since_action = 0
    record = 0
    agent = Agent()
    game = TaiyoGameAi()

    state_old = agent.get_state(game)
    velocity_zero = True
    total_reward_height_diff = 0
    just_started = True 
    done = False
    while True:
        if not velocity_zero:
            reward, done, score, velocity_zero = game.run_game()
            # reward_since_action += reward
        else:
            state_new = agent.get_state(game)
            final_move = agent.get_action(state_old)
            
            if not just_started:

                reward_height_diff = -(get_height(state_new) - get_height(state_old)) 
                # Negative since height is measured from the top
                total_reward_height_diff += reward_height_diff
                height_differences.append(total_reward_height_diff)
                
                # Train short memory
                agent.train_short_term_memory(state_old, final_move, reward_height_diff, state_new, done)

                # remember
                agent.remember(state_old, final_move, reward_height_diff, state_new, done)

                # reward_since_action = 0

                reward, done, score, velocity_zero = game.run_game(True,final_move)
            
            state_old = state_new # This was above the training before, which might have been a big problem
            
            just_started = False

        # restart if done
        if done:
            game.game_reset()
            agent.n_games += 1
            agent.train_long_term_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # if agent.n_games % 1 == 0:
            #     plot_differences(height_differences)
            
            # plotting stuff if we want
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            velocity_zero = True
            total_reward_height_diff = 0
            height_differences = []


if __name__ == '__main__':
    train()