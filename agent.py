import torch
import numpy as np
import random
from collections import deque
from taiyogameAi import TaiyoGameAi
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0 # discount
        self.memory = deque(maxlen=MAX_MEMORY)
        self.num_actions = 5
        self.model = Linear_QNet(226, 256, self.num_actions)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self,game):
        balls = game.balls
        state = -1*torch.ones(226)
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
        print(state)
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 4)
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

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    reward_since_action = 0
    record = 0
    agent = Agent()
    game = TaiyoGameAi()

    state_old = agent.get_state(game)
    velocity_zero = True
    just_started = True 
    done = False
    while True:

        if not velocity_zero:
            reward, done, score, velocity_zero = game.run_game()
            reward_since_action += reward
        else:
            state_new = agent.get_state(game)
            final_move = agent.get_action(state_old)
            state_old = state_new
            if not just_started:
                
                reward, done, score, velocity_zero = game.run_game(True,final_move)
                
                # Train short memory
                agent.train_short_term_memory(state_old, final_move, reward_since_action, state_new, done)

                # remember
                agent.remember(state_old, final_move, reward_since_action, state_new, done)

                reward_since_action = 0
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

            # plotting stuff if we want
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()