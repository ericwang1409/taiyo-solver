import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        # Create a list of all sizes for the layers
        all_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Create the linear layers using list comprehension
        self.layers = nn.ModuleList([
            nn.Linear(all_sizes[i], all_sizes[i + 1])
            for i in range(len(all_sizes) - 1)
        ])

    def forward(self, x):
        # Pass input through each layer except for the last one
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        # No activation after the last layer
        x = self.layers[-1](x)
        return x
    
    def save(self, file_name="model2.pth"):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        if os.path.isfile(file_name):
            self.load_state_dict(torch.load(file_name))
            self.eval()  # Set the model to evaluation mode
        else:
            print(f"Error: No model found at {file_name}")


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # can choose optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # TODO include some imports (QNet etc, timestamp 1:20:40) in the agent.py file
        # state = torch.tensor(state, dtype=torch.float)
        state = torch.stack([torch.tensor(s, dtype=torch.float) for s in state])
        # next_state = torch.tensor(next_state, dtype=torch.float)
        next_state = torch.stack([torch.tensor(s, dtype=torch.float) for s in next_state])
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # predicted Q values with current state
        pred = self.model(state)

        trgt = pred.clone()
        for idex in range(len(done)):
            Q_new = reward[idex]
            if not done[idex]:
                Q_new = reward[idex] + self.gamma * torch.max(self.model(next_state[idex]))
            
            trgt[idex][torch.argmax(action[idex]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(trgt, pred)
        loss.backward()

        self.optimizer.step()




