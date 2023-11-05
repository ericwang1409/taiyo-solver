import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name="model.pth"):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        xm.save(self.state_dict(), file_name) if xm.is_master_ordinal() else None


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
        state = torch.tensor(state, dtype=torch.float).to(xm.xla_device())
        next_state = torch.tensor(next_state, dtype=torch.float).to(xm.xla_device())
        action = torch.tensor(action, dtype=torch.long).to(xm.xla_device())
        reward = torch.tensor(reward, dtype=torch.float).to(xm.xla_device())

        if len(state.shape) == 1:
            state = torch.unqueeze(state, 0)
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
            
            trgt[idex][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(trgt, pred)
        loss.backward()

        xm.optimizer_step(self.optimizer, barrier=True)

        # Assuming you have some data to train your model, 
    # you would put your training loop here and make sure to load data to the TPU device
    # ...

    # Example training loop skeleton for TPU:
    def train_model():
        # Your training loop here
        pass

    def _mp_fn(rank, flags):
        torch.set_default_tensor_type('torch.FloatTensor')
        train_model()

        FLAGS = {}
        xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')




