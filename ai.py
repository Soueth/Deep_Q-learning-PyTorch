# importação das bibliotecas

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

from torch.autograd import Variable

# Criação da arquitetura da rede neural
class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        
        # neurons1 = 30
        
        # 5 -> 50 -> 30 -> 3 - full connection (dense)
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30, nb_action)
        
    def forward(self, state):
        x1 = F.relu(self.fc1(state))
        x2 = F.relu(self.fc2(x1))
        q_values = self.fc3(x2)
        return q_values

# Implementação da experiência de replay
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    # Um evento é composto por 4 valores: último estado, novo estado, última ação, última recompensa
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
# Implementação do Deep Q-Learning
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True)) * 100) # T = 7
        action = probs.multinomial(1)
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # Obtendo os valores de Q para cada ação desse estado
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # Obtendo os valores de Q para o próximo estado (max Q) para cada ação
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        # Calculando o target (Q_target) para cada ação, multiplicando o fator gamma, os valores das próximas ações e somando às recompensas
        target = self.gamma * next_outputs + batch_reward
        # Calculando a função de custo (TD_loss) usando a função de custo de Smooth L1 Loss (L1 loss) entre os valores de Q e o target
        td_loss = F.smooth_l1_loss(outputs, target)
        # Zerando os gradientes e realizando o backpropagation para ajustar os pesos da rede neural
        self.optimizer.zero_grad()
        # Atualizando os pesos da rede neural com base na função de custo e realizando o backpropagation
        td_loss.backward(retain_graph=True)
        # Atualizando os pesos da rede neural com base nos gradientes calculados pelo backpropagation
        self.optimizer.step()
        
    def update(self, reward, new_signal):
        new_state = torch.tensor(new_signal).float().unsqueeze(0)
        
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), 
                          torch.Tensor([self.last_reward])))
        
        action = self.select_action(new_state)
        
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
            
        self.last_action = action
        self.last_state = new_state
        self.last_eward = reward
        self.reward_window.append(reward)
        
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        
        return action
    
    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict()}, 'last_brain.pth')
        
    def load(self):
        if os.path.isfile('last_brain.pth'):
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Carregado com sucesso!')
        else:
            print('Erro ao carregar!')