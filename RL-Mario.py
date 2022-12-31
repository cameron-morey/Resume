import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

import RL_Mario_Logging


# Initialize Super Mario environment
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

# Limit the action-space to:
#   0. Walk Right
#   1. Jump Right
env = JoypadSpace(env, [["right"], ["right", "A"]])
#, ["left"], ["left", "A"]

env.reset()
next_state, reward, done, info = env.step(action = 0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every 'skip' -th frame """
        super().__init__(env)
        self._skip = skip
        
    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Acumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
    
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low = 0, high = 255, shape = obs_shape, dtype = np.uint8)
        
    def permute_orientation(self, observation):
        # Permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype = torch.float)
        return observation
    
    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation
    
class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low = 0, high = 255, shape = obs_shape, dtype = np.uint8)
        
    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
            )
        observation = transforms(observation).squeeze(0)
        return observation
    
env = SkipFrame(env, skip = 4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape = 84)
env = FrameStack(env, num_stack = 4)
        
        
class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        
        self.use_cuda = torch.cuda.is_available()
        
        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        
        if self.use_cuda:
            self.net = self.net.to(device = "cuda")
            
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.9999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        
        self.save_every = 5e5 # number of experiences between saving Mario Net
        
        self.gamma = 0.9
        
        self.memory = deque(maxlen = 25000) # Tutorial used 100000/50000 but was too much memory for my GPU 10000 only uses 2gb
        self.batch_size = 32
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        self.burnin = 1e4 # Minimum experiences before training
        self.learn_every = 3 # Number of experiences between updates to Q_Online
        self.sync_every = 1e4 # Number of experiences between Q_target & Q_online sync
        
        self.advance_every = 20 # My own variable used to control when the AI uses only what it knows and no Random
        self.advance_optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.025)
        
    
    def act(self, state, advance = False):
        """Given a state, choose an epsilon-greedy action and update value of step.
        
        Inputs:
            state(LazyFrame_: A single observation of the current state, dimension is (state_dim)
        Outputs:
            action_idx (int): An integer representing which action Mario will perform
        """
        
        if advance:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model = "online")
            # print(action_values)
            action_idx = torch.argmax(action_values, axis = 1).item()
            # print(action_idx)
            
        else:
            # EXPLORE
            if np.random.rand() < self.exploration_rate:
                action_idx = np.random.randint(self.action_dim)
                
            # EXPLOIT
            else:
                state = state.__array__()
                if self.use_cuda:
                    state = torch.tensor(state).cuda()
                else:
                    state = torch.tensor(state)
                state = state.unsqueeze(0)
                action_values = self.net(state, model = "online")
                action_idx = torch.argmax(action_values, axis = 1).item()
            
            # Decrease exploration_rate
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        
        # Increment step
        self.curr_step += 1
        return action_idx
    
    def cache(self, state, next_state, action, reward, done):
        """Store the experience to self.memory (replay buffer)
        
        Inputs:
            state (LazyFrame),
            next_state (LazyFrame),
            action (int),
            reward (float),
            done (bool)
        """
        state = state.__array__()
        next_state = next_state.__array__()
        
        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])
            
        self.memory.append((state, next_state, action, reward, done,))
    
    def recall(self):
        """Retrieve a batch of experiences from memory"""
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    def td_estimate(self, state, action):
        current_Q = self.net(state, model = "online")[
            np.arange(0, self.batch_size), action
        ] # Q_online(s,a)
        
        # Test/Debug
        # print(self.net(state, model = "online"))
        # print(action)
        # print(current_Q)
        return current_Q
    
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model = "online")
        best_action = torch.argmax(next_state_Q, axis = 1)
        next_Q = self.net(next_state, model = "target")[
            np.arange(0, self.batch_size), best_action
        ]
        
        # print(done)
        # print(done.float())
        # Should probably invert the done.float() value as currently adding 1 when not completing
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def learn(self, advance = False):
        """Update online action value (Q) function with a batch of experiences"""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        
        if self.curr_step % self.save_every == 0:
            self.save()
            
        if self.curr_step < self.burnin:
            return None, None
        
        if self.curr_step % self.learn_every != 0:
            return None, None
        
        # Sample from memory
        state, next_state, action, reward, done = self.recall()
        
        # Get TD Estimate
        td_est = self.td_estimate(state, action)
        
        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)
        
        if advance:
            loss = self.update_Q_online(td_est, td_tgt, advance)
        else:
            # Backpropagate loss through Q_online
            loss = self.update_Q_online(td_est, td_tgt)
        
        return (td_est.mean().item(), loss)
    
    def update_Q_online(self, td_estimate, td_target, advance = False):
        loss = self.loss_fn(td_estimate, td_target)
        print(loss)
        if advance:
            self.advance_optimizer.zero_grad()
            loss.backward()
            self.advance_optimizer.step()
            return loss.item()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()
    
    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())
        
    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        
        torch.save(
            dict(model = self.net.state_dict(), exploration_rate = self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at set{self.curr_step}")
        
    

class MarioNet(nn.Module):
    """Mini CNN structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        
        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")
            
        self.online = nn.Sequential(
            nn.Conv2d(in_channels = c, out_channels = 32, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        
        self.target = copy.deepcopy(self.online)
        
        # Q-target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False
            
    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        
        elif model == "target":
            return self.target(input)
        
        
        
        
        
        
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Mario(state_dim = (4, 84, 84), action_dim = env.action_space.n, save_dir = save_dir)

logger = RL_Mario_Logging.MetricLogger(save_dir)

episodes = 100
current_high = 0

for e in range(episodes):
    state = env.reset()
    
    if e % mario.advance_every == 0:
        print("Playing advance")
        
    current_reward = 0
    # Play the game!
    while True:
        env.render()
        
        if e % mario.advance_every == 0:
            action = mario.act(state, True)
        else:
            # Run the agent on the state
            action = mario.act(state)
        
        # Agent performs action
        next_state, reward, done, info = env.step(action)
        current_reward += reward
        
        # Remember
        mario.cache(state, next_state, action, reward, done)
        
        if e % mario.advance_every == 0:
            q, loss = mario.learn(True)
        else:
            # Learn
            q, loss = mario.learn()
        
        # Logging
        logger.log_step(reward, loss, q)
        
        # Update state
        state = next_state
        
        # Check if end of game
        if done or info["flag_get"]:
            break
    
    logger.log_episode()
    
    if current_reward > current_high:
        current_high = current_reward
        print(f"Episode {e} is the new highest with reward of {current_high}")
    elif e % mario.advance_every == 0:
        print(f"Episode {e} got a reward of {current_reward}")
    
    if e % 20 == 0:
        # print(torch.cuda.memory_summary())
        logger.record(episode = e, epsilon = mario.exploration_rate, step = mario.curr_step)
        
        
        
env.close()        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        