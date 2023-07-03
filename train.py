from random import random, randint, sample
import numpy as np
import torch
from dqnn import DQNN
from collections import deque
import gym
import numpy as np
import envs
from player import get_next_states_actions
env = gym.make('Tetris-v0', disable_env_checker=True)

def train():

    ###### Training parameters ######
    lr = 1e-3
    gamma = 0.995
    final_epsilon = 2e-3
    initial_epsilon = 1
    decay_epochs = 2000
    total_epochs = 100000
    replay_memory_length = 100000
    batch_size = 512
    #################################

    save_epochs = 1000
    save_path = "trained_NN"

    torch.manual_seed(123)
    model = DQNN()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = torch.nn.MSELoss()
    plot_freq = 100
    avg_lines = 0
    loss_cum = 0


    env.reset()
    state = torch.FloatTensor([0, 0, 0, 0])

    replay_memory = deque(maxlen=replay_memory_length)
    epoch = 0
    total_reward = 0
    while epoch < total_epochs:
        next_states_actions = get_next_states_actions(env)
        if not next_states_actions:
            possible_actions = env.state.get_action()
            print(possible_actions)
            next_states_actions = {(0, 5): torch.FloatTensor([-1, -1, -1, -1])}

        epsilon = final_epsilon + (max(decay_epochs - epoch, 0) * (
                initial_epsilon - final_epsilon) / decay_epochs)
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_states_actions.items())
        next_states = torch.stack(next_states)
        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()
        if random_action:
            index = randint(0, len(next_states) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]
        action = next_actions[index]

        _, reward, done, _ = env.step(action)
        total_reward += reward
        replay_memory.append([state, reward, next_state, done])

        if done:
            final_cleared_lines = env.state.cleared
            avg_lines += final_cleared_lines
            env.reset()
            state = torch.FloatTensor([0, 0, 0, 0])
        else:
            state = next_state
            continue
        if len(replay_memory) < replay_memory_length/10:
            continue
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(statei for statei in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(statei for statei in next_state_batch))

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        loss_cum += loss

        total_reward = 0

        if epoch > 0 and epoch % plot_freq == 0:
            lines = avg_lines/plot_freq
            print("Epoch: {}/{}, Average lines: {}, Loss: {}".format(
            epoch,
            total_epochs,
            lines,
            loss_cum/20
            ))
            avg_lines = 0
            loss_cum = 0

        if epoch > 0 and epoch % save_epochs == 0:
            torch.save(model, "{}/tetris_{}".format(save_path, epoch))

    torch.save(model, "{}/tetris".format(save_path))

if __name__ == "__main__":
    train()