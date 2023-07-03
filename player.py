import gym
import numpy as np
import envs
from time import sleep
import torch

env = gym.make('Tetris-v0', disable_env_checker=True)


def step_predict(env, action):
    """
    predict next state for an action
    """
    orient, slot = action

    next_state = env.state.copy()
    next_state.turn += 1

    # height of the field
    height = max(
        next_state.top[slot+c] - env.piece_bottom[next_state.next_piece][orient][c]
        for c in range(env.piece_width[next_state.next_piece][orient])
    )

    # for each column in the piece - fill in the appropriate blocks
    for i in range(env.piece_width[next_state.next_piece][orient]):
        # from bottom to top of brick
        for h in range(height + env.piece_bottom[next_state.next_piece][orient][i], height + env.piece_top[next_state.next_piece][orient][i]):
            if h >= env.n_rows: h = env.n_rows - 1
            next_state.field[h, i+slot] = next_state.turn

    # adjust top
    for c in range(env.piece_width[next_state.next_piece][orient]):
        h = height + env.piece_top[next_state.next_piece][orient][c]
        if h >= env.n_rows: h = env.n_rows - 1
        next_state.top[slot+c] = h

    # check for full rows - starting at the top
    env.cleared_current_turn = 0
    for r in range(height + env.piece_height[next_state.next_piece][orient] - 1, height - 1, -1):
        if r >= env.n_rows: r = env.n_rows - 1
        # if the row was full - remove it and slide above stuff down
        if np.all(next_state.field[r] > 0):
            env.cleared_current_turn += 1
            next_state.cleared += 1
            # for each column
            for c in range(env.n_cols):
                # slide down all bricks
                next_state.field[r:next_state.top[c], c] = next_state.field[(r+1):(next_state.top[c]+1), c]
                # lower the top
                next_state.top[c] -= 1
                while next_state.top[c] >= 1 and next_state.field[next_state.top[c]-1, c] == 0:
                    next_state.top[c] -= 1

    total_height = sum(next_state.top)
    bumpiness = np.sum(np.abs(next_state.top[1:] - next_state.top[:-1]))

    num_holes = 0
    for c in range(env.n_cols):
        col = next_state.field[:, c]
        r = env.n_rows - 1
        while r > 0 and col[r] == 0:
            r -= 1
        num_holes += len([x for x in col[:r] if x == 0])

    return False, torch.FloatTensor([env.cleared_current_turn, num_holes, bumpiness, total_height])

def get_next_states_actions(env):
    actions = env.get_actions()
    states_and_actions = {}

    for action in actions:
        gameover, state = step_predict(env, action)
        if gameover:
            continue
        states_and_actions[tuple(action)] = state
    return states_and_actions

def evaluate_player(num_games):
    total_lines_cleared = 0
    env.reset()
    model = torch.load("trained_model")
    for i in range(num_games):
        env.reset()
        # env.render()
        done = False
        total_reward = 0
        while not done:
            next_states_actions = get_next_states_actions(env)
            if not next_states_actions:
                next_states_actions = {(0, 5): torch.FloatTensor([-1, -1, -1, -1])}

            next_actions, next_states = zip(*next_states_actions.items())
            next_states = torch.stack(next_states)
            model.eval()
            with torch.no_grad():
                predictions = model(next_states)[:, 0]

            index = torch.argmax(predictions).item()
            action = next_actions[index]
            _, reward, done, _ = env.step(action)
            if done:
                total_lines_cleared += env.state.cleared
                print("Game", i, "cleared lines:", env.state.cleared, "  Total lines cleared:", total_lines_cleared, "  Reward:", total_reward )

    average_lines_cleared = total_lines_cleared/num_games
    print("Player cleared an average of", average_lines_cleared, "lines.")
    return

num_games = 20
evaluate_player(num_games)