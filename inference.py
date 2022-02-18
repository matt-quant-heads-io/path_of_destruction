"""
Run a trained agent and get generated maps
"""
import os
import model
from stable_baselines import PPO2

import time
from utils import make_vec_envs
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

from decimal import Decimal as D
from decimal import getcontext
getcontext().prec = 8


TILES_MAP = {"g": "door",
             "+": "key",
             "A": "player",
             "1": "bat",
             "2": "spider",
             "3": "scorpion",
             "w": "solid",
             ".": "empty"}

INT_MAP = {
    "empty": 0,
    "solid": 1,
    "player": 2,
    "key": 3,
    "door": 4,
    "bat": 5,
    "scorpion": 6,
    "spider": 7
}

# For hashing maps to avoid duplicate goal states
CHAR_MAP = {"door": 'a',
            "key": 'b',
            "player": 'c',
            "bat": 'd',
            "spider": 'e',
            "scorpion": 'f',
            "solid": 'g',
            "empty": 'h'}

REV_TILES_MAP = { "door": "g",
                  "key": "+",
                  "player": "A",
                  "bat": "1",
                  "spider": "2",
                  "scorpion": "3",
                  "solid": "w",
                  "empty": "."}


def to_char_level(map, dir=''):
    level = []

    for row in map:
        new_row = []
        for col in row:
            new_row.append(REV_TILES_MAP[col])
        # add side borders
        new_row.insert(0, 'w')
        new_row.append('w')
        level.append(new_row)
    top_bottom_border = ['w'] * len(level[0])
    level.insert(0, top_bottom_border)
    level.append(top_bottom_border)

    level_as_str = []
    for row in level:
        level_as_str.append(''.join(row) + '\n')

    with open(dir, 'w') as f:
        for row in level_as_str:
            f.write(row)
    f.close()





def transform_narrow(obs, x, y, return_onehot=True, transform=True):
    pad = 11
    pad_value = 1
    size = 22
    map = obs # obs is int
    # View Centering
    padded = np.pad(map, pad, constant_values=pad_value)
    cropped = padded[y:y + size, x:x + size]
    obs = cropped

    if return_onehot:
        obs = np.eye(8)[obs]
        if transform:
            new_obs = []
            for i in range(22):
                for j in range(22):
                    for z in range(8):
                        new_obs.append(obs[i][j][z])
            return new_obs
    return obs


def int_map_to_onehot(int_map):
    new_map = []
    for row_i in range(len(int_map)):
        new_row = []
        for col_i in range(len(int_map[0])):
            new_tile = [0]*8
            new_tile[int_map[row_i][col_i]] = 1
            new_row.append(np.array(new_tile))
        new_map.append(np.array(new_row))
    return np.array(new_map)


# Reads in .txt playable map and converts it to string[][]
def to_2d_array_level(file_name):
    level = []

    with open(file_name, 'r') as f:
        rows = f.readlines()
        for row in rows:
            new_row = []
            for char in row:
                if char != '\n':
                    new_row.append(TILES_MAP[char])
            level.append(new_row)

    # Remove the border
    truncated_level = level[1: len(level) - 1]
    level = []
    for row in truncated_level:
        new_row = row[1: len(row) - 1]
        level.append(new_row)
    return level

def compute_hamm_dist(random_map, goal):
    hamming_distance = 0.0
    random_map = list(random_map[0])
    for i in range(len(goal)):
        for j in range(len(goal[0])):
            for k in range(8):
                if random_map[i][j][k] != goal[i][j][k]:
                    hamming_distance += 1
    return float(hamming_distance / (len(random_map) * len(random_map[0])))


def transform_narrow(obs, x, y, obs_size=9, return_onehot=True, transform=True):
    pad = 22 - (22 - obs_size)
    pad_value = 1
    size = 22
    map = obs # obs is int
    # View Centering
    padded = np.pad(map, pad, constant_values=pad_value)
    cropped = padded[y:y + size, x:x + size]
    obs = cropped

    # if return_onehot:
    #     obs = np.eye(8)[obs]
    #     if transform:
    #         new_obs = []
    #         for i in range(22):
    #             for j in range(22):
    #                 for z in range(8):
    #                     new_obs.append(obs[i][j][z])
    #         return new_obs
    return obs


# Converts from string[][] to 2d int[][]
def int_arr_from_str_arr(map):
    int_map = []
    for row_idx in range(len(map)):
        new_row = []
        for col_idx in range(len(map[0])):
            new_row.append(INT_MAP[map[row_idx][col_idx]])
        int_map.append(new_row)
    return int_map

map_num_to_oh_dict = {}
root_dir = '/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/playable_maps'



def infer(game, representation, model_path, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    env_name = '{}-{}-v0'.format(game, representation)
    if game == "binary":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
        # kwargs['cropped_size'] = 22
        kwargs['cropped_size'] = 5
    elif game == "sokoban":
        model.FullyConvPolicy = model.FullyConvPolicySmallMap
        kwargs['cropped_size'] = 10
    kwargs['render'] = True

    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    obs = env.reset()
    obs = env.reset()
    dones = False
    success_count = 0
    # build map dominance plot
    build_map_dom_plot = False
    mode = 'mlp'
    non_random_start = False
    # Random start
    success_count = 0
    bounds = obs_size = kwargs['cropped_size']
    ep_len = 77
    goal_map_size = 1
    version = 3
    action_mode = 'nongreedy'
    if not non_random_start:
        agent = keras.models.load_model(f'model_obs_{obs_size}_goal_size_{goal_map_size}_model_num_{version}.h5')
        for i in range(kwargs.get('trials', 1)):
            print(f"trial {i}")
            while not dones:
                if mode == 'mlp':
                    try:
                        if action_mode == 'greedy':
                            action = np.argmax(agent.predict(np.array([obs[0]]))) + 1
                        else:
                            pred_probs = list(agent.predict(np.array([obs[0]]))[0])
                            sum_pred_probs = sum(pred_probs)
                            a = D(1 / sum_pred_probs)
                            pred_probs_scaled = [D(float(e)) * a for e in pred_probs]
                            action = np.random.choice(8, 1, p=pred_probs_scaled) + 1
                    except Exception:
                        print(f"probs dont sum to 1")
                        action = np.argmax(agent.predict(np.array([obs[0]]))[0]) + 1
                else:
                    action = [0]
                obs, _, dones, info = env.step([action])
                if info[0]["solved"]:
                    success_count += 1
                    final_map = info[0]["final_map"]
                    level_str = ''

                    for row in final_map:
                        new_row = []
                        for col in row:
                            level_str += REV_TILES_MAP[col]


                    f = open(f'pod_playable_obs_{obs_size}_goal_size_{goal_map_size}_{version}/{success_count}.txt', 'w')
                    f.write(level_str)
                    f.close()
                if kwargs.get('verbose', False):
                    pass
                if dones:
                    break
            dones = False
            obs = env.reset()
            obs = env.reset()
            success_pct = success_count / (i+1)
            print(f"(obs={obs_size}) trial {i+1}, success_pct: {success_pct}")


    success_pct = success_count / kwargs['trials']
    print(f"success pct is {success_pct}")


################################## MAIN ########################################
game = 'zelda'
representation = 'narrow'
model_path = 'models/{}/{}/model_1.pkl'.format(game, representation)
kwargs = {
    'change_percentage': 1, #0.4,
    'trials': 10000,
    'verbose': True
}

if __name__ == '__main__':
    infer(game, representation, model_path, **kwargs)