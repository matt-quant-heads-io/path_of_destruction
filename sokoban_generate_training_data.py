import math
import time

from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym_pcgrl.envs.probs.sokoban_prob import SokobanProblem
from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
import csv

import hashlib
import numpy as np
import os
import struct
from gym import error
import random
from gym_pcgrl.wrappers import CroppedImagePCGRLWrapper
from gym.envs.classic_control import rendering
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.pcgrl_env import PcgrlEnv
import pandas as pd


import sys



gameCharacters = " #@$."
REV_TILES_MAP = dict((s, gameCharacters[i]) for i, s in enumerate(["empty", "solid", "player", "crate", "target"]))
print(REV_TILES_MAP)

# Reverse the k,v in TILES MAP for persisting back as char map .txt format
# REV_TILES_MAP = {"door": "g",
#                   "key": "+",
#                   "player": "A",
#                   "bat": "1",
#                   "spider": "2",
#                   "scorpion": "3",
#                   "solid": "w",
#                   "empty": "."}


TILES_MAP = {v:k for k,v in REV_TILES_MAP.items()}

print(TILES_MAP)
# TILES_MAP = {"g": "door",
#              "+": "key",
#              "A": "player",
#              "1": "bat",
#              "2": "spider",
#              "3": "scorpion",
#              "w": "solid",
#              ".": "empty"}

INT_MAP = {k[0]: i for i,k in enumerate(REV_TILES_MAP.items())}
print(INT_MAP)

# For hashing maps to avoid duplicate goal states
CHAR_MAP = {"door": 'a',
            "key": 'b',
            "player": 'c',
            "bat": 'd',
            "spider": 'e',
            "scorpion": 'f',
            "solid": 'g',
            "empty": 'h'}



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


# Converts from string[][] to 2d int[][]
def int_arr_from_str_arr(map):
    int_map = []
    for row_idx in range(len(map)):
        new_row = []
        for col_idx in range(len(map[0])):
            new_row.append(INT_MAP[map[row_idx][col_idx]])
        int_map.append(new_row)
    return int_map


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


def act_seq_to_disk(act_seq, path):
    with open(path, "w") as f:
        wr = csv.writer(f)
        wr.writerows(act_seq)


def act_seq_from_disk(path):
    act_seqs = []
    with open(path, "r") as f:
        data = f.readlines()
        for row in data:
            act_seq = [int(n) for n in row.split('\n')[0].split(',')]
            act_seqs.append(act_seq)
    return act_seqs




# Test reading in act_seq
# print(act_seq_from_disk('/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/exp_trajectories_const_generated/narrow/init_maps_lvl0/repair_sequence_0.csv'))

"""Start with random map"""
def gen_random_map(random, width, height, prob):
    map = random.choice(list(prob.keys()),size=(height,width),p=list(prob.values())).astype(np.uint8)
    return map


def _int_list_from_bigint(bigint):
    # Special case 0
    if bigint < 0:
        raise error.Error('Seed must be non-negative, not {}'.format(bigint))
    elif bigint == 0:
        return [0]

    ints = []
    while bigint > 0:
        bigint, mod = divmod(bigint, 2 ** 32)
        ints.append(mod)
    return ints

# TODO: don't hardcode sizeof_int here
def _bigint_from_bytes(bytes):
    sizeof_int = 4
    padding = sizeof_int - len(bytes) % sizeof_int
    bytes += b'\0' * padding
    int_count = int(len(bytes) / sizeof_int)
    unpacked = struct.unpack("{}I".format(int_count), bytes)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum

def create_seed(a=None, max_bytes=8):
    """Create a strong random seed. Otherwise, Python 2 would seed using
    the system time, which might be non-robust especially in the
    presence of concurrency.

    Args:
        a (Optional[int, str]): None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the seed.
    """
    # Adapted from https://svn.python.org/projects/python/tags/r32/Lib/random.py
    if a is None:
        a = _bigint_from_bytes(os.urandom(max_bytes))
    elif isinstance(a, str):
        a = a.encode('utf8')
        a += hashlib.sha512(a).digest()
        a = _bigint_from_bytes(a[:max_bytes])
    elif isinstance(a, int):
        a = a % 2**(8 * max_bytes)
    else:
        raise error.Error('Invalid type for seed: {} ({})'.format(type(a), a))

    return a

def hash_seed(seed=None, max_bytes=8):
    """Any given evaluation is likely to have many PRNG's active at
    once. (Most commonly, because the environment is running in
    multiple processes.) There's literature indicating that having
    linear correlations between seeds of multiple PRNG's can correlate
    the outputs:

    http://blogs.unity3d.com/2015/01/07/a-primer-on-repeatable-random-numbers/
    http://stackoverflow.com/questions/1554958/how-different-do-random-seeds-need-to-be
    http://dl.acm.org/citation.cfm?id=1276928

    Thus, for sanity we hash the seeds before using them. (This scheme
    is likely not crypto-strength, but it should be good enough to get
    rid of simple correlations.)

    Args:
        seed (Optional[int]): None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the hashed seed.
    """
    if seed is None:
        seed = create_seed(max_bytes=max_bytes)
    hash = hashlib.sha512(str(seed).encode('utf8')).digest()
    return _bigint_from_bytes(hash[:max_bytes])


def np_random(seed=None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise error.Error('Seed must be a non-negative integer or omitted, not {}'.format(seed))

    seed = create_seed(seed)

    rng = np.random.RandomState()
    rng.seed(_int_list_from_bigint(hash_seed(seed)))
    return rng, seed



def find_closest_goal_map(random_map, data_size):
    smallest_hamming_dist = math.inf
    closest_map = None
    filepath = 'sokoban/sokoban_{}.txt'
    map_indices = [i for i in range(data_size)]
    random.shuffle(map_indices)
    # print(f"shuffled map indices: {map_indices}")
    while len(map_indices) > 0:
        next_idx = map_indices.pop()
        curr_goal_map = int_arr_from_str_arr(to_2d_array_level(filepath.format(next_idx)))
        temp_hamm_distance = compute_hamm_dist(random_map, curr_goal_map)
        if temp_hamm_distance < smallest_hamming_dist:
            closest_map = curr_goal_map
    return closest_map


def compute_hamm_dist(random_map, goal):
    hamming_distance = 0.0
    for i in range(len(random_map)):
        for j in range(len(random_map[0])):
            if random_map[i][j] != goal[i][j]:
                hamming_distance += 1
    return float(hamming_distance / (len(random_map) * len(random_map[0])))


def transform(obs, x, y, crop_size):
    map = obs
    # View Centering
    size = crop_size
    pad = crop_size // 2
    padded = np.pad(map, pad, constant_values=1)
    cropped = padded[y:y + size, x:x + size]
    obs = cropped
    new_obs = []
    for i in range(len(obs)):
        for j in range(len(obs[0])):
            new_tile = [0]*8
            new_tile[obs[i][j]] = 1
            new_obs.extend(new_tile)
    return new_obs

def str_arr_from_int_arr(map):
    translation_map = {v:k for k,v in INT_MAP.items()}
    str_map = []
    for row_idx in range(len(map)):
        new_row = []
        for col_idx in range(len(map[0])):
            new_row.append(translation_map[map[row_idx][col_idx]])
        str_map.append(new_row)
    # print(str_map)
    return str_map


def generate_pod_greedy(env, random_target_map, goal_starting_map, total_steps, ep_len=77, crop_size=9, render=True, epsilon=0.02):
    """
        The "no-change" action  is 1 greater than the number of tile types (value is 8)
    """

    play_trace = []
    # loop through from 0 to 13 (for 14 tile change actions)
    old_map = goal_starting_map.copy()
    random_map = random_target_map.copy()

    current_loc = [0, 0]
    env._rep._old_map = np.array([np.array(l) for l in goal_starting_map])
    env._rep._x = current_loc[1] # 0
    env._rep._y = current_loc[0] # 0
    row_idx, col_idx = current_loc[0], current_loc[1]
    tile_count = 0

    hamm = compute_hamm_dist(random_target_map, goal_starting_map)
    curr_step = 0
    episode_len = ep_len
    env.reset()
    env.reset()
    while hamm > 0.0 and curr_step <= episode_len and total_steps < 962500:
        new_map = old_map.copy()
        transition_info_at_step = [None, None, None]
        rep._x = col_idx
        rep._y = row_idx

        new_map[row_idx] = old_map[row_idx].copy()


        # TODO: change this to select new tile via:
        # TODO: sample an action
        # TODO: check if it gets us closer to starting distribution
        # TODO: if no then scratch action, if yes then continue

        # existing tile type on the goal map
        old_tile_type = old_map[row_idx][col_idx]

        # new destructive tile
        new_tile_type = random_target_map[row_idx][col_idx]

        expert_action = [row_idx, col_idx, old_tile_type]
        destructive_action = [row_idx, col_idx, new_tile_type]
        transition_info_at_step[1] = destructive_action.copy()
        transition_info_at_step[2] = expert_action.copy()
        new_map[row_idx][col_idx] = new_tile_type


        # play_trace.append((transform(old_map.copy(), col_idx,  row_idx, crop_size), expert_action.copy()))
        play_trace.append((transform(random_map.copy(), col_idx, row_idx, crop_size), expert_action.copy()))
        random_map[row_idx][col_idx] = old_tile_type
        curr_step += 1
        total_steps += 1

        old_map = new_map

        tile_count += 1
        col_idx += 1
        if col_idx >= 11:
            col_idx = 0
            row_idx += 1
            if row_idx >= 7:
                row_idx = 0

        hamm = compute_hamm_dist(random_target_map, old_map)
        if hamm == 0.0:
            play_trace.reverse()
            return play_trace, total_steps

    play_trace.reverse()
    return play_trace, total_steps


# This code is for generating the maps
def render_map(map, prob, rep, filename='', ret_image=False, pause=True):
    # format image of map for rendering
    if not filename:
        img = prob.render(map)
    else:
        img = to_2d_array_level(filename)
    img = rep.render(img, tile_size=16, border_size=(1, 1)).convert("RGB")
    img = np.array(img)
    if ret_image:
        return img
    else:
        ren = rendering.SimpleImageViewer()
        ren.imshow(img)
        # time.sleep(0.3)
        if pause:
            input(f'')
        else:
            time.sleep(.05)
        ren.close()





def int_map_to_str_map(curr_map):
    new_level = []
    for idx, row in enumerate(curr_map):
        new_row = []
        for j_idx, col in enumerate(row):
            new_row.append(REV_INT_MAP[col])
        new_level.append(new_row)
    return new_level




# Read in params:
# tile distribution for zelda_prob
# define epsilon (error)
# # TODO: Need to change this for Turtle and Narrow Reps
actions_list = [act for act in list(TILES_MAP.values())]
prob = SokobanProblem()
rep = NarrowRepresentation()


obs_ep_comobs = [(5, 77, 5)]
rng, seed = np_random(None)
epsilon = 0.1
filepath = 'sokoban/sokoban_{}.txt'
zelda_tile_distrib = {0: 0.58, 1: 0.3, 2: 0.02, 3: 0.02, 4: 0.02, 5: 0.02, 6: 0.02, 7: 0.02}

#
for obs_size, episode_len, goal_set_size in obs_ep_comobs:
    dict_len = ((obs_size ** 2) * 8)
    total_steps = 0
    exp_traj_dict = {f"col_{i}": [] for i in range(dict_len)}
    exp_traj_dict["target"] = []
    save_count = 0
    while total_steps < 962500:
        save_count += 1
        play_traces = []
        cropped_wrapper = CroppedImagePCGRLWrapper("sokoban-narrow-v0", obs_size,
                                                   **{'change_percentage': 1, 'trials': 1, 'verbose': True,
                                                    'cropped_size': obs_size, 'render': False})
        pcgrl_env = cropped_wrapper.pcgrl_env
        start_map = gen_random_map(rng, 5, 5, {0: 0.45, 1: 0.4, 2: 0.05, 3: 0.05, 4: 0.05})
        goal_map = find_closest_goal_map(start_map, goal_set_size)
        play_trace, temp_num_steps = generate_pod_greedy(pcgrl_env, start_map, goal_map, total_steps, ep_len=episode_len, crop_size=obs_size, render=False)
        total_steps = temp_num_steps
        print(f"(obs={obs_size}, ep_len={episode_len}, data_size={goal_set_size}), total_steps: {total_steps}")
        play_traces.append(play_trace)


        for p_i in play_trace:
            # print(f"p_i is {p_i}")
            # print(f"len of p_i is {len(p_i)}")
            action = p_i[1][-1]
            # print(f"action is {action}")
            exp_traj_dict["target"].append(action)
            pt = p_i[0]
            # print(f"pt is {pt}")
            # print(f"len of pt is {len(pt)}")
            assert dict_len == len(pt), f"len(pt) is {len(pt)} and dict_len is {dict_len}"
            for i in range(len(pt)):
                exp_traj_dict[f"col_{i}"].append(pt[i])

        if save_count % 250 == 0:
            print(f"saving df at ts {total_steps}")
            df = pd.DataFrame(data=exp_traj_dict)
            df.to_csv(
                f"sokoban_exp_traj_obs_{obs_size}_ep_len_77_goal_size_{goal_set_size}/pod_exp_traj_goal_step_{total_steps}.csv",
                index=False)
            exp_traj_dict = {f"col_{i}": [] for i in range(dict_len)}
            exp_traj_dict["target"] = []

    df = pd.DataFrame(data=exp_traj_dict)
    df.to_csv(f"sokoban_exp_traj_obs_{obs_size}_ep_len_77_goal_size_{goal_set_size}/pod_exp_traj_goal_step_{total_steps}.csv", index=False)








