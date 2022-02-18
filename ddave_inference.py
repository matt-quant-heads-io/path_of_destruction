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


gameCharacters = " #@H$V*"
REV_TILES_MAP = dict((s, gameCharacters[i]) for i, s in enumerate(["empty", "solid", "player", "exit", "diamond", "key", "spike"]))
print(REV_TILES_MAP)

TILES_MAP = {v:k for k,v in REV_TILES_MAP.items()}

print(TILES_MAP)


INT_MAP = {k[0]: i for i,k in enumerate(REV_TILES_MAP.items())}
print(INT_MAP)



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
        obs = np.eye(7)[obs]
        if transform:
            new_obs = []
            for i in range(22):
                for j in range(22):
                    for z in range(7):
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
        kwargs['cropped_size'] = 15
    elif game == "sokoban":
        model.FullyConvPolicy = model.FullyConvPolicySmallMap
        kwargs['cropped_size'] = 10
    kwargs['cropped_size'] = 5
    kwargs['render'] = False

    env = make_vec_envs(env_name, representation, None, 1, **kwargs)

    obs = env.reset()
    obs = env.reset()
    dones = False
    success_count = 0
    build_map_dom_plot = False
    mode = 'mlp'
    non_random_start = False
    # Random start
    success_count = 0
    bounds = obs_size = 5
    ep_len = 77
    goal_map_size = 5
    version = 1
    if not non_random_start:
        agent = keras.models.load_model(f'ddave_model_obs_{obs_size}_goal_size_{goal_map_size}_model_num_{}.h5')
        for i in range(kwargs.get('trials', 1)):
            print(f"trial {i}")
            while not dones:
                # flatten obs
                if mode == 'mlp':
                    new_obs = []
                    action = np.argmax(agent.predict(np.array([obs[0]]))) + 1
                else:
                    action = [0]
                obs, _, dones, info = env.step([action])
                # print(f"done is {dones}")
                # print(f"trial: {i+1}, success: {success_count}:  {info}")
                if info[0]["solved"]:
                    # print(f'{info[0]["final_map"]}')
                    # print(f'{np.array(info[0]["final_map"]).shape}')
                    # print(env._obs_from_buf())
                    # print(env._obs_from_buf().shape)
                    # print(env.metadata)
                    # input('')
                    # print(f"success!")
                    success_count += 1
                    final_map = info[0]["final_map"]
                    level_str = ''
                    for row in final_map:
                        new_row = []
                        for col in row:
                            level_str += REV_TILES_MAP[col]

                    f = open(f'ddave_playable_maps_obs_{obs_size}_ep_len_{ep_len}_{goal_map_size}_{version}/{success_count}.txt', 'w')
                    f.write(level_str)
                    f.close()
                    # ddave_playable_maps_obs_15_ep_len_77_5
                    # map_num_closest = -1
                    # closest_hamm_dist = math.inf
                    # # print(f"x is {info[0]['x']}, y is {info[0]['y']}")
                    # for map_number, map in map_num_to_oh_dict.items():
                    #     curr_hamm_dist = compute_hamm_dist(transform_narrow(obs, info[0]['x'], info[0]['y'], obs_size=obs_size, return_onehot=True, transform=True), map)
                    #     if curr_hamm_dist < closest_hamm_dist:
                    #         map_num_closest = map_number
                    #         closest_hamm_dist = curr_hamm_dist
                    #
                    # map_dom_dict[map_num_closest][0] += 1
                    # input('')
                if kwargs.get('verbose', False):
                    pass
                    # print(info[0])
                if dones:
                    break
            dones = False
            obs = env.reset()
            obs = env.reset()
            # time.sleep(0.2)
            success_pct = success_count / (i+1)
            print(f"(obs={obs_size}) trial {i+1}, success_pct: {success_pct}")
    else:
        # agent = keras.models.load_model(
        #     '/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/narrow_best_max_acc_5_goals_500_starts_reg_fit_incomplete.h5')
        agent = keras.models.load_model('/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/narrow_greedy_best_25_goal_maps_per_50_training_maps.h5')
        # agent = keras.models.load_model('/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/narrow_best_max_acc_50_goals_5_starts_overfit.h5')
        # TODO: This tests NON sequentially trained model "MODEL 2"
        # agent = keras.models.load_model(
        #     '/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/narrow_best_max_acc_4_goals_5_starts_randomized.h5')

        # TODO: This tests sequentially trained model on fixed 0 to 4  starting maps goal map 0 (MODEL 1)
        # agent = keras.models.load_model('/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/narrow_best_max_acc_1_goal_5_starts.h5')

        # TODO: This tests NON sequentially trained model on fixed 0 to 4  starting maps goal map 0 (MODEL 1B)
        # agent = keras.models.load_model('/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/narrow_best_max_acc_1_goal_5_starts_randomized.h5')
        # start_map = "init_map_0.txt"
        # obs = int_arr_from_str_arr(to_2d_array_level(f'/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/exp_trajectories_const_generated/narrow/init_maps_lvl1/{start_map}'))
        start_map = "init_map_47.txt"
        obs = int_arr_from_str_arr(
            to_2d_array_level(
                f'/gym-pcgil/gym_pcgil/exp_trajectories_const_generated/narrow_greedy/init_maps_lvl4/{start_map}'))
        obs = np.array([transform_narrow(obs, 0, 0, return_onehot=True, transform=False)])
        for i in range(kwargs.get('trials', 1)):
            # print(f"obs is {obs}")
            # print(f"obs shape is {obs.shape}")
            # break
            while not dones:
                # flatten obs
                # input('')
                if mode == 'mlp':
                    new_obs = []
                    for l in range(bounds):
                        for j in range(bounds):
                            for z in range(8):
                                new_obs.append(obs[0][l][j][z])
                    action = np.argmax(agent.predict(np.array([new_obs]))) + 1
                    # print(f"action is {action}")
                    # action, _ = agent.predict(obs)
                else:
                    action = [0]
                obs, _, dones, info = env.step([action])
                # print(f"done is {dones}")
                # print(f"trial: {i+1}, success: {success_count}:  {info}")
                if info[0]["solved"]:
                    # print(f"success!")
                    success_count += 1
                    # increment closest map in the dictionary
                    map_num_closest = -1
                    closest_hamm_dist = math.inf
                    for map_number, map in map_num_to_oh_dict.items():
                        curr_hamm_dist = compute_hamm_dist(obs, map)
                        if curr_hamm_dist < closest_hamm_dist:
                            map_num_closest = map_number
                            closest_hamm_dist = curr_hamm_dist

                    map_dom_dict[map_num_closest][0] += 1
                if kwargs.get('verbose', False):
                    pass
                    # print(info[0])
                if dones:
                    break
            dones = False
            obs = env.reset()
            obs = int_arr_from_str_arr(to_2d_array_level(
                f'/gym-pcgil/gym_pcgil/exp_trajectories/narrow_greedy/init_maps_lvl4/{start_map}'))
            obs = np.array([transform_narrow(obs, 0, 0, return_onehot=True, transform=False)])
            time.sleep(0.2)


    success_pct = success_count / kwargs['trials']
    # success_dict = {"success_pct": [success_pct]}
    # df_succ = pd.DataFrame(data=success_dict)
    # df_succ.to_csv(f"sucess_pct_obs_size_{obs_size}_ep_len_{ep_len}.csv", index=False)
    print(f"success pct is {success_pct}")
    # df = pd.DataFrame(data=map_dom_dict)
    # df.to_csv(f"map_dominance_obs_size_{obs_size}_ep_len_{ep_len}.csv" , index=False)

    # print(f"% playable levels generated: {(success_count/float(kwargs['trials']))*100}%")
    # plt.bar(range(len(map_dom_dict)), list(map_dom_dict.values()), align='center')
    # plt.title(f'Map Dominance {(success_count/float(kwargs["trials"]))*100}')
    # plt.xlabel('Goal Maps')
    # plt.ylabel('Dominance')
    # plt.xticks(range(len(map_dom_dict)), list(map_dom_dict.keys()))
    # # # for python 2.x:
    # # plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
    # # plt.xticks(range(len(D)), D.keys())  # in python 2.x
    # if build_map_dom_plot:
    #     plt.show()
    #     plt.savefig('map_dominance1.png')

################################## MAIN ########################################
game = 'ddave'
representation = 'narrow'
model_path = 'models/{}/{}/model_1.pkl'.format(game, representation)
kwargs = {
    'change_percentage': 1, #0.4,
    'trials': 10000,
    'verbose': True
}

if __name__ == '__main__':
    infer(game, representation, model_path, **kwargs)

