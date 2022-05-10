import numpy as np
import random
import math


def add_noise_percent(arr, std, percent, SEED, SEED_N):
    np.random.seed(SEED_N)
    noise = np.random.normal(0, std, size=arr.shape)
    flat_noise = noise.flatten()
    random.seed(SEED)
    indexs = random.sample(range(flat_noise.shape[0]), math.ceil((100 - percent) / 100. * flat_noise.shape[0]))
    flat_noise[indexs] = 0
    return arr + flat_noise.reshape(arr.shape)

def remove_noise_percent (arr, std, percent, SEED, SEED_N):
    np.random.seed(SEED_N)
    noise = np.random.normal(0, std, size=arr.shape)
    flat_noise = noise.flatten()
    random.seed(SEED)
    indexs = random.sample(range(flat_noise.shape[0]), math.ceil((100 - percent) / 100. * flat_noise.shape[0]))
    flat_noise[indexs] = 0
    return arr - flat_noise.reshape(arr.shape)
