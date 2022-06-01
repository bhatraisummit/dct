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

def zero_noise_loc (arr, percent, SEED):
    final_shape = arr.shape
    total_size = 1
    for i in final_shape:
        total_size *= i
    random.seed(SEED)
    indexs = random.sample(range(total_size), math.ceil((100 - percent) / 100. * total_size))
    locs = np.where(~np.in1d(range(total_size),indexs))
    arr = arr.flatten()
    arr[locs] = 0
    return arr.reshape(final_shape)

def remove_noise_percent (arr, std, percent, SEED, SEED_N):
    np.random.seed(SEED_N)
    noise = np.random.normal(0, std, size=arr.shape)
    flat_noise = noise.flatten()
    random.seed(SEED)
    indexs = random.sample(range(flat_noise.shape[0]), math.ceil((100 - percent) / 100. * flat_noise.shape[0]))
    flat_noise[indexs] = 0
    return arr - flat_noise.reshape(arr.shape)


# import numpy
# target_list = numpy.array(['1','b','c','d','e','f','g','h','i','j'])
# to_exclude = [1,4,5]
# print target_list[~numpy.in1d(range(len(target_list)),to_exclude)]