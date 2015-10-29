#!/usr/bin/env python3

import numpy as np
import random
import multiprocessing
import itertools
import functools
import time

"""def simulate(steps):
    pos=np.zeros(2)
    for i in range(steps):
        pos += random.choice([(-1,0),(1,0),(0,-1),(0,1)])
    return pos"""

def simulate(steps):
    to_return = np.zeroes((steps, 2))
    for i in range(steps-1):
        to_return[i+1,:] = to_return[i,:] + random.choice([(-1,0),(1,0),(0,-1),(0,1)])
    return to_return

def thresholder(f, threshold):
    x, y = f()
    return np.sqrt((x**2+y**2)) >= threshold

def Q1(f, steps, threshold, iterations):
    p = multiprocessing.Pool(multiprocessing.cpu_count)
    f = functools.partial(f, functools.partial(simulate, steps))
    results = list()
    data_set = np.ndarray((iterations, steps, 2))
    for s in range(steps-1):
        data_set[:, s+1, :] = data_set[:, s, :] + np.vstack(
                np.random.choice([-1, 0, 1, 0]),
                np.random.choice([-1, 0, 1, 0]))
    #result[:,:,:] = np.asarray(p.map(f, itertools.repeat(threshold, iterations)))
    return (np.sum(result)/result.size)

def gen_data(iterations, steps):
    data_set = np.ndarray((iterations, steps, 2))
    optionsx = [1, 0, -1, 0]
    optionsy = [1, -1]
    for s in range(steps-1):
        stepsx = np.random.choice(optionsx, iterations)
        stepsy = np.random.choice(optionsy, iterations)
        stepsy[stepsx!=0]=0
        data_set[:, s+1, 0] = data_set[:, s, 0] + stepsx
        data_set[:, s+1, 1] = data_set[:, s, 1] + stepsy


#timed = functools.partial(Q1, thresholder, 10, 3, 10000)
start = time.clock()
print(gen_data(1000000, 60))
total = time.clock()-start
print(total)
#print(Q1(thresholder, 10, 3, 1000000))

