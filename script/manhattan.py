#!/usr/bin/env python3

import numpy as np
import random
import multiprocessing
import itertools
import functools
import time

def simulate(steps):
    pos=np.zeros(2)
    for i in range(steps):
        pos += random.choice([(-1,0),(1,0),(0,-1),(0,1)])
    return pos

def generator_simulate(steps, pos=np.zeros(2)):
    for i in range(steps):
        pos += random.choice([(-1,0),(1,0),(0,-1),(0,1)])
        yield pos

def thresholder(f, threshold):
    x, y = f()
    return np.sqrt((x**2+y**2)) >= threshold

def Q1(f, steps, threshold, iterations):
    p = multiprocessing.Pool(8)
    f = functools.partial(f, functools.partial(simulate, steps))
    result = np.ndarray((iterations))
    result[:] = p.map(f, itertools.repeat(threshold, iterations))
    return np.sum(result)/result.size

timed = functools.partial(Q1, thresholder, 10, 3, 10000000)
start = time.clock()
print(timed())
total = time.clock()-start
print(total)
#print(Q1(thresholder, 10, 3, 1000000))

