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

def euclidean(data):
    return np.sqrt(data[:, 0]**2 + data[:, 1]**2)

def at_least_at(data_set, step, distance):
    return (np.sum(
        euclidean(data_set[:, step-1, :]) >= distance)
        ) / data_set.shape[0]

def at_least_by(data_set, step, distance):
    passed = np.zeros((data_set.shape[0]), dtype=bool)
    for s in range(step):
        passed[euclidean(data_set[:, s, :]) >= distance] = 1
    return np.sum(passed)/passed.size

def across_the_iron_curtain(data_set, step):
    east = np.zeros((data_set.shape[0]), dtype=bool)
    west = np.zeros((data_set.shape[0]), dtype=bool)
    for s in range(step):
        east[data_set[:,s,0]>0] = True
    west[np.logical_and(east, data_set[:,step-1,0]<0)] = True
    return np.sum(west)/west.size

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
    return data_set


#timed = functools.partial(Q1, thresholder, 10, 3, 10000)
print("Generating data...")
MAX_SIZE = 10000000
data_set = gen_data(MAX_SIZE, 60)

print("Q1:")
print(at_least_at(data_set, 10, 3))

print("Q2:")
print(at_least_at(data_set, 60, 10))

print("Q3:")
print(at_least_by(data_set, 10, 5))

print("Q4:")
print(at_least_by(data_set, 60, 10))

print("Q5:")
print(across_the_iron_curtain(data_set, 10))

print("Q6:")
print(across_the_iron_curtain(data_set, 30))
