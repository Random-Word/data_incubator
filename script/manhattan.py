#!/usr/bin/env python3

import numpy as np
import random
import multiprocessing
import itertools
import functools
import time

MAX_SIZE = 10000000
STEPS = 60
THREADS = 15

def euclidean(data):
    return np.sqrt((data[:, 0].astype(np.float64,copy=False)**2)+
            (data[:, 1].astype(np.float64,copy=False)**2))

def at_least_at(data_set, step, distance):
    total = np.sum(
        euclidean(data_set[:, step-1, :]) >= distance)
    return total, data_set.shape[0], total/data_set.shape[0]

def at_least_by(data_set, step, distance):
    passed = np.zeros((data_set.shape[0]), dtype=bool)
    for s in range(step):
        passed[euclidean(data_set[:, s, :]) >= distance] = 1
    return np.sum(passed), passed.size, np.sum(passed)/passed.size

def across_the_iron_curtain(data_set, step):
    east = np.zeros((data_set.shape[0]), dtype=bool)
    west = np.zeros((data_set.shape[0]), dtype=bool)
    for s in range(step):
        east[data_set[:,s,0]>0] = True
    west[np.logical_and(east, data_set[:,step-1,0]<0)] = True
    return np.sum(west), west.size, np.sum(west)/west.size

def avg_moves_to_distance(data_set, distance):
    not_reached_mark = np.ndarray((data_set.shape[0]), dtype = np.bool)
    not_reached_mark[:] = True
    steps_required = np.zeros((data_set.shape[0]), dtype = np.int8)
    for s in range(data_set.shape[1]):
        made_it = euclidean(data_set[:,s,:])>=distance
        steps_required[np.logical_and(not_reached_mark, made_it)] = s
        not_reached_mark[made_it] = False
    steps_required = steps_required[steps_required!=0]
    samples = steps_required.size
    return steps_required, samples, np.mean(steps_required)

def gen_data(iterations, steps):
    data_set = np.ndarray((iterations, steps, 2), dtype=np.int8)
    optionsx = [1, 0, -1, 0]
    optionsy = [1, -1]
    for s in range(steps-1):
        stepsx = np.random.choice(optionsx, iterations)
        stepsy = np.random.choice(optionsy, iterations)
        stepsy[stepsx!=0]=0
        data_set[:, s+1, 0] = data_set[:, s, 0] + stepsx
        data_set[:, s+1, 1] = data_set[:, s, 1] + stepsy
    return data_set

def gen_async(thread_count, pool):
    to_return = list()
    for i in range(thread_count):
        to_return.append(pool.apply_async(gen_data,[MAX_SIZE,STEPS]))
    return to_return


if __name__ == '__main__':
    print("Generating data...")

    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    data_set = np.vstack([x.get() for x in gen_async(THREADS, pool)])
    print("DS Shape:")
    print(data_set.shape)

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

    print("Q7:")
    print(avg_moves_to_distance(data_set, 10))

    #print("Q8:")
    #print(avg_moves_to_distance(data_set, 60))
