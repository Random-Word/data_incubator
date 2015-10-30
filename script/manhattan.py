#!/usr/bin/env python3

import numpy as np
import random
import multiprocessing
import itertools
import functools
import time

MAX_SIZE = 100000
STEPS = 60
THREADS = 15

def euclidean(data):
    return np.sqrt((data[:, 0].astype(np.float64,copy=False)**2)+
            (data[:, 1].astype(np.float64,copy=False)**2))

def at_least_at(data_set, step, distance):
    made_it = np.zeros((data_set.shape[0]), dtype=bool)
    made_it = euclidean(data_set[:, step-1, :]) >= distance
    return made_it

def at_least_by(data_set, step, distance):
    passed = np.zeros((data_set.shape[0]), dtype=bool)
    for s in range(step):
        passed[euclidean(data_set[:, s, :]) >= distance] = 1
    return passed

def across_the_iron_curtain(data_set, step):
    east = np.zeros((data_set.shape[0]), dtype=bool)
    west = np.zeros((data_set.shape[0]), dtype=bool)
    for s in range(step):
        east[data_set[:,s,0]>0] = True
    west[np.logical_and(east, data_set[:,step-1,0]<0)] = True
    return west

def avg_moves_to_distance(data_set, distance):
    not_reached_mark = np.ndarray((data_set.shape[0]), dtype = np.bool)
    not_reached_mark[:] = True
    steps_required = np.zeros((data_set.shape[0]), dtype = np.uint8)
    for s in range(data_set.shape[1]):
        made_it = euclidean(data_set[:,s,:])>=distance
        steps_required[np.logical_and(not_reached_mark, made_it)] = s
        not_reached_mark[made_it] = False
    steps_required = steps_required[steps_required!=0] #Biased estimator
    samples = steps_required.size
    return steps_required

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
        to_return.append(pool.apply_async(gen_results))
    return to_return

def gen_results():
    data_set = gen_data(MAX_SIZE,STEPS)
    results = np.zeros((6,data_set.shape[0]), dtype='bool')
    results[0,:] = at_least_at(data_set, 10, 3)
    results[1,:] = at_least_at(data_set, 60, 10)
    results[2,:] = at_least_by(data_set, 10, 5)
    results[3,:] = at_least_by(data_set, 60, 10)
    results[4,:] = across_the_iron_curtain(data_set, 10)
    results[5.:] = across_the_iron_curtain(data_set, 30)
    return results

if __name__ == '__main__':
    print("Generating data...")

    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    results = np.zeros((6,THREADS,MAX_SIZE),dtype=bool)
    for i, x in enumerate(gen_async(THREADS, pool)):
        results[:,i,:]=x.get()
    print("TESTING:")
    for i in range(THREADS):
        print(np.sum(results[0,i,:]))
        print(np.sum(results[0,i,:])/results.shape[2])
    results = np.reshape(results, (6,THREADS*MAX_SIZE))

    print("Results Shape:")
    print(results.shape)

    for i in range(results.shape[0]):
        print("Q%d:"%(i))
        s = np.sum(results[i,:])
        print(s)
        print(repr(s/results.shape[1]))
