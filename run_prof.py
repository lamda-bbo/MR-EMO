from random import random, randrange, seed
import multiprocessing
import os
import argparse
from methods import *
def generate_seed(num_instance):
    seed(0)
    seeds=[random() for _ in range(num_instance)]
    #print(seeds)
    return seeds
def func(name,seed, dirname, num_professions, setting, i):
    cmd=f'python num_professions.py -alg {name}  -p {num_professions} -set {setting} -index {i} -dir {dirname} -seed {seed}'
    print(cmd)
    os.system(cmd)
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-instances', type=int, default=10)
    args = argparser.parse_args()

    seeds=generate_seed(args.instances)

    algorithms = ["greedy", "add"]
    settings=["correction", "interview","coordination"]

    pool = multiprocessing.Pool(processes=10)

    for num_professions in [2, 3, 5, 8, 10, 15, 20, 25, 30]:
        for i in range(args.instances):
            for setting in settings:
                dirname = 'num_professions'
                for name in algorithms:
                    pool.apply_async(func, (name,seeds[i],dirname, num_professions, setting, i,))

    pool.close()
    pool.join()

