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
def func(name,seed, dirname, num_localities, setting, i,variant):
    cmd=f'python num_localities.py -alg {name}  -p {num_localities} -set {setting} -index {i} -dir {dirname} -seed {seed} -variant {variant}'
    print(cmd)
    os.system(cmd)
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-l_min', help="min num of localities", type=int, default=10)
    argparser.add_argument('-l_max', help="max num of localities", type=int, default=30)
    argparser.add_argument('-variant', type=str, default="False")
    argparser.add_argument('-instances', type=int, default=10)
    argparser.add_argument('-alg', type=str, default="gsemo_row")
    args = argparser.parse_args()

    seeds=generate_seed(args.instances)

    algorithms = []
    algorithms.append(args.alg)
    settings=[ "interview","coordination"]

    pool = multiprocessing.Pool(processes=80)

    for num_localities in range(args.l_min, args.l_max+1,2):
        for i in range(args.instances):
            for setting in settings:
                dirname = 'num_localities'
                for name in algorithms:
                    pool.apply_async(func, (name,seeds[i],dirname, num_localities, setting, i,args.variant,))

    pool.close()
    pool.join()