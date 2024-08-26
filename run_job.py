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
def func(name,seed, dirname, param, prof1_jobs,prof2_jobs,setting, i):
    cmd=f'python job_availability.py -alg {name} -job_pro1 {prof1_jobs} -set {setting} -index {i} -dir {dirname} -seed {seed} -job_pro2 {prof2_jobs} -p {param}'
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

    for prof1_jobs in [10, 20, 30, 40,50, 60, 70, 80, 90]:
        for prof2_jobs in [50]:
            for i in range(args.instances):
                for setting in settings:
                    dirname = 'job_availability'
                    param = prof2_jobs * 1000 + prof1_jobs
                    for name in algorithms:
                        pool.apply_async(func, (name,seeds[i],dirname,param,prof1_jobs,prof2_jobs, setting, i,))

    pool.close()
    pool.join()


