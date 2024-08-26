from random import random, randrange, seed
from multiprocessing import Pool
from models import *
from methods import *
import argparse


num_professions = 2  # This is a constant; changing it requires
                     # further code modifications
num_agents = 100
prof1 = 50
prof2 = num_agents - prof1
professions = [0] * prof1 + [1] * prof2
num_localities = 10  # Fix localities to 10 localities with cap 10 each
locality_caps = [10] * 10
random_samples = 1000
Real_evaluation_samples=10000

def _distribute_jobs(prof1_jobs, prof2_jobs):
    job_numbers = [(0, 0)] * num_localities
    for _ in range(prof1_jobs):
        l = randrange(num_localities)
        prof1, prof2 = job_numbers[l]
        job_numbers[l] = (prof1 + 1, prof2)
    for _ in range(prof2_jobs):
        l = randrange(num_localities)
        prof1, prof2 = job_numbers[l]
        job_numbers[l] = (prof1, prof2 + 1)
    return job_numbers


class CorrectionFunctionWrapper():
    def __init__(self, P):
        self.P = P
    def func(self, x):
        return min(x, self.P)


def test_correction(prof1_jobs, prof2_jobs):
    job_numbers = _distribute_jobs(prof1_jobs, prof2_jobs)
    qualification_probabilities = \
        [[random()] * num_localities for _ in range(num_agents)]
    correction_functions = []
    for p1, p2 in job_numbers:
        correction_functions.append((CorrectionFunctionWrapper(p1), CorrectionFunctionWrapper(p2)))
    model = RetroactiveCorrectionModel(num_agents, locality_caps,
                                       num_professions, professions,
                                       qualification_probabilities,
                                       correction_functions,
                                       random_samples,Real_evaluation_samples)
    return model


def test_interview(prof1_jobs, prof2_jobs):
    job_numbers = _distribute_jobs(prof1_jobs, prof2_jobs)
    compatibility_probabilities = [random() for _ in range(num_agents)]
    model = InterviewModel(num_agents, locality_caps, num_professions,
                           professions, job_numbers,
                           compatibility_probabilities, random_samples,Real_evaluation_samples)
    return model


def test_coordination(prof1_jobs, prof2_jobs):
    global prof1
    global prof2
    job_numbers = _distribute_jobs(prof1_jobs, prof2_jobs)
    locality_num_jobs = [prof1 + prof2 for prof1, prof2 in job_numbers]
    compatibility_probabilities = []
    for _ in range(prof1):
        competency = random()
        compatibility_probabilities.append(
            [[competency] * p1 + [0.] * p2 for p1, p2 in job_numbers])
    for _ in range(prof2):
        competency = random()
        compatibility_probabilities.append(
            [[0.] * p1 + [competency] * p2 for p1, p2 in job_numbers])
    model = CoordinationModel(num_agents, locality_caps,
                              locality_num_jobs,
                              compatibility_probabilities,
                              random_samples,Real_evaluation_samples)
    return model


def main(args):
    settings = {"correction": test_correction, "interview": test_interview,
                "coordination": test_coordination}
    #重点
    seed(args.seed)
    m = settings[args.set](args.job_pro1,args.job_pro2)
    print(args.variant)
    if (args.variant == "True"):
        flag = True
    else:
        flag = False
    if args.alg == "add":
        additive_optimization(m, args.dir, args.p, args.set, args.index)
    elif args.alg == "greedy":
        greedy_algorithm(m, args.dir, args.p, args.set, args.index)
    elif args.alg == "nsga2":
        nsga2_method(m, args.dir, args.p, args.set, args.index,False)
    elif args.alg == "gsemo" or args.alg == "gsemo_col" or args.alg == "gsemo_row" or args.alg == "gsemo_row_col":
        gsemo_algorithm(m, args.dir, args.p, args.set, args.index, args.alg, flag)
    elif args.alg == "moead":
        moead(m, args.dir, args.p, args.set, args.index)
    else :
        print("wrong algorithm name")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-alg',  type=str,default="moead")
    argparser.add_argument('-dir', help="folder", type=str)
    argparser.add_argument('-set', type=str, help="model name",default="correction")
    argparser.add_argument('-p', help="folder",type=int,default="1")
    argparser.add_argument('-index', help="instance",type=int,default="0")
    argparser.add_argument('-seed',help="seed_of_instance",type=float,default=0)
    argparser.add_argument('-job_pro1', help="prof1_jobs", type=int, default="1")
    argparser.add_argument('-job_pro2', help="prof2_jobs", type=int, default="1")
    argparser.add_argument('-variant', type=str, default="False")
    args = argparser.parse_args()

    main(args)