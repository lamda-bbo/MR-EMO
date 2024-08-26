from random import random, randrange, seed
from multiprocessing import Pool
from models import *
from methods import *
import argparse


num_professions = 2  # This is a constant; changing it requires
                     # further code modifications
num_localities = 10
random_samples = 100
Real_evaluation_samples=10000


def _distribute_professions_caps_and_jobs(num_agents):
    assert num_agents % num_localities == 0
    # Agents are split half-half between professions
    prof1 = num_agents // 2
    prof2 = num_agents - prof1
    professions = [0] * prof1 + [1] * prof2
    locality_caps = [num_agents // num_localities] * num_localities
    # Job numbers add up to the cap per locality, but `prof1` jobs
    # for profession 1 and `prof2` jobs for profession 2 are
    # randomly distributed inside these bounds.
    prof1_jobs = prof1  # Remaining jobs to distribute
    prof2_jobs = prof2
    job_numbers = []
    for cap in locality_caps:
        p1, p2 = 0, 0
        for _ in range(cap):
            if random() < prof1_jobs / (prof1_jobs + prof2_jobs):
                p1 += 1
                prof1_jobs -= 1
                assert prof1_jobs >= 0
            else:
                p2 += 1
                prof2_jobs -= 1
                assert prof2_jobs >= 0
        job_numbers.append((p1, p2))
    return prof1, prof2, professions, locality_caps, job_numbers


class CorrectionFunctionWrapper():
    def __init__(self, P):
        self.P = P
    def func(self, x):
        return min(x, self.P)


def test_correction(num_agents):
    _, _, professions, locality_caps, job_numbers = \
        _distribute_professions_caps_and_jobs(num_agents)
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


def test_interview(num_agents):
    _, _, professions, locality_caps, job_numbers = \
        _distribute_professions_caps_and_jobs(num_agents)
    compatibility_probabilities = [random() for _ in range(num_agents)]
    model = InterviewModel(num_agents, locality_caps, num_professions,
                           professions, job_numbers,
                           compatibility_probabilities, random_samples,Real_evaluation_samples)
    return model


def test_coordination(num_agents):
    prof1, prof2, professions, locality_caps, job_numbers = \
        _distribute_professions_caps_and_jobs(num_agents)
    locality_num_jobs = locality_caps
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
    m = settings[args.set](args.p)
    #print(args.variant)

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
        moead(m, args.dir, args.p, args.set, args.index)#这里没有加true 修复不可行解的代码直接在代码里加了
    else :
        print("wrong algorithm name")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-alg',  type=str,default="nsga2")
    argparser.add_argument('-dir', help="folder", type=str)
    argparser.add_argument('-set', type=str, help="model name",default="interview")
    argparser.add_argument('-p', help="vary num of agents",type=int,default=100)
    argparser.add_argument('-index', help="instance",type=int,default="0")
    argparser.add_argument('-seed',help="seed_of_instance",type=float,default=0)
    argparser.add_argument('-variant', type=str, default="False")
    args = argparser.parse_args()

    main(args)