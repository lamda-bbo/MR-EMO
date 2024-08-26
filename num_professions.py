from random import random, randrange, seed
from multiprocessing import Pool
from models import *
from methods import *
import argparse


num_agents = 100
num_localities = 10
random_samples = 1000
Real_evaluation_samples=10000
locality_caps = [10] * num_localities


def _distribute_professions_caps_and_jobs(num_professions):
    assert num_professions <= num_agents
    professions = list(range(num_professions))
    profession_counts = [1 for _ in range(num_professions)]
    for _ in range(num_agents - num_professions):
        prof = randrange(num_professions)
        professions.append(prof)
        profession_counts[prof] += 1
    job_numbers = []
    profession_remaining = profession_counts[:]
    jobs_remaining = num_agents
    for cap in locality_caps:
        ps = [0 for _ in range(num_professions)]
        for _ in range(cap):
            a = random()
            for prof in range(num_professions):
                if a < (profession_remaining[prof] / jobs_remaining):
                    ps[prof] += 1
                    profession_remaining[prof] -= 1
                    break
                a -= profession_remaining[prof] / jobs_remaining
            jobs_remaining -= 1
        job_numbers.append(tuple(ps))
    assert sum(profession_remaining) == 0
    return profession_counts, professions, job_numbers


class CorrectionFunctionWrapper():
    def __init__(self, P):
        self.P = P
    def func(self, x):
        return min(x, self.P)


def test_correction(num_professions):
    _, professions, job_numbers = \
        _distribute_professions_caps_and_jobs(num_professions)
    qualification_probabilities = \
        [[random()] * num_localities for _ in range(num_agents)]
    correction_functions = []
    for ps in job_numbers:
        correction_functions.append([CorrectionFunctionWrapper(p) for p in ps])
    model = RetroactiveCorrectionModel(num_agents, locality_caps,
                                       num_professions, professions,
                                       qualification_probabilities,
                                       correction_functions,
                                       random_samples,Real_evaluation_samples)
    return model


def test_interview(num_professions):
    _, professions, job_numbers = \
        _distribute_professions_caps_and_jobs(num_professions)
    compatibility_probabilities = [random() for _ in range(num_agents)]
    model = InterviewModel(num_agents, locality_caps, num_professions,
                           professions, job_numbers,
                           compatibility_probabilities, random_samples,Real_evaluation_samples)
    return model


def test_coordination(num_professions):
    profession_counts, professions, job_numbers = \
        _distribute_professions_caps_and_jobs(num_professions)
    locality_num_jobs = locality_caps
    compatibility_probabilities = [] 
    for i, prof in enumerate(professions):
        competency = random()
        probs = []
        for ps in job_numbers:
            a = []
            for prof2, prof2nums in enumerate(ps):
                if prof2 == prof:
                    a += [competency] * prof2nums
                else:
                    a += [0] * prof2nums
            probs.append(a)
        compatibility_probabilities.append(probs)
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
    argparser.add_argument('-alg',  type=str,default="gsemo")
    argparser.add_argument('-dir', help="folder", type=str)
    argparser.add_argument('-set', type=str, help="model name",default="coordination")
    argparser.add_argument('-p', help="vary num of localities",type=int,default="10")
    argparser.add_argument('-index', help="instance",type=int,default="0")
    argparser.add_argument('-seed',help="seed_of_instance",type=float,default=0)
    argparser.add_argument('-variant', type=str, default="False")
    args = argparser.parse_args()

    main(args)