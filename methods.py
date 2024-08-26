from math import inf
from random import random, randrange, seed, choice, uniform
import os.path
import copy
import models
#import geatpy as ea
import numpy as np
from os import listdir
from os.path import isfile, join
import sys
import math
#from gurobipy import Model as GurobiModel, GRB, quicksum
def get_path(func, dirname, param, setting, seq):
    #path = os.path.dirname(os.getcwd()) + f'/Logs/{dirname}/{setting}/{func}/{param}/'
    path = f'Logs/{dirname}/{setting}/{func}/{param}/'
    if not os.path.exists(path):
        os.makedirs(path)
    log_path = path + (str)(seq) + '.txt'

    return log_path

def mutation(model, p, elem):
    res = copy.deepcopy(elem)
    for i in range(len(res)):
        for j in range(len(res[i])):
            if uniform(0,1) < p:
                res[i][j] = 1 - res[i][j]

    return res

def new_matutain(model,elem):
    elem_arr = np.array(elem.element)
    flag=True#默认列变换
    size=len(model.locality_caps)
    if uniform(0,1)<0.5:
        flag=False
        size=model.num_agents
    p=int(uniform(0,size))
    q=int(uniform(0,size))

    if flag==True:
        col=copy.deepcopy(elem_arr[:,p])
        elem_arr[:,p]=elem_arr[:,q]
        elem_arr[:, q]=col
    else:
        row = copy.deepcopy(elem_arr[p])
        elem_arr[p] = elem_arr[q]
        elem_arr[q] = row

    return get_elem(model,elem_arr.tolist())

# def new_matutain(model,elem):
#     elem_arr = np.array(elem.element)
#     flag=True#默认列变换
#     size=len(model.locality_caps)
#     if uniform(0,1)<0.5:
#         flag=False
#         size=model.num_agents
#     p=int(uniform(0,size))
#     q=int(uniform(0,size))
#
#     if flag==True:
#         col=copy.deepcopy(elem_arr[:,p])
#         elem_arr[:,p]=elem_arr[:,q]
#         elem_arr[:, q]=col
#     else:
#         row = copy.deepcopy(elem_arr[p])
#         elem_arr[p] = elem_arr[q]
#         elem_arr[q] = row
#
#     return get_elem(model,elem_arr.tolist())

def find_optimal_nn(model, archived_set):
    best_value = -1
    best_res = [None for _ in range(model.num_agents)]
    for e in archived_set:
        if e.f1_value > best_value and e.f2_value == 0:#这里要加上e.f2_value
            best_value, best_res = e.f1_value, e.locality_per_agent

    return best_res

def find_optimal_gsemo(model, archived_set):
    best_value = -1
    best_res = [None for _ in range(model.num_agents)]
    for e in archived_set:
        if e.f1_value > best_value :
            best_value, best_res = e.f1_value, e.locality_per_agent

    return best_value, best_res

class Elem(object):
    def __init__(self, f1_value, f2_value, element, locality_per_agent):
        super(Elem, self).__init__()
        self.f1_value = f1_value
        self.f2_value = f2_value
        self.element = element
        self.locality_per_agent = locality_per_agent

def get_elem(model, elem):

    f1 = f2 = 0
    caps = [0 for _ in range(len(model.locality_caps))]
    locality_per_agent = [None for _ in range(model.num_agents)]
    for i in range(len(elem)):
        f = 0
        for j in range(len(elem[i])):
            if elem[i][j] == 1:
                locality_per_agent[i] = j
                f += 1
                caps[j] += 1
            else:
                f2 += 1
        if f > 1:
            f1 = -1
    for i in range(len(caps)):
        if caps[i] > model.locality_caps[i]:
            f1 = -1
    if f1 != -1:
        f1 = model.utility_for_matching(locality_per_agent)

    return Elem(f1, f2, elem, locality_per_agent)

def evalVars(Vars,model):  # 定义目标函数（不含约束）
    f1max=model.num_agents
    f1min=-1
    f2max=model.num_agents*len(model.locality_caps)
    f2min=0
    epsilon=1e-6
    f1_denominator=f1max-f1min+epsilon
    f2_denominator=f2max-f2min+epsilon
    ObjV=np.array([0, 0])
    for solution in Vars:
        solution.resize((model.num_agents,len(model.locality_caps)))
        elem=get_elem(model,solution)
        f1=(elem.f1_value-f1min)/f1_denominator
        f2=(elem.f2_value-f2min)/f2_denominator
        f=np.array([f1,f2])
        ObjV = np.vstack((ObjV,f))#elem的f2是|x|_0
    ObjV=np.delete(ObjV, [0],axis = 0)
    return ObjV

def moead(model, dirname, param, setting, seq):
    P=100#种群规模
    name = 'moea/d'  # 初始化name（函数名称，可以随意设置）
    M = 2  # 目标数
    Dim = len(model.locality_caps)* model.num_agents # 初始化Dim（决策变量维数）
    maxormins = [-1] * M  # 初始化max or min（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
    varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
    lb = [0] * Dim  # 决策变量下界
    ub = [1] * Dim  # 决策变量上界
    lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
    ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）

    log_path = get_path('moead_variant', dirname, param, setting, seq)

    problem = ea.Problem(name=name, M=M, maxormins=maxormins,Dim=Dim,varTypes=varTypes,lb=lb,ub=ub,lbin=lbin,ubin=ubin,evalVars=evalVars)
    # 构建算法

    algorithm = ea.moea_MOEAD_templet(problem,ea.Population(Encoding='BG', NIND=P),  # 种群规模
                                      MAXEVALS=(int)(pow(model.num_agents, 2) * len(model.locality_caps))*100,  # 最大评估次数
                                      logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    # 求解
    p=ea.optimize(algorithm, verbose=False, drawing=0, outputMsg=False, drawLog=False, saveFlag=False,model=model,log_path=log_path)
    return p

def repairInfeasibleSolution(model,elem):
    a= np.array(elem.element)
    locality_per_agent = elem.locality_per_agent
    matroid_agent = a.sum(axis=1)  # 横轴求和
    for i in range(model.num_agents):
        if matroid_agent[i]>=2:
            row=a[i]
            index=[]
            for j in range(len(model.locality_caps)):
                if row[j]==1:
                    index.append(j)
            position=choice(index)
            locality_per_agent[i]=position
            for item in index:
                if position!=item:
                    a[i][item]=0
    matroid_locality=a.sum(axis=0) # 纵轴求和
    for i in range(len(model.locality_caps)):
        if matroid_locality[i]>model.locality_caps[i]:
            diff=matroid_locality[i]-model.locality_caps[i]
            col=a[:,i:i+1]
            index=[]
            for j in range(model.num_agents):
                if col[j]==1:
                    index.append(j)
            for it in range(diff):
                position=choice(index)
                a[position][i]=0
                index.remove(position)
                locality_per_agent[position]=None

    elem.element=a.tolist()
    elem.locality_per_agent=locality_per_agent
    elem.f1_value= model.utility_for_matching(locality_per_agent)
    elem.f2_value=a.shape[0]*a.shape[1]-a.sum()
    return elem

#algorithm- gsmeo
def gsemo_algorithm(model, dirname, param, setting, seq,alg,update_variant=False):
    if alg=="gsemo":
        name="gsemo"
        col_p=0
    elif alg=="gsemo_col" :
        name="gsemo_col"
        col_p=0.5
    elif alg=="gsemo_row":
        name="gsemo_row"
        col_p=0.5
    elif alg=="gsemo_row_col":
        name="gsemo_row_col"
        col_p=0.5

    if update_variant==True:
        name+="_variant"


    p = 1.0 / (model.num_agents * len(model.locality_caps))
    archived_set = [get_elem(model, [[0 for _ in range(len(model.locality_caps))] for _ in range(model.num_agents)])]
    T=100
    greedy_time = (int)(pow(model.num_agents, 2) * len(model.locality_caps))
    

    log_path = get_path(name, dirname, param, setting, seq)
    #print(log_path)
    # if isfile(log_path):
    #     return
    for it in range(T):
        for _ in range(greedy_time):
            parent=choice(archived_set)
            if uniform(0,1)<col_p:
                elem = new_matutain(model,parent)
            else:
                elem = get_elem(model, mutation(model, p, parent.element))

            if elem.f1_value==-1 and update_variant==True:
                elem=repairInfeasibleSolution(model,elem)

            f1, f2 = elem.f1_value, elem.f2_value


            flag = True
            for e in archived_set:
                if ((e.f1_value > f1) and (e.f2_value >= f2)) or ((e.f1_value >= f1) and (e.f2_value > f2)):
                    flag = False
                    break

            if flag == True:
                archived_set = [e for e in archived_set if not ((f1 >= e.f1_value) and (f2 >= e.f2_value))]
                archived_set.append(elem)

        # store temporary optimal value
        best_value, best_res = find_optimal_gsemo(model, archived_set)

        if it == T - 1:
            result = model.utility_for_matching(best_res,memoize=False, Real_evaluation=True)
        else:
            result = best_value
        with open(log_path, 'a') as f:
            f.write(f'{result}\n')

    return archived_set

#algorithm - Greedy algorithm
def greedy_algorithm(model, dirname, param, setting, seq):
    """The greedy algorithm for maximizing an (approximately) submodular utility function.

    Args:
        model (models.Model): The submodular model to use

    Returns:
        pair (locality_per_agent,best_value) of type (list of int/None, float).
        The first component is the matching, the second its queried value in the model.
    """
    locality_per_agent = [None for _ in range(model.num_agents)]
    caps_remaining = [cap for cap in model.locality_caps]

    for _ in range(min(model.num_agents, sum(caps_remaining))):
        best_pair = None
        best_value = -inf
        for i, match in enumerate(locality_per_agent):
            if match != None:
                continue

            for l, spaces in enumerate(caps_remaining):
                if spaces <= 0:
                    continue

                locality_per_agent[i] = l
                utility = model.utility_for_matching(locality_per_agent)
                locality_per_agent[i] = None

                if utility > best_value:
                    best_pair = (i, l)
                    best_value = utility

        assert best_pair != None
        i, l = best_pair
        locality_per_agent[i] = l
        caps_remaining[l] -= 1

    log_path = get_path('greedy', dirname, param, setting, seq)
    with open(log_path, 'a') as f:
        f.write((str)(model.utility_for_matching(locality_per_agent,memoize=False, Real_evaluation=True)))

# algorithm - additive optimization
def additive_optimization(model, dirname, param, setting, seq):
    """Optimize the model exactly, but just based on marginal utilities of
    individual migrant-locality pairs and assuming additivity.
    Args:
        model (models.Model): The submodular model to use
    Returns:
        pair (locality_per_agent,best_value) of type (list of int/None, float).
        The first component is the matching, the second its queried value in
        the model.
    """
    gm = GurobiModel()
    gm.setParam("OutputFlag", False)

    variables = []
    matching = [None for _ in range(model.num_agents)]
    objective = 0
    for i in range(model.num_agents):
        agent_vars = []
        for l in range(len(model.locality_caps)):
            matching[i] = l
            utility = model.utility_for_matching(matching)#上一步分配i到地点l 下一步重置 所以这里计算的就是单个人在单个地点 在没有竞争的条件下，能不能被雇佣，值就是0/1
            matching[i] = None

            v = gm.addVar(vtype=GRB.INTEGER, name=f"m_{i}_{l}")#add Var添加变量 类型：整数 名字：m_{i}_{l}
            gm.addConstr(v >= 0)#给变量v添加约束条件 非0即1 因为是整数类型
            gm.addConstr(v <= 1)
            agent_vars.append(v)

            objective += utility * v #根据

        variables.append(agent_vars)
        gm.addConstr(quicksum(agent_vars) <= 1)

    for l in range(len(model.locality_caps)):
        gm.addConstr(    quicksum(   variables[i][l] for i in range(model.num_agents)  )     <= model.locality_caps[l]              )

    gm.setObjective(objective, GRB.MAXIMIZE)
    gm.optimize()

    assert gm.status == GRB.OPTIMAL
    for i in range(model.num_agents):
        for l in range(len(model.locality_caps)):
            if variables[i][l].X > 0.5:
                matching[i] = l
                break

    path = get_path('additive', dirname, param, setting, seq)
    with open(path, 'a') as f:
        f.write(str(model.utility_for_matching(matching,memoize=False, Real_evaluation=True)))

# algorithm - enumeration method to find optimal value
def enumeration_method(model, dirname, param, setting, seq):
    best_value = -1
    best_res = [None for _ in range(model.num_agents)]

    T = 1 << (len(model.locality_caps) * model.num_agents)
    locality_per_agent = [None for _ in range(model.num_agents)]
    for it in range(T):
        selected_elem = [[0 for _ in range(len(model.locality_caps))] for _ in range(model.num_agents)]
        for i in range(len(selected_elem)): # agent
            for j in range(len(selected_elem[i])): # locality
                selected_elem[i][j] = (it >> (i * len(model.locality_caps) + j)) % 2

        elem = get_elem(model, selected_elem)
        f1, locality_per_agent = elem.f1_value, elem.locality_per_agent

        if f1 > best_value:
            best_value, best_res = f1, locality_per_agent

    log_path = get_path('optimal', dirname, param, setting, seq)
    with open(log_path, 'a') as f:
        f.write((str)(model.utility_for_matching(best_res,Real_evaluation=True)))

# algorithm - nsga2
def nsga2_method(model, dirname, param, setting, seq,update_variant=False):
    mutation_rate = 1.0 / (model.num_agents * len(model.locality_caps))
    crossover_rate = 0.9
    mutation_operate=1.0
    num_objectives=2

    if update_variant==True:
        name="nsga2_variant"
    else:
        name="nsga2"

    #随机产生100初始解
    P = []
    #population_num=100
    population_num = 2*(sum(model.locality_caps)+1)

    P.append(get_elem(model, [[0 for _ in range(len(model.locality_caps))] for _ in range(model.num_agents)]))
    for i in range(population_num-1):
        elem = get_elem(model, mutation(model, mutation_rate, choice(P).element))

        if elem.f1_value == -1 and update_variant == True:
            elem = repairInfeasibleSolution(model, elem)
        P.append(elem)

    query_times=population_num

    #设置T倍的贪心算法轮数
    T = 100
    greedy_time = (int)(pow(model.num_agents, 2) * len(model.locality_caps))

    log_path = get_path(name, dirname, param, setting, seq)

    Q = []
    it=1
    while query_times<T*greedy_time:
        R = []
        R.extend(P)
        R.extend(Q)

        fronts = fast_nondominated_sort(R)

        del P[:]

        for front in fronts.values():
            if len(front) == 0:
                break

            crowding_distance_assignment(front,num_objectives);
            P.extend(front)

            if len(P) >= population_num:
                break

        sort_crowding(P)

        if len(P) > population_num:
            del P[population_num:]

        Q = make_new_pop(model,P,mutation_rate,crossover_rate,mutation_operate,update_variant)
        query_times+=population_num

        if query_times/greedy_time>=it:
            it+=1
            # store temporary optimal value
            best_value, best_res = find_optimal_gsemo(model, P)

            if query_times >= T*greedy_time:
                result = model.utility_for_matching(best_res, memoize=False, Real_evaluation=True)

            else:
                result = best_value
               
            with open(log_path, 'a') as f:
                f.write(f'{result}\n')

    return P

def sort_objective(P, obj_idx):
        for i in range(len(P) - 1, -1, -1):
            for j in range(1, i + 1):
                s1 = P[j - 1]
                s2 = P[j]
                if obj_idx==0:
                    if s1.f1_value > s2.f1_value:
                        P[j - 1] = s2
                        P[j] = s1
                else:
                    if s1.f2_value > s2.f2_value:
                        P[j - 1] = s2
                        P[j] = s1

def sort_crowding(P):
        for i in range(len(P) - 1, -1, -1):
            for j in range(1, i + 1):
                s1 = P[j - 1]
                s2 = P[j]

                if crowded_comparison(s1, s2) < 0:
                    P[j - 1] = s2
                    P[j] = s1

def make_new_pop(model,P,mutation_rate,crossover_rate,mutation_operate,update_variant=False):
        '''
        Make new population Q, offspring of P.
        '''
        Q = []

        while len(Q) != len(P):
            selected_solutions = [None, None]

            while selected_solutions[0] == selected_solutions[1]:
                for i in range(2):
                    s1 =choice(P)
                    s2 = s1
                    while s1 == s2:
                        s2 = choice(P)

                    if crowded_comparison(s1, s2) > 0:
                        selected_solutions[i] = s1

                    else:
                        selected_solutions[i] = s2

            if random() < crossover_rate:
                child_solution_element = crossover(model,selected_solutions[0],selected_solutions[1],mutation_operate)

                if random() < mutation_operate:
                    child_solution = get_elem(model, mutation(model, mutation_rate, child_solution_element))

                else:
                    child_solution=get_elem(model,child_solution_element)


                if child_solution.f1_value == -1 and update_variant == True:
                    child_solution = repairInfeasibleSolution(model, child_solution)

                Q.append(child_solution)
        return Q

def is_in_fronts(p,front):
    for i in range(len(front)):
        if front[i].element==p.element:
            return True
    return False

def fast_nondominated_sort(P):
        '''
        Discover Pareto fronts in P, based on non-domination criterion.
        '''
        fronts = {}

        S = {}
        n = {}
        for s in P:
            S[s] = []
            n[s] = 0

        fronts[1] = []

        for p in P:
            for q in P:
                if p == q:
                    continue
                if ( (p.f1_value > q.f1_value) and (p.f2_value >= q.f2_value)) or ((p.f1_value >= q.f1_value) and (p.f2_value > q.f2_value)):
                    S[p].append(q)

                elif( (q.f1_value > p.f1_value) and (q.f2_value >= p.f2_value)) or ((q.f1_value >= p.f1_value) and (q.f2_value > p.f2_value)):
                    n[p] += 1

            #if n[p] == 0 and not is_in_fronts(p,fronts[1]):#需要判断是否已经加入 修补之前代码的BUG
            #注意！！！后来想这里应该是正确的，相同的解也可以包含进来，代码也不会被卡住了
            if n[p] == 0 :
                p.rank=1
                fronts[1].append(p)

        i = 1

        while len(fronts[i]) != 0:
            next_front = []

            for r in fronts[i]:
                for s in S[r]:
                    n[s] -= 1
                    if n[s] == 0:
                        s.rank=i+1
                        next_front.append(s)

            i += 1
            fronts[i] = next_front

        return fronts

def crowding_distance_assignment(front,num_objectives):
        '''
        Assign a crowding distance for each solution in the front.
        '''
        for p in front:
            p.distance = 0

        for obj_index in range(num_objectives):
            sort_objective(front, obj_index)

            front[0].distance = float('inf')
            front[len(front) - 1].distance = float('inf')

            for i in range(1, len(front) - 1):
                front[i].distance += (front[i + 1].distance - front[i - 1].distance)

def crowded_comparison(s1, s2):
    '''
    Compare the two solutions based on crowded comparison.
    '''
    if s1.rank < s2.rank:
        return 1

    elif s1.rank > s2.rank:
        return -1

    elif s1.distance > s2.distance:
        return 1

    elif s1.distance < s2.distance:
        return -1

    else:
        return 0

def crossover(model,elem1,elem2,mutation_operate):
    #binary presentation: uniform crossover
    # child_elem = []
    # for i in range(model.num_agents):
    #     row=[]
    #     for j in range(len(model.locality_caps)):
    #         if random()<0.5:
    #             row.append(elem1.element[i][j])
    #         else:
    #             row.append(elem2.element[i][j])
    #     child_elem.append(row)
    # return get_elem(model, child_elem)

    #one-point crossover

    point=randrange(1, model.num_agents-1)
    #print(point)
    child_elem1 = []
    #child_elem2 = []
    for i in range(model.num_agents):
        if i<point:
            row1 = elem1.element[i]
            #row2 = elem2.element[i]
        else:
            row1 = elem2.element[i]
            #row2 = elem1.element[i]
        child_elem1.append(row1)
        #child_elem2.append(row2)
    #return get_elem(model, child_elem1)
    return child_elem1

