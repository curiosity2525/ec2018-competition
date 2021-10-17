# -*- coding: utf-8 -*-
import array
import random
import json
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3d

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools


# 第1目的関数：最小化，第2目的関数：最大化
creator.create("Multi_Objective", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.Multi_Objective)

toolbox = base.Toolbox()

# 遺伝子が取り得る値の下限値，上限値
BOUND_LOW, BOUND_UP = 0.0, 1.0
# 遺伝子長（設計変数の次元数）
NDIM = 2
# 制約の数
CONS_NUM = 2

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
# 個体の設定
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
# 集団の設定
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 評価関数の定義
def moon_function(individual):
    # 1：設計変数値の書き込み
    filevar = open("pop_vars_eval.txt", "w")
    for ndim_i in range(NDIM):
        filevar.write(str(individual[ndim_i]))
        filevar.write('\t')
    filevar.write('\n')
    filevar.close() 

    # 2：run.shの実行
    subprocess.call("./run.sh")

    # 3：評価値（pop_objs_eval.txt）の読み込み
    fileobj = open("pop_objs_eval.txt", "r")
    objs_ind_txt_data = fileobj.read()
    objs_ind_txt_sp_data = []
    #print(objs_ind_txt_sp_data)
    objs_ind_txt_sp_data += objs_ind_txt_data.split("\t")
    
    # 評価モジュールの評価値をfitnessに格納
    fit1 = float(objs_ind_txt_sp_data[0])
    fit2 = float(objs_ind_txt_sp_data[1])
    fit3 = float(objs_ind_txt_sp_data[2])
    #print("fit1="+str(fit1))
    fileobj.close()

    # 4：制約違反情報（pop_cons_eval.txt）の読み込み
    filecon = open("pop_cons_eval.txt", "r")
    cons_ind_txt_data = filecon.read()
    cons_ind_txt_sp_data = []
    cons_ind_txt_sp_data += cons_ind_txt_data.split("\t")

    cons_ind_float_data = np.zeros(len(cons_ind_txt_sp_data))
    for cons_i in range(CONS_NUM):
        cons_ind_float_data[cons_i] = float(cons_ind_txt_sp_data[cons_i])

    vio_num = 0
    for cons_i in range(CONS_NUM):
        if cons_ind_float_data[cons_i] < 0:
            vio_num += 1
    filecon.close()

    # 評価関数値の設定
    f1 = fit1 + vio_num
    f2 = fit2 + vio_num
    f3 = fit3 + vio_num

    return f1, f2, f3

# 評価関数の設定
toolbox.register("evaluate", moon_function)
# 交叉オペレータの設定
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
# 突然変異オペレータの設定
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
# 選択オペレータの設定
toolbox.register("select", tools.selNSGA2)


# 最適化（アルゴリズム：NSGA2）
def optimization(seed=None):
    # 初期設定
    ## 乱数の種
    random.seed(seed)
    ## 最大世代数
    NGEN = 100
    ## 個体数
    MU = 300
    ## 交叉率
    CXPB = 0.5

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "max"

    # 初期集団の生成
    pop = toolbox.population(n=MU)
    ## 初期集団の記録
    pop_ini = pop[:]

    # 初期集団の評価
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    # 初期集団の生存選択
    pop = toolbox.select(pop, len(pop))

    # ログ
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # 終了条件：世代数がNGENになるまで
    for gen in range(1, NGEN):

        # 複製選択
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # 次世代集団の生成
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            ## 交叉
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            ## 突然変異
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # 評価
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 生存選択
        pop = toolbox.select(pop + offspring, MU)

        # ログ
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    print("Final population hypervolume is %f" % hypervolume(pop, [1.0, 0.0, 1.0]))

    return pop, pop_ini, logbook

# main関数
if __name__ == "__main__":
    # 最適化の実行 ； 戻り値：最終世代の集団，初期集団，状態
    pop, pop_ini, stats = optimization(64)

    # パレート最適解出力
    ## 初期集団の適応度
    fitnesses_ini = np.array([list(pop_ini[i].fitness.values) for i in range(len(pop_ini))])
    # 最終世代の適応度
    fitnesses = np.array([list(pop[i].fitness.values) for i in range(len(pop))])
    ## プロット関連
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(fitnesses[:,0], fitnesses[:,1], fitnesses[:,2], "r.", label="Optimized")
    ax.scatter(fitnesses_ini[:,0], fitnesses_ini[:,1], fitnesses_ini[:,2],"b.", label="Initial")
    ref_points = tools.uniform_reference_points(3, 2)
    ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], marker="o", s=24, label="Reference Points")
    ax.view_init(elev=11, azim=-25)
    ax.autoscale(tight=True)
    plt.legend()
    plt.tight_layout()
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.ylabel("f3")
    title = "fitnesses(Gen=100)"
    plt.savefig(title +".png", dpi=300)