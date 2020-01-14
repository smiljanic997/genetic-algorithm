import numpy as np
import random
import collections
import operator
import itertools
from math import exp
import time
import warnings

N_ITER = 1500
populationSize = 36
precision = 3
Gd_x = Gd_y = -3
Gg_x = Gg_y = 3
p_recomb = 0.25     # probability of recombination
p_mut = 0.05        # probability of mutation

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def print(self):
        print("(", self.x,", ", self.y, ")")
    def get_binary_coded(self, n):
        return bin(self.x)[2:].zfill(n), bin(self.y)[2:].zfill(n)

def fun(x,y):
    # this is the two-variable function for which minimum and maximum will be found
    return 3 * ((1-x)**2)*np.exp(-x**2-(y+1)**2) - 10*(x/5-x**3-y**5)*np.exp(-x**2-y**2)- 1/3*np.exp(-(x+1)**2-y**2)

def get_bd(x, Gd, Gg, n):
    return int(np.floor(((x - Gd) / (Gg - Gd)) * (2**n - 1)))  # returns bd

def get_x_from_bd(bd, Gd, Gg, n):
    return Gd + (Gg - Gd) / (2**n - 1) * bd    # returns x

def get_number_of_bits(Gd, Gg, p):
    return int(np.ceil(np.log2((Gg - Gd) * 10**p + 1)))  # returns n

def generate_population(size):
    list = []
    for i in range(size):
        x = random.uniform(-3.0, 3.0)
        y = random.uniform(-3.0, 3.0)
        list.append(Point(x, y))
    return list
        
def get_bd_point(point, Gd, Gg, p):
    n = get_number_of_bits(Gd, Gg, p)
    bdx = get_bd(point.x, Gd, Gg, n)
    bdy = get_bd(point.y, Gd, Gg, n)
    return Point(bdx,bdy)

def get_point_from_bd(bd, Gd, Gg, n):
    x = get_x_from_bd(bd.x, Gd, Gg, n)
    y = get_x_from_bd(bd.y, Gd, Gg, n)
    return Point(x, y)

def get_min_point(population):
    min_point = population[0]
    for p in population:
        if fun(p.x, p.y) < fun(min_point.x, min_point.y):
            min_point = p
    return min_point

def get_max_point(population):
    max_point = population[0]
    for p in population:
        if fun(p.x, p.y) > fun(max_point.x, max_point.y):
            max_point = p
    return max_point


def ff_min(point, max_point):
    # fitness function for minimum
    return fun(max_point.x, max_point.y) - fun(point.x, point.y)


def ff_max(point, min_point):
    # fitness function for maximum
    return fun(point.x, point.y) - fun(min_point.x, min_point.y)

def tuples_chromosome_grade_max(population):
    # generates pairs of chromosome and its grade for whole population (max)
    t_list = []
    maxPoint = get_max_point(population)
    minPoint = get_min_point(population)
    for chrom in population:
        if ff_max(chrom, maxPoint) == 0:
            t_list.append((chrom, ff_max(chrom, minPoint)))
    for chrom in population:
        if ff_max(chrom, maxPoint) != 0:
            t_list.append((chrom, ff_max(chrom, minPoint)))
    return t_list

def tuples_chromosome_grade_min(population):
    # generates pairs of chromosome and its grade for whole population (min)
    t_list = []
    maxPoint = get_max_point(population)
    minPoint = get_min_point(population)
    for chrom in population:
        if ff_min(chrom, maxPoint) == 0:
            t_list.append((chrom, ff_min(chrom, maxPoint)))
    for chrom in population:
        if ff_min(chrom, maxPoint) != 0:
            t_list.append((chrom, ff_max(chrom, maxPoint)))
    return t_list

# TODO > combine these two functions above into one

def get_probs(grades):
    total = sum(grades)
    probs = []
    for g in grades:
        probs.append(g / total)
    return probs

def get_cumulative_grades(probabilities):
    cumulative_grades = []
    current = probabilities[0]
    cumulative_grades.append(current)
    for i in range(1, len(probabilities)):
        current = current + probabilities[i]
        cumulative_grades.append(current)

    return cumulative_grades

def roulette_selection(population, look_for):
    # use roulette selection to choose which chromosomes will recombinate
    if look_for == 'max':
        pairs = tuples_chromosome_grade_max(population)
    elif look_for == 'min':
        pairs = tuples_chromosome_grade_min(population)
    else:
        raise ValueError('Wrong parameter.')
    grades = [x[1] for x in pairs]
    probs = get_probs(grades)
    cumulative_grades = get_cumulative_grades(probs)
    
    selected = []
    for j in range(len(population)):
        r = random.uniform(0,1)
        i = 0
        while r > cumulative_grades[i]:
            i = i + 1
        selected.append(pairs[i])
    ret_sel = []
    for t in selected:
        ret_sel.append(t[0])
    return ret_sel

def shuffle(population):
    return random.shuffle(population)

def recombinate_pair(chrom1, chrom2):
    # recombinates two chromosomes in one point
    # since chromosome contains two points (x, y), x1, x2 and y1, y2 are recombinated in different points
    n = get_number_of_bits(Gd_x, Gg_x, precision)
    p1 = get_bd_point(chrom1, Gd_x, Gg_x, precision)
    p2 = get_bd_point(chrom2, Gd_x, Gg_x, precision)
    x1, y1 = p1.get_binary_coded(n)
    x2, y2 = p2.get_binary_coded(n)
    crossover_point = random.randint(0, n - 1)
    new_x1_bd = x1[:len(x1) - crossover_point - 1] + x2[len(x2) - crossover_point - 1:]
    new_x2_bd = x2[:len(x2) - crossover_point - 1] + x1[len(x1) - crossover_point - 1:]
    new_y1_bd = y1[:len(y1) - crossover_point - 1] + y2[len(y2) - crossover_point - 1:]
    new_y2_bd = y2[:len(y2) - crossover_point - 1] + y1[len(y1) - crossover_point - 1:]
    bd_point_1 = Point(int(new_x1_bd, 2), int(new_y1_bd, 2))
    bd_point_2 = Point(int(new_x2_bd, 2), int(new_y2_bd, 2))
    point_1 = get_point_from_bd(bd_point_1, Gd_x, Gg_x, n)
    point_2 = get_point_from_bd(bd_point_2, Gd_x, Gg_x, n)
    return point_1, point_2
    

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
    

def crossover(population):
    shuffle(population)
    recombinated_population = []
    pairs = list(chunks(population, 2))
    for pair in pairs:
        r = random.uniform(0, 1)
        if r < p_recomb:
            recombinated_population.append(recombinate_pair(pair[0], pair[1]))
        else:
            recombinated_population.append(pair)
    return list(itertools.chain(*recombinated_population))


def chromosome_mutation(chrom, n, mutation_point_x, mutation_point_y):
    # mutate one chromosome in one point
    bd_point = get_bd_point(chrom, Gd_x, Gg_x, precision)
    x_c, y_c = bd_point.get_binary_coded(n)
    x_coord = list(x_c)
    y_coord = list(y_c)
    if x_coord[- mutation_point_x - 1] == '0':
        x_coord[- mutation_point_x - 1] = '1'
    else:
        x_coord[- mutation_point_x - 1] = '0'
        
    if y_coord[- mutation_point_y - 1] == '0':
        y_coord[- mutation_point_y - 1] = '1'
    else:
        y_coord[- mutation_point_y - 1] = '0'
    
    x = int(''.join(x_coord), 2)
    y = int(''.join(y_coord), 2)
    return get_point_from_bd(Point(x, y), Gd_x, Gg_x, n)


def mutate(population):
    n = get_number_of_bits(Gd_x, Gg_x, precision)
    mutation_point_x = random.randint(0, n - 1)
    mutation_point_y = random.randint(0, n - 1)
    mutated_population = []
    for chrom in population:
        r = random.uniform(0, 1)
        if r < p_mut:
            mutated_population.append(chromosome_mutation(chrom, n, mutation_point_x, mutation_point_y))
        else:
            mutated_population.append(chrom)
    return mutated_population

def find_minimum():
    test_pop = generate_population(populationSize)

    for i in range(N_ITER):
        r_sel = roulette_selection(test_pop, look_for = 'min')
        crossed = crossover(r_sel)
        test_pop = mutate(crossed)

    min_point = test_pop[0]
    for t in test_pop:
        if fun(t.x, t.y) < fun(min_point.x, min_point.y):
            min_point = t
    return min_point


def find_maximum():
    test_pop = generate_population(populationSize)

    for i in range(N_ITER):
        r_sel = roulette_selection(test_pop, look_for = 'max')
        crossed = crossover(r_sel)
        test_pop = mutate(crossed)

    max_point = test_pop[0]
    for t in test_pop:
        if fun(t.x, t.y) > fun(max_point.x, max_point.y):
            max_point = t
    return max_point


def main():
    warnings.filterwarnings("ignore")
    print('Calculating...')
    # tms = time.time()

    print('Looking for minimum...')
    minimum_point = find_minimum()
    print('(', minimum_point.x, ',', minimum_point.y, ') ->', fun(minimum_point.x, minimum_point.y))
    print('Looking for maximum...')
    maximum_point = find_maximum()
    print('(', maximum_point.x, ',', maximum_point.y, ') ->', fun(maximum_point.x, maximum_point.y))

    # tme = time.time()
    
    # print(round(tme - tms, 2))

if __name__ == "__main__":
    main()
