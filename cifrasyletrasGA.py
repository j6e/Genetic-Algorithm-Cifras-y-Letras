# "Cifras y Letras" Genetic Algorithm
# Joan García Esquerdo - 2016
# Code developed with the aid of my Genetic Algorithms class notes,
# Wikipedia, Colin's Drake repo on GA and StackOverflow

import random
import math
import itertools
import time
import argparse
from tqdm import tqdm


# Gobal Variables
# Define the objective and the algorithm parameters

parser = argparse.ArgumentParser(description="Cifras y Letras GA.")
parser.add_argument("NUMBERS", type=int,
                    help="Number of numbers to operate. Default 6", default=6, nargs="?")
parser.add_argument("OBJECTIVE", type=int,
                    help="Upper bound of the objective. Default 999", default=999, nargs="?")
parser.add_argument("POPULATION", type=int,
                    help="Size of the population. Default 50", default=50, nargs="?")
parser.add_argument("GENERATIONS", type=int,
                    help="Number of generations. Default 2000", default=2000, nargs="?")
parser.add_argument("--verbose", help="Verbose.", action="store_true")

args = parser.parse_args()

NUMBERS = [random.randint(1, 100) for i in range(args.NUMBERS)]
OPERATORS = ["+", "-", "*", "/"]
OBJECTIVE = random.randint(100, args.OBJECTIVE)
DNA_SIZE = len(NUMBERS) - 1
POPULATION_SIZE = args.POPULATION
GENERATIONS = args.GENERATIONS
VERB = args.verbose

# Problem specific functions
# Functions used as support that are not related to the Genetic Algorithm


def operation(op, n1, n2):
    '''
    Given an operator and 2 numbers, calculates: n1 operator n2
    '''
    result = n1
    if op == "+":
        result += n2
    elif op == "-":
        result -= n2
    elif op == "*":
        result *= n2
    elif op == "/":
        if result % n2 == 0:
            result //= n2
        else:
            return math.inf
    return result


def iterative_best():
    '''
    Algorithm that calculates the best order of operators in a sequential brute force way.
    '''
    best_result = math.inf
    for chainop in tqdm(itertools.product(OPERATORS, repeat=DNA_SIZE), desc="Iterative Solution",total=pow(4, DNA_SIZE)):
        result = NUMBERS[0]
        for i, op in enumerate(chainop):
            result = operation(op, result, NUMBERS[i + 1])
        if result == OBJECTIVE:
            best_result = result
            best_chain = chainop
            print("\nFound exact solution! Ending the loop...")
            break
        elif abs(result - OBJECTIVE) < abs(best_result - OBJECTIVE):
            best_result = result
            best_chain = chainop

    return list(best_chain), best_result


def weighted_choice(items):
    """
    Chooses a random element from items, where items is a list of tuples in
    the form (item, weight). weight determines the probability of choosing its
    respective item.
    """
    weight_total = sum((item[1] for item in items))
    n = random.uniform(0, weight_total)
    for item, weight in items:
        if n < weight:
            return item
        n -= weight
    return item


def random_op():
    """
    Return a random operator. Useful for generating mutations and random population.
    """
    return OPERATORS[random.randint(0, 3)]


def random_population():
    """
    Return a list of POPULATION_SIZE individuals.
    """
    population = []
    for i in range(POPULATION_SIZE):
        dna = []
        for j in range(DNA_SIZE):
            dna.append(random_op())
        population.append(dna)
    return population


# Genetic Algorithm functions

def fenotipye(dna):
    '''
    Calculates the fenotype given a set of operations (DNA)
    '''
    result = NUMBERS[0]
    for i in range(len(dna)):
        result = operation(dna[i], result, NUMBERS[i + 1])
    return result


def fitness(dna):
    """
    Calculates the diference between the objective value and the fenotype value.
    Less is better
    """
    return abs(OBJECTIVE - fenotipye(dna))


def mutate(dna):
    """
    For each gene in the DNA, there is a 1/mutation_chance chance that it will be
    switched out with a random operator. This ensures diversity in the
    population!!!
    """
    dna_out = []
    mutation_chance = 100
    for i in range(DNA_SIZE):
        if int(random.random() * mutation_chance) == 1:
            dna_out.append(random_op())
        else:
            dna_out.append(dna[i])
    return dna_out


def crossover(dna1, dna2):
    """
    Slices both dna1 and dna2 into 3 parts at a random index within their
    length and merges them.
    """
    pos_ini = int(random.random() * DNA_SIZE)
    pos_fin = pos_ini + int(random.random() * (DNA_SIZE - pos_ini))
    return(dna1[:pos_ini] + dna2[pos_ini:pos_fin] + dna1[pos_fin:], dna2[:pos_ini] + dna1[pos_ini:pos_fin] + dna2[pos_fin:])


# Main program
# Generate a population and simulate GENERATIONS generations.

if __name__ == "__main__":
    print("\nNUMBERS: %s" % NUMBERS)
    print("OBJECTIVE: %d\n" % OBJECTIVE)

    # Generate initial population.
    time_ini = time.clock()
    population = random_population()
    best_fitness = fitness(population[0])
    best_dna = population[0]
    # Simulate all of the generations.
    for generation in tqdm(range(GENERATIONS), desc="GA Solution",total=GENERATIONS):
        # print("Generation %s... Random sample: '%s'" % (generation, population[0]))
        if VERB:
            print("\nGeneration %s \nPopulation:" % generation)

        weighted_population = []

        # Add individuals and their respective fitness levels to the weighted
        # population list.
        for individual in population:
            fitness_val = fitness(individual)
            if VERB:
                print("  -Individual: %s  -Fenotype: %7s  -Fitness: %7s" %
                      (individual, fenotipye(individual), fitness_val))
            # Compare to best
            if fitness_val <= best_fitness:
                best_fitness = fitness_val
                best_dna = individual

            # Generate the (individual,fitness) pair
            pair = (individual, 100.0 / (fitness_val + 1.0))

            weighted_population.append(pair)

        population = []

        # Select two random individuals, based on their fitness probabilites, cross
        # their genes over at a random point, mutate them, and add them back to the
        # population for the next iteration.
        for _ in range(POPULATION_SIZE // 2):
            # Selection
            ind1 = weighted_choice(weighted_population)
            ind2 = weighted_choice(weighted_population)

            # Crossover
            ind1, ind2 = crossover(ind1, ind2)

            # Mutate and add back into the population.
            population.append(mutate(ind1))
            population.append(mutate(ind2))

    # Display the highest-ranked operation sequence after all generations have been iterated
    # over. Also show the global best and the iterative solution for comparation.
    fittest_string = population[0]
    minimum_fitness = fitness(population[0])

    for individual in population:
        ind_fitness = fitness(individual)
        if ind_fitness <= minimum_fitness:
            fittest_string = individual
            minimum_fitness = ind_fitness
    time_end = time.clock() - time_ini


    print("Last population best individual: %s  -Fenotype: %s" % (fittest_string, fenotipye(best_dna)))
    print("Global Best individual:          %s  -Fenotype: %s" % (best_dna, fenotipye(best_dna)))
    print("Genetic algorithm time: %.4f" % time_end)
    time_ini = time.clock()
    iterative_solution = iterative_best()
    time_end = time.clock() - time_ini
    print("Iterative Best:                  %s  -Fenotype: %s" % iterative_solution)
    print("Brute force algorithm time: %.4f" % time_end)
    exit(0)
