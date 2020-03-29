
import pandas as pd
import numpy as np

#functions

def calculate_fitness(coefficients, pop):
    fitness = np.sum(pop*coefficients, axis=1)
    
    return fitness

def mating_parents(pop, fitness, parents_size):
    parents = np.empty((parents_size, pop.shape[1]))

    for parent_num in range(parents_size):
        max_fitness_ID = np.where(fitness == np.max(fitness))
        max_fitness_ID = max_fitness_ID[0][0]
        parents[parent_num, :] = pop[max_fitness_ID, :]
        fitness[max_fitness_ID] = -99999999999

    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1]/2)
    for k in range(offspring_size[0]):
        parent1_ID = k%parents.shape[0]
        parent2_ID = (k+1)%parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_ID, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_ID, crossover_point:]
     
    return offspring



def mutation(offspring_crossover, mutations_size=1):
    mutations_counter = np.uint8(offspring_crossover.shape[1] / mutations_size)
    for idx in range(offspring_crossover.shape[0]):
        gene_ID = mutations_counter - 1
        for mutation_num in range(mutations_size):
            random_value = np.random.uniform(0, 1.0, 1)
            offspring_crossover[idx, gene_ID] = offspring_crossover[idx, gene_ID] + random_value
            gene_ID = gene_ID + mutations_counter
            
    return offspring_crossover


def pop_actual(a):
    sum_matrix = a.sum(axis=1)
    for j in range(3):
        for i in range(10):    
            a[i][j] = a[i][j]/sum_matrix[i]
            
    return a

def constraintFunc(b):
    subs = np.zeros((10,3))
    for i in range(10):
        for j in range(3):
            if (b[i][j]<0.1):
                sub = (0.1 - b[i][j])
                subs[i][j] = sub

    sum_sub = subs.sum(axis=1)
    b[b<0.1]=0.1
    row_occurrences = np.count_nonzero(b == 0.1, axis=1)
    for i in range(10):
        if (row_occurrences[i] == 1):
            for j in range(3):
                if (b[i][j]!=0.1):
                    b[i][j] = b[i][j]-(sum_sub[i]/2)
        else:
            for j in range(3):
                if (b[i][j]!=0.1):
                    b[i][j] = b[i][j]-sum_sub[i]
    
    return b

#code

coefficients = [0.6972, 0.5, 0.9223]

num_coefficients = 3

row_size = 10

parents_mating_size = 6

pop_size = (row_size,num_coefficients)

new_population = np.random.uniform(low=10, high=20.0, size=pop_size)
new_population = pop_actual(new_population)
print ("Initial population:\n", new_population)


best_outputs = []
num_generations = 5

for generation in range(num_generations):

    print("Generation : ", generation)

    fitness = calculate_fitness(coefficients, new_population)
    
    best_outputs.append(np.max(np.sum(new_population*coefficients, axis=1)))

    parents = mating_parents(new_population, fitness, parents_mating_size) 

    offspring_crossover = crossover(parents,offspring_size=(pop_size[0]-parents.shape[0], num_coefficients)) 

    offspring_mutation = mutation(offspring_crossover)


    new_population[0:parents.shape[0], :] = parents

    new_population[parents.shape[0]:, :] = offspring_mutation
    
    new_population = pop_actual(new_population)
    new_population = constraintFunc(new_population)


    print("Best result : ", np.max(np.sum(new_population*coefficients, axis=1))) 

fitness = calculate_fitness(coefficients, new_population)

best_fitness_ID = np.where(fitness == np.max(fitness))
best_fitness_ID = np.asarray(best_fitness_ID)

if(best_fitness_ID.ndim!=1):
    c = int(np.random.choice(best_fitness_ID.shape[0], 1, replace=False))    
    d = int(np.random.choice(best_fitness_ID[c], 1, replace=False))
    
    best_fitness_ID = best_fitness_ID[c][d]
    
    
print("Best solution : ", new_population[best_fitness_ID])
print("Best solution fitness : ", fitness[best_fitness_ID])


import matplotlib.pyplot
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()