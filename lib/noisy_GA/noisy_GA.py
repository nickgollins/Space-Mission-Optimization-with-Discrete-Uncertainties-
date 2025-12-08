import numpy as np
import pandas as pd
import pygmo as pg
from copy import copy
from scipy.special import logsumexp
import random
from itertools import product
from dataclasses import dataclass
from typing import Optional, Any
import concurrent.futures
from functools import partial
from time import time, sleep
# import tqdm
import csv
import os
# from multiprocessing import Manager, Process
# from concurrent.futures import ThreadPoolExecutor, as_completed

from lib.noisy_GA.design_dict import design_history_entry

"""
Noisy Genetic Algorithm

Genetic algorithm for when discrete decision variables have some noise associated with them.
"""

@dataclass
class noisy_GA:
    '''
        :param gen: Number of generations
        :param prob_cross: Crossover probability
        :param prob_mut:  Mutation probability per gene
        :param noise: Noise distrubition (list)
        :param inner_problem: Constrained pygmo problem
        :param noise_samples_per_point: How many noise samples to take per design point
        :param point_samples_per_gen: How many design points to sample the noise distribtion of per generation
        :param end_noise_samples_per_point: How many noise samples to take per design point at the end of the algorithm
                scales linearly with generation number
        :param output_file: File to write progress to
         (sample the best ones)
    '''

    noise:np.ndarray[Any, np.dtype[np.float16]]
    gen:int
    prob_cross:float = 0.2
    prob_mut:float = 0.01
    num_backprop_per_point:int = 1
    noise_samples_per_point:int = 9
    point_samples_per_gen:int = 1
    end_noise_samples_per_point:Optional[int]=None
    output_file:Optional[str] = False
    MOO:bool = False
    frac_cores:float = 0.25 # Fraction of cores to use for parallel backpropagation

    def __post_init__(self):
        assert self.gen > 0, "Number of generations must be greater than 0"
        assert self.noise_samples_per_point > 0, "Noise samples per point must be greater than 0"
        assert self.point_samples_per_gen > 0, "Point samples per generation must be greater than 0"
        assert self.prob_cross <= 1 and self.prob_cross >= 0, "Crossover probability must be between 0 and 1"
        assert self.prob_mut <= 1 and self.prob_mut >= 0, "Mutation probability must be between 0 and 1"

class NGA():
    def __init__(self,
                 noisy_GA:noisy_GA):
        '''Noisy Genetic Algorithm

        Genetic algorithm for when discrete decision variables have some noise associated with them.

        Inputs:
            noisy_GA: noisy_GA dataclass
            
        '''

        self.__iter = noisy_GA.gen
        self.noise = noisy_GA.noise  # Noise distribution
        self.prob_cross = noisy_GA.prob_cross
        self.prob_mut = noisy_GA.prob_mut

        self.noise_len = np.array([n.shape[0] for n in self.noise], dtype=int)  # Length of noise for each variable

        print("Building noise combinations array...")
        self.combinations = np.array(tuple(product(*[[n for n in range(N)] for N in self.noise_len])))  # All possible combinations of noise

        print("Building noise probability array...")
        self.distr_prob = np.array([np.prod([self.noise[i][x] for i, x in enumerate(self.combinations[j])]) for j in
                            range(len(self.combinations))], dtype=np.float64)  # Probabilities of each combination occuring
    
        
        print("Taking log of probability array...")
        self.distr_log_prob = np.array([np.sum([np.log(self.noise[i][x]) for i, x in enumerate(self.combinations[j])]) for j in
                            range(len(self.combinations))], dtype=np.float64)  # Log probability: easier to handle small numbers
        
        self.base_point_prob = self.distr_log_prob[0]
                                
        self.design_dict = {}  # Design dictionary

        self.num_backprop_per_point = noisy_GA.num_backprop_per_point
        if self.num_backprop_per_point == 1:
            self.num_backprop_per_point = len(self.combinations)-1

        self.noise_samples_per_point = noisy_GA.noise_samples_per_point
        self.point_samples_per_gen = noisy_GA.point_samples_per_gen
        self.end_noise_samples_per_point = noisy_GA.end_noise_samples_per_point
        
        self.MOO = noisy_GA.MOO
        
        self.frac_cores = noisy_GA.frac_cores

        self.output_file = noisy_GA.output_file

    def evolve(self, 
               pop:pg.population,
               feasible_check:callable,
               weighting_func:callable,
               keep_champion:bool=True,
               output_file:Optional[str]=None
            ) -> pg.population:
        '''Evolve the population over self.__iter generations

        Inputs:
            pop: Initial population
            constraint_func: Function that takes a decision vector and returns a boolean indicating if it is feasible

        Returns:
            Evolved population
        '''
        num_indiv = len(pop)

        if num_indiv == 0:
            return pop

        prob = pop.problem
        lb, ub = prob.get_bounds()

        assert self.noise.shape[0] == len(lb) == len(ub), "Every decision must have a noise distribution"

        for it in range(self.__iter):
            print("Generation ", it, ": Current champion w/o noise: ", pop.champion_x, pop.champion_f)
            print("Updating design dictionary...")
            self.update_dicts(pop)
            backprop_start = time()
            print("Back-propagating...")
            self.backpropagate(pop, feasible_check)
            print("Back-propagation time: ", time()-backprop_start)
            forwardprop_start = time()
            print("Forward-propagating...")
            self.forwardpropagate(pop, feasible_check)
            print("Forward-propagation time: ", time()-forwardprop_start)
            print("Finding new champion...")
            champion_start = time()
            champion = self.find_champ()
            print("Champion find time: ", time()-champion_start)

            # print(f"Current champion w/ noise: {champion['design_point']} Expected Value = {champion['expected_value']}")

            if self.output_file is not False:
                with open(self.output_file, 'a') as f:
                    f.write(', '.join(str(item) for item in [it, champion]) + '\n')

            print("Getting new population...")
            get_new_pop_start = time()
            if keep_champion: pop = self.get_new_pop(pop, weighting_func=weighting_func, feasible_check=feasible_check, champion=champion)
            else: pop = self.get_new_pop(pop, weighting_func=weighting_func, feasible_check=feasible_check)
            print("Get new pop time: ", time()-get_new_pop_start)

            if self.end_noise_samples_per_point is not None:
                # Linearly proceeding changing in samples per point
                self.noise_samples_per_point = int(self.noise_samples_per_point + (self.end_noise_samples_per_point-self.noise_samples_per_point)*it/self.__iter)

            if output_file is not None:
                with open(output_file, 'a') as f:
                    self.printdict2file(output_file)

        return pop
    
    def update_dicts(self, pop:pg.population) -> None:
        """Check if the current population members exist in the evaluation history, and update the design dictionary accordingly. 
        Nothing happens if the current population is already in the evaluation history.

        Inputs:
            pop: Current population of decision vectors
        
        """
        for i in range(len(pop)):
            current_point = pop2tuple(pop.get_x()[i])
            fitness = pop.get_f()[i]

            if current_point in self.design_dict:
                entry = self.design_dict[current_point]
                if self.check_feasibility(fitness):
                    if entry.check_base_point_eval(self.base_point_prob):
                        pass
                    else:
                        entry.update_feasible(fitness[0], self.base_point_prob)
                        entry.base_objective = fitness[0]
                else:
                    entry = None
            else:
                if self.check_feasibility(fitness):
                    self.add_new_point(current_point, fitness[0], self.base_point_prob)
                    entry = self.design_dict[current_point]
                    entry.base_objective = fitness[0]
                else:
                    self.design_dict[current_point] = None
        return
               
    def backpropagate_parallel(self, shared_dict:dict, shared_set:set, core:int, original_point, fitness, backprop_samples, feasibility_check):
        entry = shared_dict[original_point]
        if entry is not None:   
            for i, distr_point in enumerate(backprop_samples):
            # for ind, distr_point in enumerate(self.get_distr(original_point, lb)[core*len(self.combinations)//num_cores:(core+1)*len(self.combinations)//num_cores]):
                # ind=ind+core*len(self.combinations)//num_cores
                ind = core + i*int(os.cpu_count()*self.frac_cores)
                
                distr_point = pop2tuple(distr_point)
                if feasibility_check(distr_point) and distr_point not in self.design_dict[original_point].prop_map:
                    if distr_point in self.design_dict:
                        distr_entry = self.design_dict[distr_point]
                        if distr_entry is not None:
                            if original_point not in distr_entry.prop_map:
                                distr_entry.add_to_prop_map(original_point)
                                if self.check_feasibility(fitness):
                                    distr_entry.update_feasible(fitness[0], self.distr_log_prob[ind+1])                           
                                else:
                                    distr_entry.update_infeasible(self.distr_log_prob[ind+1])
                    else:
                        if self.check_feasibility(fitness):
                            self.add_new_point(distr_point, fitness[0], self.distr_log_prob[ind+1])
                            self.design_dict[distr_point].add_to_prop_map(original_point)
                        else:
                            self.add_new_point(distr_point, infeasible_prob=self.distr_log_prob[ind+1])
                            self.design_dict[distr_point].add_to_prop_map(original_point)
                    shared_dict[distr_point] = self.design_dict[distr_point]
                
                shared_set += [(distr_point)]


        return 

    def backpropagate(self, 
                  pop:pg.population,
                  feasibility_check:callable
                ) -> None:
        '''Backpropagate the fitness of the population over the noise distribution to the design dictionary

        Inputs:
            pop: Currently evaluated population of decision vectors
            feasibility_check: Function that takes a decision vector and returns a boolean indicating if it is feasible
                according to the UDP
        '''
       
        prob = pop.problem
        lb = prob.get_bounds()[0]
        
        for i in range(len(pop)):
            original_point = pop2tuple(pop.get_x()[i])
            assert original_point in self.design_dict, "Trying to backpropagate a point that hasn't yet been evaluated"
            backprop_samples = self.get_distr(original_point, lb)[:self.num_backprop_per_point]
            fitness = pop.get_f()[i]
            entry = self.design_dict[original_point]
            if entry is not None:                
                for i, distr_point in enumerate(backprop_samples):                
                    distr_point = pop2tuple(distr_point)
                    if feasibility_check(distr_point) and distr_point not in self.design_dict[original_point].prop_map:
                        if distr_point in self.design_dict:
                            distr_entry = self.design_dict[distr_point]
                            if distr_entry is not None:
                                if original_point not in distr_entry.prop_map:
                                    distr_entry.add_to_prop_map(original_point)
                                    if self.check_feasibility(fitness):
                                        distr_entry.update_feasible(fitness[0], self.distr_log_prob[i+1])                           
                                    else:
                                        distr_entry.update_infeasible(self.distr_log_prob[i+1])
                        else:
                            if self.check_feasibility(fitness):
                                self.add_new_point(distr_point, fitness[0], self.distr_log_prob[i+1])
                                self.design_dict[distr_point].add_to_prop_map(original_point)
                            else:
                                self.add_new_point(distr_point, infeasible_prob=self.distr_log_prob[i+1])
                                self.design_dict[distr_point].add_to_prop_map(original_point)
                        self.design_dict[original_point].add_to_prop_map(distr_point)
           

    def forwardpropagate(self, 
                         pop:pg.population, 
                         feasibility_check:callable,
                         break_count:int=1000
                        ) -> None:
        '''Forward propagate the fitness of the specified number of population members (self.point_samples_per_gen)
         over the noise distribution to the design dictionary with (self.point_samples_per_gen) samples per sampled
          individual

        Inputs:
            pop: Currently evaluated population of decision vectors
            feasibility_check: Function that takes a decision vector and returns a boolean indicating if it is feasible
                according to the UDP
            break_count: Counter to break the sample generation loop if it gets stuck
        '''
        
        prob = pop.problem
            
        for i in np.argsort([pop.get_f()[j][0] for j in range(len(pop))])[:self.point_samples_per_gen+1]:
            start_time = time()
            original_point = pop2tuple(pop.get_x()[i])
            assert original_point in self.design_dict, f"Point {original_point} not evaluated before forward propagation"

            entry = self.design_dict[original_point]

            if entry is None:
                print("Attempting to propagate from infeasible point, moving to next pop")
            elif sum(np.exp(entry.feasible_prob_list)) + sum(np.exp(entry.infeasible_prob_list)) == 1:
                print(f"All points already propagated from {original_point}, moving to next pop")
            elif sum(np.exp(entry.feasible_prob_list)) + sum(np.exp(entry.infeasible_prob_list)) > 1.0001:
                print(entry.check_attr())
                raise ValueError(f"Sum of feasible and infeasible probabilities for {original_point} i = {sum(np.exp(entry.feasible_prob_list)) + sum(np.exp(entry.infeasible_prob_list))}")
            else:
                sample_counter = 0
                break_counter = 0

                while sample_counter < self.noise_samples_per_point and break_counter < break_count:
                    # Sample a point from the distribution of individual according to weighted distribution
                    noise_sample = random.choices(range(1,len(self.combinations)), weights=self.distr_prob[1:])[0]
                    sample = tuple(a+b for a,b in zip(original_point, pop2tuple(self.combinations[noise_sample])))

                    # Discard sample if already propagated
                    if sample in entry.prop_map:
                        break_counter += 1
                    else:
                        entry.add_to_prop_map(sample)
                        sample_counter += 1
                        sample_log_prob = self.distr_log_prob[noise_sample]
                        sample_fitness = prob.fitness(sample)

                        if self.check_feasibility(sample_fitness) and feasibility_check(sample):
                            # print("Feasible forward prop", original_point, '->', sample)
                            entry.update_feasible(sample_fitness[0], sample_log_prob)
                            if sample in self.design_dict:
                                sample_entry = self.design_dict[sample]
                                sample_entry.base_objective = sample_fitness[0]
                                if original_point not in sample_entry.prop_map:
                                    sample_entry.add_to_prop_map(original_point)
                                    if self.base_point_prob not in sample_entry.feasible_prob_list:
                                        sample_entry.update_feasible(sample_fitness[0], self.base_point_prob)
                                    else:
                                        pass
                                # self.update_point_feasible(sample, sample_fitness[0], self.base_point_prob)
                            else:
                                self.add_new_point(sample, sample_fitness[0], self.base_point_prob)
                                sample_entry = self.design_dict[sample]
                                sample_entry.base_objective = sample_fitness[0]
                            
                            # self.update_point_feasible(original_point, sample_fitness[0], sample_log_prob)
                        else:
                            # print("infeasible forward prop", original_point, '->', sample)
                            entry.update_infeasible(sample_log_prob)
                            if sample in self.design_dict:
                                sample_entry = self.design_dict[sample]
                                sample_entry = None
                            else:
                                self.design_dict[sample] = None

            print(f"Forward propagated from {original_point} in {time()-start_time} seconds")


    def add_new_point(self, 
                    design_point:tuple[int], 
                    objective:Optional[float]=None, 
                    feasible_prob:Optional[float]=None,
                    infeasible_prob:Optional[float]=None
                ) -> None:
        """Add the base point evaluation to the design dictionary.

        Inputs:
            design_point: Design point to add
            objective: Objective value of the design point
            feasible_prob: Feasible probability of the design point
            infeasible_prob: Infeasible probability of the design point        
        """
        
        assert (objective is not None and feasible_prob is not None) \
            or (objective is None and feasible_prob is None), "If objective is provided, feasible probability must be provided. If one is not provided, neither should the other."
        
        assert (feasible_prob is None and infeasible_prob is not None) \
            or (feasible_prob is not None and infeasible_prob is None), "Either feasible or infeasible probability must be provided, but not both"

        assert design_point not in self.design_dict.keys(), f"Design point {design_point} already exists in design dictionary"

        if objective is not None:
            self.design_dict[design_point] = design_history_entry(point=design_point,
                                                                objective_list=np.array([objective], dtype=np.float64),
                                                                feasible_prob_list=np.array([feasible_prob], dtype=np.float64))
        else:
            self.design_dict[design_point] = design_history_entry(point=design_point,
                                                                infeasible_prob_list=np.array([infeasible_prob], dtype=np.float64))

        return

    def find_champ(self):
        '''Find the best expected value in the design dictionary for which the base point has been evaluated
        
        '''
        best_exp_obj = 1E15
        for design in self.design_dict.values():
            if design is not None:
                if design.expected_value is not None:
                    if design.expected_value < best_exp_obj and design.check_base_point_eval(self.base_point_prob):
                        best_exp_obj = design.expected_value
                        best_design = design
        if self.MOO:
            best_prob_infeas = 1
            for design in self.design_dict.values():
                if design is not None:
                    if design.expected_value is not None:
                        if np.isclose(design.expected_value, best_exp_obj) and design.check_base_point_eval(self.base_point_prob):
                            if design.expected_infeasible_prob is None:
                                expected_prob_infeas = 1E-15 
                            else: 
                                expected_prob_infeas = design.expected_infeasible_prob
                            if expected_prob_infeas < best_prob_infeas:
                                best_design = design
                                best_prob_infeas = expected_prob_infeas

        return best_design
           
    def calc_selection_weights(self, design_entry, feasible_check:callable, weighting_func:callable):
        if design_entry is not None:
            if design_entry.expected_value not in (None, 1E15) and feasible_check(design_entry.point):
                weight = weighting_func(design_entry)
                return weight, design_entry.point
            
    def get_new_pop(self, 
                    pop:pg.population,
                    feasible_check:callable,
                    weighting_func:callable,
                    champion:Optional[tuple[int]]=None
                ) -> pg.population:
        """Get a new population from the best points in the design dictionary according to the weighting function.
            If the champion is provided, it is kept in the population with certainty.

        Inputs:
            pop:  Current population
            champion: Current champion point
            weighting_func: Function to weight the probability of selection of the members of the new population

        Returns:
            pop: Updated population
        """
        num_indiv = len(pop)

        weights = []
        pop_options = []

        for design_entry in self.design_dict.values():
            if design_entry is not None:
                if design_entry.expected_value is None:
                    expected_value  = 1E15
                else:
                    expected_value = design_entry.expected_value
                if design_entry.expected_infeasible_prob in (0, None):
                    expected_infeasible_prob = 1E-10
                else:
                    expected_infeasible_prob = design_entry.expected_infeasible_prob
                weights.append(weighting_func(expected_value*expected_infeasible_prob))
                pop_options.append(design_entry.point)
            

        new_pop = random.choices(pop_options, weights=weights, k=num_indiv) 

        if self.prob_cross != 0: new_pop = self.crossover(new_pop, feasible_check)
        if self.prob_mut != 0: new_pop = self.mutate(new_pop, pop.problem.get_bounds()[0], pop.problem.get_bounds()[1], feasible_check)    

        for i, design in enumerate(new_pop):
            if champion is not None and i == 0:
                if len(champion.feasible_prob_list) == 1:
                    assert len(champion.objective_list) == 1, "Champion only has one feasible evaluation but has multiple feasible objectives"
                    f = np.append(champion.objective_list, [0]*len(pop.get_f()[0][1:]))
                else:
                    j = np.where(champion.feasible_prob_list == self.base_point_prob)[0][0]
                    f = np.append(champion.objective_list[j], [0]*len(pop.get_f()[0][1:]))
                pop.set_xf(i, champion.point, f)
            else:
                if design in self.design_dict:
                    design_entry = self.design_dict[design]
                    assert design_entry is not None, f"Design point {design} is in design dictionary but is infeasible"
                    if design_entry.check_base_point_eval(self.base_point_prob):
                        j = np.where(design_entry.feasible_prob_list == self.base_point_prob)[0][0]
                        f = np.append(design_entry.objective_list[j], [0]*len(pop.get_f()[0][1:]))
                        pop.set_xf(i, design_entry.point, f)
                    else:
                       pop.set_x(i, design_entry.point) 
                else:
                    pop.set_x(i, design)
        
        print("New pop:", pop)
        return pop

    def crossover(self,
                  pop:list[tuple[int]],
                  feasibility_check:callable
                ) -> list[tuple[int]]:
        '''Crossover two random pop individuals in piecewise fashion, formatted for noisy GA

        Inputs:
           pop: Input population formatted as list of design points
           feasibility_check: Function that takes a decision vector and returns a boolean indicating if it is feasible

        Returns:
           new_pop: Updated population
        '''
        new_pop = []
        counter = 0

        while len(new_pop) < len(pop):
            if random.random() < self.prob_cross:  # Crossover with user-defined probability
                # Pick two random pops
                parent1, parent2 = random.randint(0, len(pop) - 1), random.randint(0, len(pop) - 1)
                # One point split
                split = random.randint(0, len(pop[0]))  # Randomly select a split point
                crossedover = np.concatenate((pop[parent1][:split], pop[parent2][split:]))
                if feasibility_check(crossedover):
                    new_pop.append(tuple(crossedover))
                    counter += 1
            else:  # Otherwise keep an original pop member
                new_pop.append(pop[counter])
                counter += 1
        return new_pop

    def mutate(self, 
               pop:list[tuple[int]],
               lb: list[int],
               ub: list[int],
               feasibility_check:callable
            ) -> list[tuple[int]]:
        '''Mutate each component of each pop member with a user-defined probability

        Inputs:
            pop: Input population
            lb: Lower bounds of the design space
            ub: Upper bounds of the design space
            feasibility_check: Function that takes a decision vector and returns a boolean indicating if it is feasible

        Returns:
            new_pop: Updated population
        '''
        counter = 0
        new_pop = []
        while len(new_pop) < len(pop):
            mutated = pop[counter]
            mutated = [x for x in mutated]
            for j in range(len(mutated)):
                if random.random() < self.prob_mut:
                    mutated[j] = random.randint(int(lb[j]), int(ub[j]))
            if feasibility_check(mutated) and all(mutated != ind for ind in new_pop):
                new_pop.append(tuple(mutated))
                counter += 1
        return new_pop

    def get_distr(self, 
                  pop_point:list[int|float], 
                  lb:list[int|float]
                ) -> list:
        """
        Get the points in the design space which a pop point fits within the noise distribution of

        Args:
            :param pop_point: Design vector length n
            :param noise_len:  Vector length n containing the range in design space that the noise can distort by
        Returns:
            :return: List of feasible design points in the noise distribution, each of which are vectors length n.
        """
        distr = []
        for i in range(1, len(self.combinations)):
            new_distr = np.array(pop_point) - np.array(self.combinations[i])
            if min(new_distr - lb) >= 0:
                distr.append(tuple(new_distr))
        return distr

    def calc_pareto_prob(self, design_entry) -> float:
        '''Calculate the probability of a design point being on the pareto front
        
        Args:
            :param design_entry: Design point to calculate the probability of being on the pareto front
        Returns:
            :return: Probability of the design point being on the pareto front
        '''
        prob = 1

        for other_entry in self.design_dict.values():
            if other_entry is not None:
                if other_entry.point != design_entry.point:
                    if other_entry.expected_value is not None and other_entry.expected_value_std_error is not None:
                        if design_entry.expected_value_std_error is None:
                            expected_value_std_error = 1E-15
                        else:
                            expected_value_std_error = design_entry.expected_value_std_error
                        m_obj = (design_entry.expected_value - other_entry.expected_value)/other_entry.expected_value_std_error
                        s_obj = expected_value_std_error/other_entry.expected_value_std_error
                        prob *= 1 - 1/(1 + np.exp(-2.5*m_obj/np.sqrt(2+2*s_obj**2)))
                    if other_entry.expected_infeasible_prob is not None and other_entry.expected_infeasible_prob_std_error is not None:
                        if design_entry.expected_infeasible_prob in (0, None):
                            expected_prob_infeas = 1E-15 # Avoid division by zero
                        else: 
                            expected_prob_infeas = design_entry.expected_infeasible_prob
                        if design_entry.expected_infeasible_prob_std_error is None:
                            expected_prob_infeas_std_error = 1E-15
                        else:
                            expected_prob_infeas_std_error = design_entry.expected_infeasible_prob_std_error
                        m_infeas_prob = (expected_prob_infeas - other_entry.expected_infeasible_prob)/other_entry.expected_infeasible_prob_std_error
                        s_infeas_prob = expected_prob_infeas_std_error/other_entry.expected_infeasible_prob_std_error
                        prob *= 1 - 1/(1 + np.exp(-2.5*m_infeas_prob/np.sqrt(2+2*s_infeas_prob**2)))

        assert prob >= 0 and prob <= 1, f"Probability of design point {design_entry.point} being on the pareto front is not between 0 and 1"
        return prob

    def check_feasibility(self, fitness_vector:list[float]) -> bool:
        if fitness_vector[0] < 1E15 and all(fitness_vector[1:] <= 0):
            return True
        else:  
            return False
        
    def printdict2file(self, filename:str) -> None:
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['point', 'base_objective', 'expected_value', 'expected_value_error', 
                          'expected_prob_infeasibility', 'expected_prob_infeasibility_error', 
                          'sum_total_prob']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for point, entry in self.design_dict.items():
                if entry is not None:
                    writer.writerow({
                        'point': point,
                        'base_objective': entry.base_objective,
                        'expected_value': entry.expected_value,
                        'expected_value_error': entry.expected_value_std_error,
                        'expected_prob_infeasibility': entry.expected_infeasible_prob,
                        'expected_prob_infeasibility_error': entry.expected_infeasible_prob_std_error,
                        'sum_total_prob': sum(np.exp(entry.feasible_prob_list)) + sum(np.exp(entry.infeasible_prob_list))
                    })
           
def pop2tuple(pop_member:np.ndarray) -> list[tuple]:
    '''Convert a population member to a tuple
    '''
    return tuple([int(x) for x in pop_member])    
