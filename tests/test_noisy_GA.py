import pytest
import numpy as np
import pygmo as pg

from example_problem import ExampleProblem

import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from lib.noisy_GA.noisy_GA import noisy_GA, NGA
from lib.noisy_GA.design_dict import design_history_entry


@pytest.fixture
def example_noisy_GA() -> noisy_GA:
    return noisy_GA(noise=np.array([[0.5, 0.3, 0.2] for _ in range(2)]),
                    gen = 2,
                    noise_samples_per_point=2,
                    num_backprop_per_point=1,
                    )

@pytest.fixture
def example_MO0_noisy_GA() -> noisy_GA:
    return noisy_GA(noise=np.array([[0.5, 0.3, 0.2] for _ in range(2)]),
                    gen = 2,
                    noise_samples_per_point=2,
                    MOO=True
                    )

@pytest.fixture
def example_NGA(example_noisy_GA) -> NGA:
    return NGA(noisy_GA=example_noisy_GA)

@pytest.fixture
def test_dict() -> np.array:
    point = (0, 1)
    entry = design_history_entry(point = point, 
                        objective_list=np.array([100], dtype=np.float64), 
                        feasible_prob_list=np.array([np.log(0.25)], dtype=np.float64),
                    )
    return {point: entry}

@pytest.fixture
def test_dict2() -> np.array:
    point = (1, 2)
    entry = design_history_entry(point = point, 
                        objective_list=np.array([200], dtype=np.float64), 
                        feasible_prob_list=np.array([np.log(0.25)], dtype=np.float64),
                    )
    return {point: entry}

@pytest.fixture
def test_dict1_update_feasible() -> np.array:
    return np.array([([0, 1], [100, 200], [np.log(0.25), np.log(0.05)], 116.666666667, [])],
                    dtype = [('design_point', 'O'), 
                        ('objective_list', 'O'), 
                        ('feasible_prob_list', 'O'), 
                        ('expected_value', 'f8'), 
                        ('infeasible_prob_list', 'O')]
                    )

@pytest.fixture
def test_dict1_update_infeasible() -> np.array:
    return np.array([([0, 1], [100], [np.log(0.25)], 100, [np.log(0.05)])],
                    dtype = [('design_point', 'O'), 
                        ('objective_list', 'O'), 
                        ('feasible_prob_list', 'O'), 
                        ('expected_value', 'f8'), 
                        ('infeasible_prob_list', 'O')]
                    )
               
@pytest.fixture
def test_problem():
    return ExampleProblem()

@pytest.fixture
def test_population1() -> list[int]:
    return [0, 1]

@pytest.fixture
def test_population2() -> list[int]:
    return [5, 4]

@pytest.fixture
def test_population3() -> list[int]:
    return [1, 1]

def test_noisy_GA_init(example_noisy_GA) -> None:
    assert example_noisy_GA.noise[0][0] == 0.5
    assert example_noisy_GA.noise[0][1] == 0.3
    assert example_noisy_GA.noise[0][2] == 0.2
    assert example_noisy_GA.gen == 2
    assert example_noisy_GA.prob_cross == 0.2
    assert example_noisy_GA.prob_mut == 0.01
    assert example_noisy_GA.noise_samples_per_point == 2
    assert example_noisy_GA.point_samples_per_gen == 1  


def test_NGA_setup(example_NGA) -> None:
    assert np.allclose(example_NGA.distr_prob, np.array([0.25, 0.15, 0.1, 0.15, 0.09, 0.06, 0.1, 0.06, 0.04]))
    assert np.allclose(example_NGA.distr_log_prob, np.array([np.log(0.25), np.log(0.15), np.log(0.1), np.log(0.15), np.log(0.09),
                                                            np.log(0.06), np.log(0.1), np.log(0.06), np.log(0.04)]))
    assert example_NGA.base_point_prob == np.log(0.25)


def test_add_new_point(example_NGA, test_dict) -> None:
    example_NGA.add_new_point((0, 1), objective=100, feasible_prob=np.log(0.25))
    for i, attr in enumerate(example_NGA.design_dict[(0,1)].check_attr()):
        assert np.array_equal(attr, test_dict[(0,1)].check_attr()[i])


def test_find_champ(example_NGA, test_dict2) -> None:
    example_NGA.add_new_point((0, 1), objective=100, feasible_prob=np.log(0.25))
    example_NGA.add_new_point((1, 2), objective=200, feasible_prob=np.log(0.25))

    assert example_NGA.find_champ().point == (0,1)
                                

def test_update_dicts(example_NGA, test_problem, test_population1, test_population2) -> None:
    problem = test_problem
    algo = example_NGA
    pop = pg.population(problem, 0)
    pop.push_back(test_population1)
    pop.push_back(test_population2)
    
    test_design_dict = {(0, 1):  None,
                        (5, 4): design_history_entry(point = (5, 4), objective_list = np.array([161]), feasible_prob_list=np.array([np.log(0.25)]),
                                                     expected_value=161, base_objective=161)
                        }

    algo.update_dicts(pop)

    for key in algo.design_dict.keys():
        if algo.design_dict[key] is None:
            assert test_design_dict[key] is None, f"Key: {key}, algo: {algo.design_dict[key]}, test: {test_design_dict[key]}"
        else:
            for i, attr in enumerate(algo.design_dict[key].check_attr()):
                assert np.array_equal(attr, test_design_dict[key].check_attr()[i]), f"Key: {key}, i: {i}, attr: {attr}, test_attr: {test_design_dict[key].check_attr()[i]}"

def test_get_distr(example_NGA, test_problem, test_population2) -> None:
    problem = test_problem
    algo = example_NGA
    pop = pg.population(problem, 0)
    pop.push_back(test_population2)

    test_distr = [[5,3], [5,2],
                  [4,4], [4,3], [4,2],
                  [3,4], [3,3], [3,2]]

    assert  np.array_equal(algo.get_distr(pop.get_x()[0], problem.get_bounds()[0]), test_distr)


def test_backpropagate(example_NGA, test_problem, test_population1, test_population2) -> None:
    problem = test_problem
    algo = example_NGA
    pop = pg.population(problem, 0)
    pop.push_back(test_population1)
    pop.push_back(test_population2)
    algo.update_dicts(pop)
    algo.backpropagate(pop, problem.test_feasible)
    test_backpropagated_dict = {(0,1): None,
                                (5, 4): design_history_entry(point=(5,4), objective_list=np.array([161]), feasible_prob_list=[np.log(0.25)], expected_value=161, base_objective=161,
                                                             prop_map={(5,3), (5,2), (4,4), (4,3), (4,2), (3,4), (3,3), (3,2)}),
                                (5, 3): design_history_entry(point=(5,3), objective_list=np.array([161]), feasible_prob_list=[np.log(0.15)],
                                                             prop_map={(5,4)}),
                                (5, 2): design_history_entry(point=(5,2), objective_list=np.array([161]), feasible_prob_list=[np.log(0.1)],
                                                             prop_map={(5,4)}),
                                (4, 4): design_history_entry(point=(4,4), objective_list=np.array([161]), feasible_prob_list=[np.log(0.15)],
                                                             prop_map={(5,4)}),
                                (4, 3): design_history_entry(point=(4,3), objective_list=np.array([161]), feasible_prob_list=[np.log(0.09)],
                                                             prop_map={(5,4)}),
                                (4, 2): design_history_entry(point=(4,2), objective_list=np.array([161]), feasible_prob_list=[np.log(0.06)],
                                                             prop_map={(5,4)}),
                                (3, 4): design_history_entry(point=(3,4), objective_list=np.array([161]), feasible_prob_list=[np.log(0.1)],
                                                             prop_map={(5,4)}),
                                (3, 3): design_history_entry(point=(3,3), objective_list=np.array([161]), feasible_prob_list=[np.log(0.06)],
                                                             prop_map={(5,4)}),
                                (3, 2): design_history_entry(point=(3,2), objective_list=np.array([161]), feasible_prob_list=[np.log(0.04)],
                                                             prop_map={(5,4)})
                                }

    
    for key in algo.design_dict.keys():
        if algo.design_dict[key] is None:
            assert test_backpropagated_dict[key] is None, f"Key: {key}, algo: {algo.design_dict[key]}, test: {test_backpropagated_dict[key]}"
        else:
            for i, attr in enumerate(algo.design_dict[key].check_attr()):
                assert np.array_equal(attr, test_backpropagated_dict[key].check_attr()[i]), f"Key: {key}, i: {i}, attr: {attr}, test_attr: {test_backpropagated_dict[key].check_attr()[i]}"

def test_forwardpropagate(example_NGA, test_problem, test_population1, test_population3) -> None:
    problem = test_problem
    algo = example_NGA
    pop = pg.population(problem, 0)
    pop.push_back(test_population1)
    pop.push_back(test_population3)
    algo.update_dicts(pop)
    algo.forwardpropagate(pop, problem.test_feasible)

    assert len(algo.design_dict) == 4

def test_get_new_pop(example_NGA, test_problem, test_population1, test_population2) -> None:
    problem = test_problem
    algo = example_NGA
    pop = pg.population(problem, 0)
    pop.push_back(test_population1)
    pop.push_back(test_population2)
    algo.update_dicts(pop)
    algo.backpropagate(pop, problem.test_feasible)

    algo.get_new_pop(pop, problem.test_feasible, lambda x: 1/x)

    assert len(pop) == 2
    assert test_population1 not in pop.get_x()
    for i in pop.get_x():
        assert problem.test_feasible(i)
                       
def test_evolve(example_NGA, test_problem, test_population1, test_population2) -> None:
    problem = test_problem
    algo = example_NGA
    pop = pg.population(problem, 0)

    pop.push_back(test_population1)
    pop.push_back(test_population2)
    pop = algo.evolve(pop, problem.test_feasible, weighting_func=lambda x: 1/x)

    assert len(pop) == 2
    for i in pop.get_x():
        assert problem.test_feasible(i)

def test_MOO_evolve(example_MO0_noisy_GA, test_problem, test_population1, test_population2) -> None:
    problem = test_problem
    example_MO0_noisy_GA.gen = 10
    algo = NGA(example_MO0_noisy_GA)
    pop = pg.population(problem, 0)

    pop.push_back(test_population1)
    pop.push_back(test_population2)
    weighting = lambda x: 1/(x)
    pop = algo.evolve(pop, problem.test_feasible, weighting_func=weighting)

    assert len(pop) == 2
    for i in pop.get_x():
        assert problem.test_feasible(i)

def test_printdict2file(example_MO0_noisy_GA, test_problem, test_population1, test_population2) -> None:
    problem = test_problem
    example_MO0_noisy_GA.gen = 10
    algo = NGA(example_MO0_noisy_GA)
    pop = pg.population(problem, 0)

    pop.push_back(test_population1)
    pop.push_back(test_population2)
    weighting = lambda x: 1/(x)
    pop = algo.evolve(pop, problem.test_feasible, weighting_func=weighting)

    algo.printdict2file('output/tests/test_file.csv')

def test_eq_42_rearrange() -> None:
    A = 5
    X = 3
    std_A = 0.5
    std_X = 0.2
    m = (A - X)/std_X
    s = std_A/std_X
    prob_1 = 1/(1 + np.exp(-2.5*m/np.sqrt(2+2*s**2)))

    prob_2 = 1 - 1/(1 + np.exp(-2.5*(X - A)/np.sqrt(2*std_X**2 + 2*std_A**2)))

    assert np.isclose(prob_1, prob_2), f"prob_1: {prob_1}, prob_2: {prob_2}"

    

