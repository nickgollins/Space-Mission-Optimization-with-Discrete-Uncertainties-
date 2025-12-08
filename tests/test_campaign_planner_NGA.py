import pytest
import numpy as np
import pygmo as pg

import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from lib.campaign_planning.campaign_planner import CampaignPlanner, CampaignPlannerParameters
from lib.campaign_planning.lunar_logistics import LunarLogMissionParameters, LunarScheduling
from lib.campaign_planning.LET import SYNODIC_PERIOD

from lib.noisy_GA.noisy_GA import noisy_GA, NGA

@pytest.fixture
def example_mission_parameters() -> LunarLogMissionParameters:
    return LunarLogMissionParameters(campaign_data_file='data/CampaignPlanningInputs/test_inputs/campaign_requirements_test.csv',
                                    vehicle_data_file='data/CampaignPlanningInputs/test_inputs/vehicle_data_test.csv'
                                    )

@pytest.fixture
def example_planner_parameters() -> CampaignPlannerParameters:
    return CampaignPlannerParameters(MILP_framework='gurobi', MIPgap=0.05)

@pytest.fixture
def example_planner(example_mission_parameters, example_planner_parameters) -> CampaignPlanner:
    return CampaignPlanner(mission_params=example_mission_parameters, solver_params=example_planner_parameters)

@pytest.fixture
def example_MO0_noisy_GA() -> noisy_GA:
    return noisy_GA(noise=np.array([[0.5, 0.3, 0.2] for _ in range(2)]),
                    gen = 30,
                    noise_samples_per_point=8,
                    point_samples_per_gen=10,
                    MOO=True,
                    prob_mut=0,
                    prob_cross=0
                    )

@pytest.fixture
def example_NGA(example_MO0_noisy_GA) -> NGA:
    return NGA(noisy_GA=example_MO0_noisy_GA)

def test_campaign(example_planner, example_NGA) -> None:
    problem = example_planner
    algo = example_NGA
    pop = pg.population(problem, 0)
    
    for _ in range(10):
        pop.push_back(example_planner.get_feasible_x())

    pop = algo.evolve(pop, problem.test_feasible, weighting_func=lambda x: 1/x, keep_champion=False)

    assert len(pop) == 10
    for i in pop.get_x():
        assert problem.test_feasible(i)

    champion = algo.find_champ()
    assert np.isclose(champion.expected_value, 7168.69, rtol=1e-2), f"Expected value: {champion.expected_value}"          
    assert np.isclose(champion.expected_infeasible_prob, 0.5), f"Expected infeasible probability: {champion.expected_infeasible_prob}"