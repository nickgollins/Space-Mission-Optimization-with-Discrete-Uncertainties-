import pytest
import numpy as np

import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from lib.campaign_planning.campaign_planner import CampaignPlanner, CampaignPlannerParameters
from lib.campaign_planning.lunar_logistics import LunarLogMissionParameters, LunarScheduling
from lib.campaign_planning.LET import SYNODIC_PERIOD

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

def test_campaign_planner_init(example_planner) -> None:
    assert example_planner.campaign_reqs == [
                                            [5., 14., 0., 4., 0., 12., '', '', ''],
                                            [5., 430., 0., 4., 13., 24., '', '', ''],
                                        ], f"Unexpected campaign_reqs: {example_planner.campaign_reqs}"
    expected_vehicle_data = {
                            'name': ['Astrobotic Griffin', 'Intuitive Machines Nova-C'],
                            'payload_cap': [625., 100., ],
                            'prop_cap': [3323.012662, 1012.472211],
                            'dry_mass': [1951.987338, 787.5277891],
                            'Isp': [340., 370.],
                            'prop_type': ['Storable', 'Methalox'],
                            'mixture_ratio': [2.61, 3.6],
                            'oxy_boil_off_rate': [0., 2.050735337013889e-05],
                            'fuel_boil_off_rate': [0., 6.562353078444445e-05],
                            'oxy_ratio': [0.7229916897506925, 0.782608695652174],
                            'launch_frequency': [12.,6.],
                            'available_from': [24., 0.],
                            'launch_cost': [0., 0.],
                            'domain': [[[0,0], [0,3], [3,4], [3,3], [4,4], [0,5], [5,3]],
                                    [[0,0], [0,3], [3,4], [3,3], [4,4], [0,5], [5,3] ]],
                            }
    
    for attr, expected in expected_vehicle_data.items():
        assert getattr(example_planner.spacecraft, attr) == expected

    assert example_planner.number_decision_variables == 2
    assert example_planner.get_nix() == 2
    assert example_planner.get_nec() == 0
    assert example_planner.get_nic() == 0

def test_get_feasible_x(example_planner) -> None:
    x = example_planner.get_feasible_x()
    assert x[0] >= 0 and x[0] <= 12
    assert x[1] >= 13 and x[1] <= 25

def test_build_timeline(example_planner) -> None:
    test_x = [0, 13]
    sorted_unique_timeline, MILP_timeline, real_time = example_planner.build_timeline(test_x)

    assert np.array_equal(sorted_unique_timeline, [0, 13])
    assert np.array_equal(MILP_timeline, [0, 1, 2, 3])
    assert real_time == [0, SYNODIC_PERIOD*13]

def test_demand_matrix(example_planner) -> None: 
    test_x = [0, 13]
    sorted_unique_timeline, MILP_timeline, real_time = example_planner.build_timeline(test_x)

    test_restricted_parameters = LunarLogMissionParameters(campaign_data_file='data/CampaignPlanningInputs/test_inputs/campaign_requirements_test.csv',
                                                        vehicle_data_file='data/CampaignPlanningInputs/test_inputs/vehicle_data_test.csv',
                                                        metaheuristic_decision=test_x,
                                                        restricted_timeline=MILP_timeline,
                                                        restricted_real_time=real_time
                                                        )
    
    test_restricted_model = LunarScheduling(test_restricted_parameters)
    test_restricted_model.init_demand_matrices()

    test_demand_matrix = np.zeros((5, 8, 4, 2), dtype=np.float32)
    test_demand_matrix[0, 5, 0, 0] = 14
    test_demand_matrix[4, 5, 0, 0] = -14
    test_demand_matrix[0, 5, 2, 0] = 430
    test_demand_matrix[4, 5, 2, 0] = -430
    test_demand_matrix[0, 0, 0, 1] = 1
    test_demand_matrix[0, 0, 2, 1] = 2

    assert np.shape(test_restricted_model.demand_matrix) == np.shape(test_demand_matrix)
    assert test_demand_matrix[0, 5, 0, 0] == test_restricted_model.demand_matrix[0, 5, 0, 0]
    assert test_demand_matrix[4, 5, 0, 0] == test_restricted_model.demand_matrix[4, 5, 0, 0]
    assert test_demand_matrix[4, 5, 0, 0] == test_restricted_model.demand_matrix[4, 5, 0, 0]
    assert test_demand_matrix[0, 5, 2, 0] == test_restricted_model.demand_matrix[0, 5, 2, 0]
    assert test_demand_matrix[4, 5, 2, 0] == test_restricted_model.demand_matrix[4, 5, 2, 0]
    assert test_demand_matrix[0, 0, 0, 1] == test_restricted_model.demand_matrix[0, 0, 0, 1]
    assert test_demand_matrix[0, 0, 2, 1] == test_restricted_model.demand_matrix[0, 0, 2, 1]
    assert test_demand_matrix[0, 0, 0, 0] == test_restricted_model.demand_matrix[0, 0, 0, 0]
    assert test_demand_matrix[0, 0, 2, 0] == test_restricted_model.demand_matrix[0, 0, 2, 0]