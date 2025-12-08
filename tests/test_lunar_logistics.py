import pytest
import numpy as np
import sys, os

from gurobipy import GRB

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from lib.campaign_planning.lunar_logistics import LunarLogMissionParameters, LunarScheduling
from lib.campaign_planning.LET import SYNODIC_PERIOD

@pytest.fixture
def example_parameters() -> LunarLogMissionParameters:
    return LunarLogMissionParameters(campaign_data_file='data/CampaignPlanningInputs/test_inputs/campaign_requirements_test.csv',
                                    vehicle_data_file='data/CampaignPlanningInputs/test_inputs/vehicle_data_test.csv')

@pytest.fixture
def example_metaheuristic_parameters() -> LunarLogMissionParameters:
    return LunarLogMissionParameters(campaign_data_file='data/CampaignPlanningInputs/test_inputs/campaign_requirements_test.csv',
                                    vehicle_data_file='data/CampaignPlanningInputs/test_inputs/vehicle_data_test.csv',
                                    metaheuristic_decision=[0, 24],
                                    restricted_timeline=[0, 1, 2, 3],
                                    restricted_real_time=[0, 24*SYNODIC_PERIOD])

@pytest.fixture
def example_lunar_scheduling_model(example_parameters) -> LunarScheduling:
    return LunarScheduling(mission_params=example_parameters)

@pytest.fixture
def example_metaheuristic_lunar_scheduling_model(example_metaheuristic_parameters) -> LunarScheduling:
    return LunarScheduling(mission_params=example_metaheuristic_parameters)


def test_lunar_scheduling_init(example_lunar_scheduling_model) -> None:
    assert example_lunar_scheduling_model.campaign_reqs == [
                                                            [5., 14., 0., 4., 0., 12., '', '', ''],
                                                            [5., 430., 0., 4., 13., 24., '', '', '']
                                                        ]
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
                            'domain': [[[0,0], [0,3], [3,4], [3,3], [4,4], [0,5], [5,3] ],
                                        [[0,0], [0,3], [3,4], [3,3], [4,4], [0,5], [5,3] ]],
                            }
    
    for attr, expected in expected_vehicle_data.items():
        assert getattr(example_lunar_scheduling_model.spacecraft, attr) == expected, f"Failed on {attr}"

    expected_model_data = {
                            'stacks': [],
                            'network_input': [
                                                [[0, 0], [0, 0], [0, 0, 0, 0, 0, 0], 1, [2, 0, 1, 0]],
                                                [[0, 1], [1, 100], [1, 1, 1, 1, 1, 1], 1, [0, 0, 0, 6.3]],
                                                [[0, 2], [1, 100], [1, 1, 1, 1, 1, 1], 1, [0, 5, 0, 0.42]],
                                                [[0, 3], [1, 100], [1, 1, 1, 1, 1, 1], 1, [0, 3, 0, 0.89]],
                                                [[0, 4], [0, 0], [0, 0, 0, 0, 0, 0], 0, [0, 0, 0, 0]],

                                                [[1, 0], [0, 0], [0, 0, 0, 0, 0, 0], 0, [1, 0, 0, 0]],
                                                [[1, 1], [0, 0], [0, 0, 0, 0, 0, 0], 1, [2, 16, 1, 0]],
                                                [[1, 2], [0, 0], [0, 0, 0, 0, 0, 0], 1, [0, 5, 0, 3.15]],
                                                [[1, 3], [0, 0], [0, 0, 0, 0, 0, 0], 1, [0, 3, 0, 3.15]],
                                                [[1, 4], [0, 0], [0, 0, 0, 0, 0, 0], 0, [0, 0, 0, 0]],

                                                [[2, 0], [0, 0], [0, 0, 0, 0, 0, 0], 1, [1, 5, 0, 0.42]],
                                                [[2, 1], [0, 0], [0, 0, 0, 0, 0, 0], 0, [1, 0, 0, 0]],
                                                [[2, 2], [0, 0], [0, 0, 0, 0, 0, 0], 1, [2, 6, 1, 0]],
                                                [[2, 3], [0, 0], [0, 0, 0, 0, 0, 0], 1, [0, 0.5, 0, 0.7]],
                                                [[2, 4], [0, 0], [0, 0, 0, 0, 0, 0], 0, [0, 0, 0, 0]],

                                                [[3, 0], [0, 0], [0, 0, 0, 0, 0, 0], 1, [1, 3, 0, 0.89]],
                                                [[3, 1], [0, 0], [0, 0, 0, 0, 0, 0], 0, [1, 0, 0, 0]],
                                                [[3, 2], [0, 0], [0, 0, 0, 0, 0, 0], 1, [1, 0.5, 0, 0.7]],
                                                [[3, 3], [0, 0], [0, 0, 0, 0, 0, 0], 1, [2, 5, 1, 0.15]],
                                                [[3, 4], [0, 0], [0, 0, 0, 0, 0, 0], 1, [0, 1, 0, 1.87]],

                                                [[4, 0], [0, 0], [0, 0, 0, 0, 0, 0], 0, [1, 0, 0, 0]],
                                                [[4, 1], [0, 0], [0, 0, 0, 0, 0, 0], 0, [1, 0, 0, 0]],
                                                [[4, 2], [0, 0], [0, 0, 0, 0, 0, 0], 0, [1, 0, 0, 0]],
                                                [[4, 3], [0, 0], [0, 0, 0, 0, 0, 0], 1, [1, 1, 0, 1.87]],
                                                [[4, 4], [0, 0], [0, 0, 0, 0, 0, 0], 1, [2, 3, 1, 0]],
                                            ],
                            'number_nodes': 5,
                            'N': range(2),
                            'T': range(50)
                            }

    for attr, expected in expected_model_data.items():
        assert getattr(example_lunar_scheduling_model, attr) == expected, f"Failed on {attr, expected}"

def test_build_primal_binary(example_lunar_scheduling_model) -> None:
    model = example_lunar_scheduling_model.build_primal_model(method='binary')
    model.optimize()
    assert model.Status == GRB.OPTIMAL
    assert model.objVal != 0

def test_build_primal_metaheuristic(example_metaheuristic_lunar_scheduling_model) -> None:
    model = example_metaheuristic_lunar_scheduling_model.build_primal_model(method='metaheuristic')
    model.optimize()
    assert model.Status == GRB.OPTIMAL
    assert model.objVal != 0
    assert sum(model.getVarByName("xF[%d,3,4,3,0,0]" % (n)).X for n in range(2)) == 14
    assert sum(model.getVarByName("xF[%d,3,4,3,1,2]" % (n)).X for n in range(2)) >= 430, f"Second payload not on (3,4) arc at time 2: {sum(model.getVarByName('xF[%d,3,4,3,1,2]' % (n)).X for n in range(2))}"

def test_model_comparison(example_lunar_scheduling_model, example_metaheuristic_lunar_scheduling_model) -> None:
    discrete_model = example_lunar_scheduling_model.build_primal_model(method='binary')
    metaheuristic_model = example_metaheuristic_lunar_scheduling_model.build_primal_model(method='metaheuristic')

    discrete_model.optimize()
    metaheuristic_model.optimize()

    assert np.isclose(discrete_model.objVal, metaheuristic_model.objVal)
