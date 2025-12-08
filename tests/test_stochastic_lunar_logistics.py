import pytest
import numpy as np
import sys, os

from gurobipy import GRB

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from lib.campaign_planning.stochastic_lunar_logistics import StochasticLunarLogMissionParameters, StochasticLunarScheduling

@pytest.fixture
def example_parameters() -> StochasticLunarLogMissionParameters:
    return StochasticLunarLogMissionParameters(campaign_data_file='data/CampaignPlanningInputs/test_inputs/campaign_requirements_test.csv',
                                            vehicle_data_file='data/CampaignPlanningInputs/test_inputs/vehicle_data_test.csv',
                                            noise=np.array([[0.7, 0.3] for _ in range(2)]))

@pytest.fixture
def example_lunar_scheduling_model(example_parameters) -> StochasticLunarScheduling:
    return StochasticLunarScheduling(mission_params=example_parameters)

def test_stochastic_lunar_scheduling(example_lunar_scheduling_model) -> None:
    # assert np.array_equal(example_lunar_scheduling_model.combinations, [[0,0], [0,1], [0,2],
    #                                                                     [1,0], [1,1], [1,2],
    #                                                                     [2,0], [2,1], [2,2]])
    model = example_lunar_scheduling_model.build_primal_model()
    model.optimize()
    assert model.status == GRB.OPTIMAL    
    assert sum(v.x for v in model.getVars() if 'alpha' in v.varName) == 1
    