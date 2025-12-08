import pytest
import numpy as np
from example_problem import ExampleProblem

import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from lib.noisy_GA.design_dict import design_history_entry

@pytest.fixture
def design_history_entry_fixture():
    return design_history_entry(objective_list = np.array([1.0]),
                                feasible_prob_list = np.array([np.log(0.5)])    
                            )

def test_update_point_feasible(design_history_entry_fixture) -> None:
    design_history_entry_fixture.update_feasible(2.0, np.log(0.25))
    assert np.array_equal(design_history_entry_fixture.objective_list, np.array([1.0, 2.0]))
    assert np.array_equal(design_history_entry_fixture.feasible_prob_list, np.array([np.log(0.5), np.log(0.25)]))
    assert np.isclose(design_history_entry_fixture.expected_value, 4/3)

def test_update_point_infeasible(design_history_entry_fixture) -> None:
    design_history_entry_fixture.update_infeasible(np.log(0.25))
    assert np.array_equal(design_history_entry_fixture.infeasible_prob_list, np.array([np.log(0.25)]))
    assert np.isclose(design_history_entry_fixture.expected_infeasible_prob, 1/3)

def test_add_to_prop_map(design_history_entry_fixture) -> None:
    design_history_entry_fixture.add_to_prop_map((0, 1))
    assert np.array_equal(design_history_entry_fixture.prop_map, {(0,1)})

def test_check_base_point_eval(design_history_entry_fixture) -> None:
    assert design_history_entry_fixture.check_base_point_eval(np.log(0.5))
