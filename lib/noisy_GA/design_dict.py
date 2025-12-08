import numpy as np
from scipy.special import logsumexp
from scipy.stats import t
from typing import Optional, List
from dataclasses import dataclass, field

@dataclass
class design_history_entry:
    objective_list: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    feasible_prob_list: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    infeasible_prob_list: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    point:Optional[tuple[int]] = None   

    base_objective:Optional[float]=None

    prop_map:Optional[set[tuple[int]]] = field(default_factory=set)

    expected_value:Optional[np.float64] = None
    expected_value_std_error:Optional[np.float64] = None

    expected_infeasible_prob:Optional[np.float64] = None
    expected_infeasible_prob_std_error:Optional[np.float64] = None


    def __post_init__(self):
        self.update_expected_value()
        return
    
    def __str__(self) -> str:
        return str(self.point)
    
    def check_attr(self) -> list:
        return [value for attr, value in vars(self).items()]

    def update_feasible(self, 
                        objective:float, 
                        feas_prob:float
                    ) -> None:
        self.objective_list = np.append(self.objective_list, objective)
        self.feasible_prob_list = np.append(self.feasible_prob_list, feas_prob)       
        self.update_expected_value()
        self.calc_exp_prob_inf()
        self.calc_std_errors()
        return
    
    def update_infeasible(self, infeas_prob) -> None:
        self.infeasible_prob_list = np.append(self.infeasible_prob_list, infeas_prob)
        self.calc_exp_prob_inf()
        self.calc_std_errors()
        return
    
    def update_expected_value(self) -> None:
        assert len(self.objective_list) == len(self.feasible_prob_list), "Length of objective_list and feasible_prob_list must be the same"
        
        if len(self.objective_list) == 0:
            self.expected_value = None
        elif len(self.objective_list) == 1:
            self.expected_value = self.objective_list[0]
        else:
            self.expected_value = np.exp(logsumexp([np.log(obj) + self.feasible_prob_list[i] for i, obj in enumerate(self.objective_list)]) -
                                        logsumexp(self.feasible_prob_list))
        return
    
    def calc_exp_prob_inf(self) -> None:
        if len(self.feasible_prob_list) == 0 or len(self.infeasible_prob_list) == 0:
            self.expected_infeasible_prob = None
        else:
            self.expected_infeasible_prob = sum(np.exp(self.infeasible_prob_list)) / (sum(np.exp(self.infeasible_prob_list)) + sum(np.exp(self.feasible_prob_list)))

        return 
    
    def add_to_prop_map(self, 
                        propagation_point:tuple[int]
                    ) -> None:
        """Add a propgation pairing to the propagation mapping dictionary

        Inputs:
            propagation_point: Propagation design point        
        """
        assert propagation_point not in self.prop_map, f"{propagation_point} already exists in {self} propagation mapping dictionary"
        self.prop_map.add((propagation_point))
        assert propagation_point in self.prop_map, f"{propagation_point} not added to {self} propagation mapping dictionary"
        return
    
    def check_base_point_eval(self, base_point_prob) -> bool:
        """Check if the base point has been evaluated

        Returns:
            bool: True if the base point has been evaluated
        """
        return base_point_prob in self.feasible_prob_list or base_point_prob in self.infeasible_prob_list
    
    def calc_std_errors(self) -> None:
        """Calculate the standard errors for the expected value and expected infeasible probability
        """
        if len(self.objective_list) <= 1:
            self.expected_value_std_error = None
        else:
            assert self.expected_value is not None, "Expected value must be calculated before standard error"
            self.expected_value_std_error = np.sqrt(sum([((obj - self.expected_value) ** 2) * np.exp(self.feasible_prob_list[i]) for i, obj in enumerate(self.objective_list)]) /
                                                    sum(np.exp(self.feasible_prob_list)) / (len(self.objective_list) - 1))

        if len(self.infeasible_prob_list) == 0 or len(self.feasible_prob_list) == 0:
            self.expected_infeasible_prob_std_error = None
        else:
            assert self.expected_infeasible_prob is not None, "Expected infeasible probability must be calculated before standard error"
            self.expected_infeasible_prob_std_error = np.sqrt(sum([((np.exp(inf_prob) - self.expected_infeasible_prob) ** 2) * np.exp(inf_prob) 
                                                                   for inf_prob in self.infeasible_prob_list]) /
                                                            sum(np.exp(self.infeasible_prob_list)) / (len(self.infeasible_prob_list) + len(self.feasible_prob_list)- 1))
        return

    
    
    
