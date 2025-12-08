import numpy as np
from gurobipy import GRB
import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from lib.campaign_planning.stochastic_lunar_logistics import StochasticLunarLogMissionParameters, StochasticLunarScheduling
from lib.campaign_planning.lunar_logistics import LunarScheduling, LunarLogMissionParameters
from plotting import plot_networks

if __name__ == '__main__':
    MIQP_parameters = StochasticLunarLogMissionParameters(
                            campaign_data_file="data/CampaignPlanningInputs/campaign_requirements_MIQP.csv",
                            vehicle_data_file="data/CampaignPlanningInputs/vehicle_data_MIQP.csv",
                            noise=np.array([[0.7, 0.3] for _ in range(6)]),
                        )
    
    stochastic_lunar_scheduling_model = StochasticLunarScheduling(mission_params=MIQP_parameters)

    model = stochastic_lunar_scheduling_model.build_primal_model()

    # Set parameters to store multiple solutions
    model.setParam(GRB.Param.PoolSolutions, 100)  # Store up to 100 solutions
    model.setParam('LogFile', 'output/MIQCP_case_study/MIQCP_log_Mar3.txt')
    model.setParam(GRB.Param.TimeLimit, 31000) 

    model.optimize()

    # Improve bounds
    model.setParam(GRB.Param.MIPFocus, 2)  # Focus on improving the best bound
    model.optimize()

    # Retrieve and print the objective values of all solutions in the pool
    solutions = np.zeros((model.SolCount, 2))
    for i in range(model.SolCount):
        model.params.SolutionNumber = i
        model.setParam(GRB.Param.ObjNumber, 0)
        solutions[i, 0] = model.ObjNVal
        model.setParam(GRB.Param.ObjNumber, 1)
        solutions[i, 1] = model.ObjNVal
        print('Solution', i, 'Obj 0:', solutions[i, 0], 'Obj 1:', solutions[i, 1])

    np.savetxt('output/MIQCP_case_study/MIQCP_results_Mar3.txt', solutions, delimiter=',')