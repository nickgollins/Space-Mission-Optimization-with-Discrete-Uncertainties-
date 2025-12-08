import numpy as np
import pandas as pd
import pygmo as pg
import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from lib.campaign_planning.campaign_planner import CampaignPlanner, CampaignPlannerParameters
from lib.campaign_planning.lunar_logistics import LunarScheduling, LunarLogMissionParameters
from lib.noisy_GA.noisy_GA import noisy_GA, NGA

if __name__ == '__main__':
    CLPS_parameters = LunarLogMissionParameters(
                            campaign_data_file="data/CampaignPlanningInputs/campaign_requirements_Artemis.csv",
                            vehicle_data_file="data/CampaignPlanningInputs/vehicle_data.csv"
                        )
    
    # stochastic_lunar_scheduling_model = LunarScheduling(mission_params=CLPS_parameters)

    planner_parameters = CampaignPlannerParameters(MILP_framework='gurobi', MIPgap=0.03)

    problem = CampaignPlanner(mission_params=CLPS_parameters, 
                              solver_params=planner_parameters,
                              timing=True)

    NGA_parameters = noisy_GA(noise=np.array([[0.7, 0.3] for _ in range(19)]),
                            gen=100,
                            num_backprop_per_point=5000,
                            noise_samples_per_point=10,
                            point_samples_per_gen=15,
                            end_noise_samples_per_point=200,
                            MOO=True,
                            prob_mut=0.01,
                            prob_cross=0.01,
                            frac_cores=1/4
                        )

    algo = NGA(noisy_GA=NGA_parameters)

    pop = pg.population(problem, 0)
    
    for _ in range(50):
        pop.push_back(problem.get_feasible_x())
        
    # for x in [[1, 11, 5, 11, 16, 22, 26, 22, 25, 25, 40, 44, 47, 47, 66, 72, 73, 73, 73],
    #         #   [0, 5, 15, 15, 16, 15, 22, 24, 32, 32, 47, 51, 51, 51, 65, 70, 72, 75, 75],
    #         #   [2, 4, 14, 3, 13, 27, 28, 26, 34, 34, 33, 51, 51, 51, 70, 73, 65, 75, 75],
    #         #   [5, 3, 11, 3, 14, 19, 20, 15, 28, 28, 35, 44, 45, 45, 74, 74, 66, 74, 74],
    #         #   [5, 14, 15, 15, 12, 23, 24, 15, 33, 33, 40, 40, 43, 43, 75, 75, 70, 75, 75],
    #         #   [2, 11, 12, 15, 13, 26, 27, 17, 26, 26, 43, 46, 51, 51, 65, 73, 64, 75, 75],
    #         #   [4, 7, 5, 4, 18, 24, 27, 19, 25, 25, 35, 42, 49, 49, 69, 72, 68, 72, 72],
    #         #   [0, 4, 13, 7, 12, 25, 26, 27, 25, 25, 29, 48, 50, 50, 67, 68, 64, 70, 70],
    #         #   [4, 8, 4, 15, 13, 27, 28, 27, 33, 33, 30, 39, 43, 43, 73, 73, 75, 75, 75],
    #           [2, 7, 8, 14, 16, 20, 27, 19, 36, 36, 32, 40, 42, 42, 70, 71, 70, 75, 75]]:
    #     pop.push_back(x)

    weighting = lambda x: 1/x

    pop = algo.evolve(pop, 
                      feasible_check=problem.test_feasible, 
                      keep_champion=True,
                      weighting_func=weighting,
                      output_file="output/case_studies/Artemis_correct_variance.csv")

    champion = algo.find_champ()
    print(champion)
