# Noisy-GA-Scheduling
 
This project is about large (many mission) exploration campaign schedule optimization with launch uncertainty. The aim of the project is to produce the Pareto front for the trade-off between performance and robustness for the campaign schedule decisions. It consists of 4 main components:

1. A multi-objective stochastic mixed-integer quadratically constrained network flow problem formulation
    -  problem code in lib\campaign_planning\stochastic_lunar_logistics.py 
    -  executed by src\run_MIQCP.py
2. A framework for constructing a restricted, deterministic network flow problem based on a set of integer scheduling variables
    - lib\campaign_planning\campaign_planner.py
    - see https://doi.org/10.2514/1.A35828 for a description
3. The deterministic network flow problem constructed by the above hierarchical framework
    - lib\campaign_planning\lunar_logistics.py
4. A multi-objective noisy integer evolutionary algorithm for optimising the schedule decisions made in 2., using the objective returned by 3. as one objective. The other objective is the expected probability of infeasibility.
    - lib\noisy_GA\noisy_GA.py
    - executed by src\run_NGA.py

A full description of all of the above is available in the following reference [1]

[1] Gollins, N; Grieser, Z; Ho, K, Multi-Objective Optimization of Space Exploration Campaign Schedules with Stochastic Launch Delay, Journal of Spacecraft and Rockets, 2026, https://doi.org/10.2514/1.A36412

## Installation
```
conda create -n <env-name>
conda activate <env-name>
conda install poetry
conda install pygmo
poetry env use <path to conda env python.exe>
poetry install 
```

## Testing
Run all tests using ```pytest```. 

## Inputs
The exploration campaign to be optimized should be defined in two .csv files: one which defines the set of campaign payloads; and one which defines the set of available logistics vehicles. These files should be placed in the data folder. Some examples are provided in data/CampaignPlanningsInputs.

### Payload definition file
The payload definition file defines one payload per row, with the first row containing the column headers. Do not change the column headers - the scripts that process the inputs expect to find specific data in each column. Detailed descriptions of the columns can be found in <TODO: add DOI for paper once published>.

### Vehicle definition file
The vehicle definition file defines one payload per row, with the first row containing the column headers. Do not change the column headers - the scripts that process the inputs expect to find specific data in each column. Detailed descriptions of the columns can be found in <TODO: add DOI for paper once published>.

### Network definition file
The data input folder also contains a network definition file. The current formulation of this code library is not robust to user-edits to the network definition. Please see <TODO: add reference to open space logistics code> for this functionality.

## Running the methods
Some example scripts for running the NMOEA and the MIQP are contained in the src/ folder. The structures of the scripts are as follows:

### MIQP
- Set up the MIQP inputs parameters, including the payload and vehicle data files, and the desired noise to be applied to the launch time of each payload. The noise should be defined as a 2-dimensional list, where each $(i,j)$ entry contains the probability of payload $i$ being delayed by $j$ discrete time steps. For example:

```p
MIQP_parameters = StochasticLunarLogMissionParameters(
                            campaign_data_file="data/CampaignPlanningInputs/campaign_requirements_MIQP.csv",
                            vehicle_data_file="data/CampaignPlanningInputs/vehicle_data_MIQP.csv",
                            noise=np.array([[0.7, 0.3] for _ in range(6)]),
                        )
    
stochastic_lunar_scheduling_model = StochasticLunarScheduling(mission_params=MIQP_parameters)
```

- Next, we build the model and solve by calling the StochasticLunarScheduling objects's ```.build_primal_model``` method, which returns a ```gurobipy``` model. The model is solved by calling ```model.optimize()```:

```p
model = stochastic_lunar_scheduling_model.build_primal_model()
model.optimize()
```

### NMOEA
- Similar to the MIQP, we begin by setting up the problem structures. The ```CampaignPlanner``` object does this for us, taking the mission solver parameters as inputs.

```p
CLPS_parameters = LunarLogMissionParameters(
                            campaign_data_file="data/CampaignPlanningInputs/campaign_requirements_Artemis.csv",
                            vehicle_data_file="data/CampaignPlanningInputs/vehicle_data.csv"
                        )

planner_parameters = CampaignPlannerParameters(MILP_framework='gurobi', MIPgap=0.03)

problem = CampaignPlanner(mission_params=CLPS_parameters, 
                          solver_params=planner_parameters,
                          timing=True)
```

- Next, we set up the NMOEA's parameters:

```p
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
```

- The NMOEA uses the ```pygmo``` library for population management. Therefore, we next set up the population. The ```CampaignPlanner``` object has a ```get_feasible_x()``` method that finds a pseudo-random feasible solution. We use this method to initialize the population with feasible candidates:

```p
pop = pg.population(problem, 0)
    
for _ in range(50):
    pop.push_back(problem.get_feasible_x())
```

- Next, we define the probability weighting to be used when selecting new populations for each generation from the search history. To do this, we use a ```lambda``` function, which takes the product of a candidate solutions objectives as input and returns the weighting for selection. For example:

```p
weighting = lambda x: 1/x
```

inversely weights the selection of a solution by its objective, ideal for minimization problems (lowest objective is most likely to be selected).

- Finally, we begin the optimization process by calling the NMOEA's ```.evolve()``` method:

```p
pop = algo.evolve(pop, 
                 feasible_check=problem.test_feasible, 
                 keep_champion=True,
                 weighting_func=weighting,
                 output_file="output/case_studies/Artemis_correct_variance.csv")
```
# Acknowledgment
This material is based upon work partially supported by the National Science Foundation under Award No. 1942559. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
  


