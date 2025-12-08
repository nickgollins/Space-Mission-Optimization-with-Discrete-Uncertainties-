from .lunar_logistics import LunarLogMissionParameters, LunarScheduling, get_network_size
from .vehicle_model import VehicleModel, get_vehicle_data
from .LET import load_LETs, SYNODIC_PERIOD
from .misc import import_csv_file, import_json_file, process_campaign_reqs
import sys, os
import numpy as np
import random
import time
from gurobipy import GRB, Model
from dataclasses import dataclass
from pyomo.environ import value
from pyomo.opt import SolverFactory, TerminationCondition
from copy import copy
from typing import Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

@dataclass
class CampaignPlannerParameters:
    MILP_framework:str='gurobi'
    MIPgap:float=0.01
    verbose:bool=False

    def __post_init__(self):
        assert self.MILP_framework in ['pyomo', 'gurobi']
        assert self.MIPgap >= 0.0


class CampaignPlanner:
    def __init__(self, 
                 mission_params: LunarLogMissionParameters,
                 solver_params: CampaignPlannerParameters,
                 timing:bool=False
                ):
        self.mission_params = mission_params
        # Load programmatic stuff
        self.campaign_timeline = import_csv_file(mission_params.campaign_data_file)
        self.campaign_reqs = process_campaign_reqs(self.campaign_timeline)
        self.timing = timing

        # Load vehicle stuff
        self.vehicle_data = import_csv_file(mission_params.vehicle_data_file)
        self.spacecraft = get_vehicle_data(self.vehicle_data)
        print('Vehicle data:')
        for i in range(len(self.spacecraft.name)):
            print('Vehicle {}:'.format(i), self.spacecraft.name[i], self.spacecraft.prop_cap[i], self.spacecraft.dry_mass[i])
        self.stacks = []
        if mission_params.stacking_definition_file != '':
            self.stacks = import_json_file(mission_params.stacking_definition_file)
            self.spacecraft.getStacks(self.stacks)
            print('Stack data:')
            for i in range(len(self.stacks)):
                print('Stack {}:'.format(i), self.stacks[i], [self.spacecraft.name[j] for j in self.stacks[i]])
        self.number_vehicles = np.size(self.spacecraft.prop_cap)
        self.number_non_stack_vehicles = self.number_vehicles - len(self.stacks)

        # Load logistics network stuff
        self.network_data = import_json_file(mission_params.network_data_file)
        self.number_nodes = get_network_size(self.network_data)
        if mission_params.LET:
            self.LET_network = load_LETs(spacecraft=self.spacecraft)
        else:
            self.LET_network = False

        # Initialize pygmo stuff
        self.number_decision_variables = len(self.campaign_reqs)

        self.ISRU = mission_params.ISRU

        self.MIPgap = solver_params.MIPgap
        self.MILP_framework = solver_params.MILP_framework

        self.verbose = solver_params.verbose

        return

    def fitness(self, x):
        ceqs, cineqs = self.get_constrs(x)
        if any([i != 0 for i in ceqs]) or any([i > 0 for i in cineqs]):
            obj = [1E15]
        else:
            obj = self.evaluate(x)
        return obj + ceqs + cineqs

    def get_bounds(self):
        lx = []
        ux = []
        for row in self.campaign_reqs:  # Each row is a payload
            lx.append(int(row[4]))  # Time window start
            ux.append(int(row[5]))  # Time window end
        return lx, ux

    def get_nix(self):
        return self.number_decision_variables

    def get_nec(self):
        self.nec = 0
        for row in self.campaign_reqs:
            if row[8] != '':  # Entry 6 contains necessary precursor payloads
                for co_payload in [int(el) for el in row[8].split(',') if el not in ['', ' ']]:  
                    try:
                        int(co_payload)
                        self.nec += 1
                    except:
                        continue
        print('Number of equality constraints = ', self.nec)
        return self.nec

    def get_nic(self):
        self.nic = 0
        for row in self.campaign_reqs:
            # self.nic += 1  # Every payload at least has a vehicle availability constraint
            if row[6] != '':  # Entry 6 contains necessary precursor payloads
                for precursor_payload in [int(el) for el in row[6].split(',') if el not in ['', ' ']]:
                    try:
                        int(precursor_payload)
                        self.nic += 1
                    except:
                        continue
            if row[7] != '':
                for precursor_payload in [int(el) for el in row[7].split(',') if el not in ['', ' ']]:
                    try:
                        int(precursor_payload)
                        self.nic += 1
                    except:
                        continue

        print('Number of inequality constraints = ', self.nic)
        return self.nic

    def pretty(self, 
               x:list[int], 
               output:str='output/logistics_network_output.xlsx',
            ):  # Print results in nice way
        tstart = time.time()
        model = self.evaluate(x, output_file=output)
        tend = time.time()
        print('Model solved. Time taken: ', tend - tstart)
        return model

    def evaluate(self, 
                 x: list[int], 
                 output_file:Optional[str]=None):
        x = [int(el) for el in x]
        # Build timelines
        if self.LET_network is not False:
            sorted_unique_timeline, MILP_timeline, real_time, max_tof = self.build_timeline(x, LET_network=self.LET_network)
        else:
            sorted_unique_timeline, MILP_timeline, real_time = self.build_timeline(x, LET_network=self.LET_network)

        if self.MILP_framework == 'pyomo':
            raise NotImplementedError
        elif self.MILP_framework == 'gurobi':
            if self.timing: solve_time = time.time()
            local_scheduling_params = LunarLogMissionParameters(campaign_data_file=self.mission_params.campaign_data_file,
                                                                vehicle_data_file=self.mission_params.vehicle_data_file,
                                                                network_data_file=self.mission_params.network_data_file,
                                                                stacking_definition_file=self.mission_params.stacking_definition_file,
                                                                LET=self.mission_params.LET,
                                                                ISRU=self.mission_params.ISRU,
                                                                output=self.mission_params.output,
                                                                metaheuristic_decision=x,
                                                                restricted_timeline=MILP_timeline,
                                                                restricted_real_time=real_time
                                                                )

            scheduler = LunarScheduling(local_scheduling_params)
            model = scheduler.build_primal_model(method='metaheuristic')
            if output_file is None:
                if self.verbose is False:
                    model.setParam('OutputFlag', 0)
                model.optimize()
                if model.status == GRB.OPTIMAL:    
                    obj = model.objVal
                else:
                    obj = 1E15
                if self.timing: print(f"Solve time with x = {x} is {time.time() - solve_time} with status {model.status}")
                return [obj]
        
            else:
                model = scheduler.build_primal_model(method='metaheuristic')
                model.optimize()
                if model.status == GRB.OPTIMAL:
                    scheduler.model_results_to_excel(model, output_file)
                return model, scheduler.xI_index, scheduler.xF_index

    def get_constrs(self, x):
        # Constraint stuff (get ceqs, cineqs)
        # Sequencing constraints (payload x must launch before payload y)
        cineqs = []
        ceqs = []
        i = 0
        for row in self.campaign_reqs:
            if row[6] not in ['', ' ']:  # Entry 6 contains necessary precursor payloads with soft inequality
                for precursor_payload in [int(el) for el in row[6].split(',')]:
                    # cineqs.append(x[int(precursor_payload) * 2] - x[i * 2])
                    cineqs.append(x[int(precursor_payload)] - x[i])
            if row[7] not in ['', ' ']:  # Entry 7 contains necessary precursor payloads with strict inequality
                for precursor_payload in [int(el) for el in row[7].split(',')]:
                    # cineqs.append(x[int(precursor_payload) * 2] + 1 - x[i * 2])
                    cineqs.append(x[int(precursor_payload)] + 1 - x[i])
            if row[8] not in ['', ' ']:  # Entry 8 contains necessary co-payloads
                for co_payload in [int(el) for el in row[8].split(',')]:
                    # ceqs.append(x[int(co_payload) * 2] - x[i * 2])
                    ceqs.append(x[int(co_payload)] - x[i])
            i += 1
        return ceqs, cineqs

    def get_feasible_x(self):  # Generates random feasible decision vector
        test_x = [0 for _ in range(len(self.get_bounds()[0]))]  # Initialise decision vector
        test_vehicle = [random.randrange(0, self.number_non_stack_vehicles) for _ in range(len(test_x))]

        for i in range(len(test_x)):
            # Launch time assignment
            # Check for equality constraint
            co_payload_list = self.campaign_reqs[int(i)][8]  # Get co-payloads
            if co_payload_list not in ['', ' ']:
                for j in co_payload_list.split(','):
                    test_x[i] = test_x[int(j)]  # Set necessary co-payloads to launch at the same time
                    test_vehicle[i] = test_vehicle[int(j)]  # Set necessary co-payloads to same vehicle
            else:
                lower_bound = self.get_constrained_time_lower_bound(i, test_x)
                upper_bound = self.get_bounds()[1][i] + 1
                if lower_bound == upper_bound:
                    test_x[i] = lower_bound
                elif lower_bound > upper_bound:  # If no feasible time found, kill this attempt
                    test_x = [0 for _ in test_x]
                    return test_x
                else:
                    test_x[i] = random.randrange(lower_bound, upper_bound)


                # print('Payload ', i, 'Time window: ', test_x[i])
                count = 0
                it_limit = 0
                while count < 1 and it_limit < 1E5:
                    valid_vehicle_list = [n for n in range(self.number_non_stack_vehicles)]
                    remove_vehicles = []
                    if self.campaign_reqs[int(i)][2] == 0:  # Check if launched from Earth
                        for n in range(self.number_vehicles-len(self.stacks)):
                            if self.spacecraft.available_from[n] > test_x[i] or []: # First check if vehicle is available at this time
                                valid_vehicle_list.remove(n)

                        if valid_vehicle_list != []:  # If any vehicles remain, continue
                            for n in valid_vehicle_list:
                                payload_n = 0
                                for j in range(0, i + 1): # Check all payloads up to and including this one
                                    if test_x[j] == test_x[i] and (test_vehicle[j] == n or j == i):  # Sum payloads already assigned to this vehicle at this time
                                        if self.campaign_reqs[j][0] != 1:  # Check if not crew
                                            payload_n += self.campaign_reqs[j][1]
                                        else:
                                            payload_n += self.campaign_reqs[j][1] * 100  # Crew cost is 100

                                if self.spacecraft.payload_cap[n] < payload_n:  # Check if payload is within capacity, remove if not
                                    # print('Removing ', n)
                                    remove_vehicles.append(n)
                        valid_vehicle_list = [n for n in valid_vehicle_list if n not in remove_vehicles]

                        if valid_vehicle_list != []:  # If any vehicles remain, continue
                            remove_vehicles = []
                            for n in valid_vehicle_list:
                                for j in range(0, i + 1):  # Check for other launches on same vehicle with launch frequency constraint
                                    if test_vehicle[j] == n and (abs(test_x[j] - test_x[i]) != 0 and abs(test_x[j] - test_x[i]) < self.spacecraft.launch_frequency[n]):
                                        remove_vehicles.append(n)
                                        break

                        valid_vehicle_list = [n for n in valid_vehicle_list if n not in remove_vehicles]

                    else:  # Check if launched in space, only allow in-space vehicles if so
                        for n in range(self.number_vehicles - len(self.stacks)):
                            if ([3, 2] not in self.spacecraft.domain[n]) or ([2, 0] not in self.spacecraft.domain[n]):
                                valid_vehicle_list.remove(n)
                    if valid_vehicle_list != []:
                        test_vehicle[i] = random.choice(valid_vehicle_list)  # Pick a random suitable vehicle

                        count = 1  # First condition filled
                    else:  # Trying to launch to close to another launch of the all large enough vehicles
                        new_lower_bound = self.get_constrained_time_lower_bound(i, test_x, test_x[i])
                        if new_lower_bound != upper_bound: # Check that there is actually multiple time windows available
                            test_x[i] = random.randrange(new_lower_bound, self.get_bounds()[1][i]+1)
                        else:
                            test_x[i] = new_lower_bound
                    it_limit += 1

        return test_x

    def copayload_check(self, i, x):
        '''
            :param i: Which index of x is being checked
            :param x: Decision vector
            :return: Time stamps of relevant co-payloads
        '''
        co_payload_list = self.campaign_reqs[int(i)][8]  # Get co-payloads
        if co_payload_list != '':
            copayload_times = [x[int(j)] for j in co_payload_list.split(',')]
            if all(i == copayload_times[0] for i in copayload_times):
                return copayload_times[0]
            else:
                print('ERROR: Co-payloads not launched at same time')
                return 0
        else:
            return 0

    def get_constrained_time_lower_bound(self, i, test_x, vehicle_bound=0):
        constraint_bounds = []  # Initialise new constraints
        constraint_bounds.append(self.get_bounds()[0][i])  # Add lower bound
        soft_precursor_list = self.campaign_reqs[i][6]  # Get soft precursors
        strict_precursor_list = self.campaign_reqs[i][7]  # Get strict precursors

        if soft_precursor_list not in ['', ' ']:
            for j in soft_precursor_list.split(','):
                constraint_bounds.append(test_x[int(j)])

        if strict_precursor_list not in ['', ' ']:
            for j in strict_precursor_list.split(','):
                constraint_bounds.append(test_x[int(j)]+1)

        new_lower_bound = max(constraint_bounds)  # Set new lower bound for payload launch based on already-established precursors timestamps
        return new_lower_bound

    def build_timeline(self, x, LET_network=False):
        # Build timelines
        step_length = SYNODIC_PERIOD  # 28.5 day time increments
        timestamp_list = [int(t) for t in x]  # Convert x's to integers
        if LET_network is not False:
            max_LET_tof = max([sum(LET_network[n][arc][4][2] for arc in [0, 1]) for n in range(self.number_vehicles)])
            min_LET_tof = min([sum(LET_network[n][arc][4][2] for arc in [0, 1]) for n in range(self.number_vehicles)])
            # If payload is not crew, and programmatic requirements allow, add preceding time steps to allow LET
            for ind, time in enumerate(timestamp_list):
                if self.campaign_reqs[ind][0] != 1 and time - max_LET_tof/2 >= self.campaign_reqs[ind][4]:
                    if time - max_LET_tof/2 >= 0:  # First check longest tof LET
                        timestamp_list = np.concatenate((timestamp_list, [t for t in range(time - int(max_LET_tof/2), time)]))
                    elif min_LET_tof/2 >= 0:  # Then check shortest tof LET
                        timestamp_list = np.concatenate((timestamp_list, [t for t in range(time - int(min_LET_tof/2), time)]))
            sorted_unique_timeline = np.sort(np.unique(timestamp_list))
            real_time = []
            # Trim timeline
            MILP_timeline = []

            for i in range(len(sorted_unique_timeline)):
                real_time.append(int(sorted_unique_timeline[i]) * step_length)
                MILP_timeline.append(2 * i)
                MILP_timeline.append(2 * i + 1)
            MILP_timeline = np.array(MILP_timeline)  # Convert to numpy array for LogisticsModel.py numpy stuff
        else:
            sorted_unique_timeline = np.sort(np.unique(timestamp_list))
            # Trim timeline
            MILP_timeline = []
            real_time = []
            for i in range(len(sorted_unique_timeline)):
                real_time.append(int(sorted_unique_timeline[i]) * step_length)
                MILP_timeline.append(2*i)
                MILP_timeline.append(2*i + 1)
            MILP_timeline = np.array(MILP_timeline)  # Convert to numpy array for LogisticsModel.py numpy stuff

        if not LET_network:
            return sorted_unique_timeline, MILP_timeline, real_time
        else:
            return sorted_unique_timeline, MILP_timeline, real_time, int(max_LET_tof)

    def test_feasible(self, x) -> bool:
        ceqs, cineqs = self.get_constrs(x)
        if ceqs != []:
            ceq_check = all([i <= 0 for i in ceqs])
        else:
            ceq_check = True
        if cineqs != []:
            cineq_check = all([i <= 0 for i in cineqs])
        else:
            cineq_check = True

        constr_check = ceq_check and cineq_check
        bound_check = all([x[i] >= self.get_bounds()[0][i] and x[i] <= self.get_bounds()[1][i] for i in range(len(x))])
        return constr_check and bound_check