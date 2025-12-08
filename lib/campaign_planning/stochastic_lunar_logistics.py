import pandas as pd
import numpy as np
from typing import Optional, Any
from dataclasses import dataclass
from .LET import (DAYS_PER_SUNANGLE, SYNODIC_PERIOD)
from .vehicle_model import get_vehicle_data
from .misc import import_json_file, import_csv_file, process_campaign_reqs
from gurobipy import Model, tuplelist, GRB, Var
from copy import copy
from itertools import product

@dataclass()
class StochasticLunarLogMissionParameters():
    noise:np.ndarray[Any, np.dtype[np.float16]]

    # network_data_file = 'data/CampaignPlanningInputs/network_data.json'
    # campaign_data_file = 'data/CampaignPlanningInputs/campaign_requirements_CLPS.csv'
    # vehicle_data_file = 'data/CampaignPlanningInputs/vehicle_data_CLPS.csv'
    stacking_definition_file:str = ''
    network_data_file:str = 'data/CampaignPlanningInputs/network_data.json'
    campaign_data_file:str = 'data/CampaignPlanningInputs/campaign_requirements_artemis2B_LET.csv'
    vehicle_data_file:str = 'data/CampaignPlanningInputs/vehicle_data_Artemis2B_LET.csv'
    # stacking_definition_file:str = 'data/CampaignPlanningInputs/stacks_artemis2B.json'

    LET:bool = False
    ISRU:bool = False
    output:bool|str = False

    metaheuristic_decision:Optional[list[int]] = None
    restricted_timeline:Optional[list[int]] = None
    restricted_real_time:Optional[list[float]] = None

    def __post_init__(self) -> None:
        if self.metaheuristic_decision is not None:
            assert self.restricted_timeline is not None, "If metaheuristic_decision is defined, restricted_timeline must also be defined"
        if self.restricted_timeline is not None:
            assert self.restricted_real_time is not None, "If restricted_timeline is defined, restricted_real_time must also be defined"
        if self.restricted_timeline is None:
            assert self.restricted_real_time is None, "If restricted_timeline is not defined, restricted_real_time should not be defined"


def get_network_size(network_input) -> int:
    '''
        Find the largest index number of node in the network

        Input:
            network_input: network data file
    '''
    number_nodes = 0
    for edge_section in network_input:
        xValueIJ = edge_section[0]
        if xValueIJ[1] > number_nodes:
            number_nodes = xValueIJ[1]

    return number_nodes+1


class StochasticLunarScheduling:
    def __init__(self, 
                 mission_params:StochasticLunarLogMissionParameters
                ) -> None:
        '''
            Initialise the discrete scheduling model

            Input:
                mission_params: dataclass containing mission parameters
        '''   
        self.metaheuristic_decision = mission_params.metaheuristic_decision

        campaign_timeline = import_csv_file(mission_params.campaign_data_file)
        self.campaign_reqs = process_campaign_reqs(campaign_timeline)
        print(self.campaign_reqs)
        vehicle_data = import_csv_file(mission_params.vehicle_data_file)
        self.spacecraft = get_vehicle_data(vehicle_data)

        self.stacks = []

        if mission_params.stacking_definition_file != '':
            self.stacks = import_json_file(mission_params.stacking_definition_file)
            self.spacecraft.get_stacks(self.stacks)

        number_vehicles = np.size(self.spacecraft.prop_cap)
        self.number_non_stack_vehicles = number_vehicles - len(self.stacks)

        self.network_input = import_json_file(mission_params.network_data_file)
        self.number_nodes = get_network_size(self.network_input)

        self.LET_network = mission_params.LET
        self.ISRU = mission_params.ISRU
        self.output = mission_params.output

        self.restricted_timeline = mission_params.restricted_timeline
        self.restricted_real_time = mission_params.restricted_real_time

        self.noise = mission_params.noise
        # print("Initialising noise...")
        self.init_noise()

        # print("Initialising network...")
        self.init_network()

        if self.LET_network is not False:  self.init_LET()
        if self.ISRU: self.init_ISRU()
        else:
            self.ISRU_rate = 0

        self.init_maintenance()

        self.init_consum()
        # print("Initialising demand matrices...")
        self.init_demand_matrices()
        # print("Initialising schedule...")
        self.init_schedule()
        # print("Initialising variables indices...")
        self.init_vars()
        # print("Stochastic scheduler initialised")
        return
    
    def init_noise(self) -> None:
        self.noise_len = np.array([n.shape[0] for n in self.noise], dtype=int)  # Length of noise for each variable

        self.combinations = np.array(list(product(*[[2*n for n in range(N)] for N in self.noise_len])), dtype=np.float16)  # All possible combinations of noise

        self.distr_prob = np.array([np.prod([self.noise[i][int(x//2)] for i, x in enumerate(self.combinations[j])]) for j in
                            range(len(self.combinations))], dtype=np.float16)  # Probabilities of each combination occuring
        
        self.distr_log_prob = np.array([np.sum([np.log(self.noise[i][int(x//2)]) for i, x in enumerate(self.combinations[j])]) for j in
                            range(len(self.combinations))], dtype=np.float16)  # Log probability: easier to handle small numbers
        
        self.number_permutations = len(self.combinations )

        self.bigM = 1E6
        return
    
    def init_network(self) -> None:
        '''
            Extract information from network input to arc arrays
        '''
        self.number_nodes = get_network_size(self.network_input)
        if self.LET_network is not False:
            self.number_nodes += 1  # Add WSB node

        self.number_int_payloads = len(self.network_input[0][1])  # Number of integer payload types: 0 -vehicle(#); 1-crew(#);
        self.number_float_payloads = len(self.network_input[0][2])  # Number of float payload types: 0 -plant; 1-maintenance; 2-Consumption; 3-habitat+payload; 4-oxygen; # 5-kerosene   
        self.number_payload_types = self.number_int_payloads + self.number_float_payloads  # Total number of payload types
        self.number_vehicles = np.size(self.spacecraft.prop_cap)
        self.non_stack_vehicles = self.number_vehicles - len(self.stacks)
        self.number_payloads = len(self.campaign_reqs)
        

        # Define indices
        self.N = range(self.number_vehicles)
        self.I = range(self.number_nodes)
        self.CI = range(self.number_int_payloads)
        self.CF = range(self.number_float_payloads)
        self.C = range(self.number_payload_types)
        self.IO = range(2)


        ## TIMELINE MUST BE INCREASED BY LENGTH OF DELAY DISTRUBUTION
        self.T = range(int((max([row[5] for row in self.campaign_reqs])+1 + len(self.noise)) * 2))
        self.real_time = [t*SYNODIC_PERIOD for t in self.T]
        
        self.P = range(self.number_payloads)
        self.omega = range(self.number_permutations)

        # Define parameters
        self.arc_cost_matrix_F = np.zeros((self.number_vehicles, self.number_nodes, self.number_nodes, self.number_float_payloads, 2, max(self.T)+1), dtype=np.int32)
                                        
        self.arc_cost_matrix_I = np.zeros((self.number_vehicles, self.number_nodes, self.number_nodes, self.number_int_payloads, 2, max(self.T)+1), dtype=np.int32)

        # Existing arcs
        self.arc_exist = np.zeros((self.number_nodes, self.number_nodes, max(self.T)+1), dtype=bool)

        # Propellant mass fraction for transfer
        self.arc_FPI = np.zeros((self.number_vehicles, self.number_nodes, self.number_nodes, max(self.T)+1), dtype=np.float32)
                
        # Arc time of flight
        self.arc_disc_time = np.zeros((self.number_vehicles, self.number_nodes, self.number_nodes, max(self.T)+1), dtype=np.int32)
                    
        # Real edge time (for oxygen, water , food)
        self.arc_real_time = np.zeros((self.number_vehicles, self.number_nodes, self.number_nodes, max(self.T)+1), dtype=np.float32)

        self.valid_arc = np.zeros((self.number_vehicles, self.number_nodes, self.number_nodes, max(self.T)+1), dtype=bool)

        for edge_section in self.network_input:
            i, j = edge_section[0]
            int_payload_mass = edge_section[1]
            float_payload_mass = edge_section[2]
            for n in self.N:
                for t in self.T:
                    for c in self.CI:
                        if c == 0:  # Spacecraft mass
                            self.arc_cost_matrix_I[n,i,j,c,0,t] = int_payload_mass[c] * self.spacecraft.dry_mass[n]
                        else:
                            self.arc_cost_matrix_I[n,i,j,c,0,t] = int_payload_mass[c]
                    for c in self.CF:
                        self.arc_cost_matrix_F[n,i,j,c,0,t] = float_payload_mass[c]
                    if (t % 2) == 0 and edge_section[4][0] != 1 and edge_section[3] == 1:  # Outbound transfer arc exists on even time step
                        self.arc_exist[i,j,t] = edge_section[3]  # Does arc exist?
                    elif (t % 2) != 0 and edge_section[4][0] != 0 and edge_section[3] == 1:
                        self.arc_exist[i,j,t] = edge_section[3]  # Does arc exist?
                    self.arc_disc_time[n,i,j,t] = edge_section[4][2]  # "fake" time
                    if (t % 2) == 0 or ((t % 2) != 0 and i != j):  # Transfer arcs take transfer time. Even holdovers take surface mission length.             
                        self.arc_real_time[n,i,j,t] = edge_section[4][1] / DAYS_PER_SUNANGLE  # Real edge time (for oxygen, food, water)
                    elif t != max(self.T):  # Odd holdovers move to next point on the higher level timeline
                        self.arc_real_time[n,i,j,t] = (self.real_time[int((t - 1) / 2 + 1)] - self.real_time[int((t - 1) / 2)] - edge_section[4][1]) / DAYS_PER_SUNANGLE                
                    if (i == 1 and j == 1) is False:  # FPI is not time dependent unless LLO holdover arc
                        self.arc_FPI[n,i,j,t] = 1 - np.exp(-(edge_section[4][3] * 1000) / (self.spacecraft.Isp[n] * 9.8))  # Propellant mass fraction
                    else:  # LLO stationkeeping dV is m/s per second per year, so scale by real time assocaited with the arc
                        self.arc_FPI[n,i,j,t] = 1 - np.exp(
                            -(edge_section[4][3] * self.arc_real_time[n,i,j,t] / 365 * DAYS_PER_SUNANGLE * 1000) / (self.spacecraft.Isp[n] * 9.8))
                        
        for t in self.T:
            for i in self.I:
                for j in self.I:
                    if self.arc_exist[i,j,t]:
                        for n in self.N:
                            if [i, j] in self.spacecraft.domain[n]:
                                self.valid_arc[n,i,j,t] = 1
        return
    
    def init_LET(self) -> None:
        '''
            Extract information from LET network data to arc arrays
        '''
        for n in self.N:
            for edge_section in self.LET_network[n]:
                i, j = edge_section[0]
                int_payload_mass = edge_section[1]
                float_payload_mass = edge_section[2]
                for t in self.T:
                    for c in self.CI:
                        if c == 0:  # Spacecraft mass
                            self.arc_cost_matrix_I[n,i,j,c,0,t] = int_payload_mass[c] * self.spacecraft.dry_mass[n]
                        else:
                            self.arc_cost_matrix_I[n,i,j,c,0,t] = int_payload_mass[c]
                    for c in self.CF:
                        self.arc_cost_matrix_F[n,i,j,c,0,t] = float_payload_mass[c]
                    self.arc_FPI[n,i,j,t] = 1 - np.exp(-(edge_section[4][3] * 1000) / (self.spacecraft.Isp[n] * 9.8))  # Propellant mass fraction
                    if (t % 2) == 0 or ((t % 2) != 0 and i != j):  # Transfer arcs take transfer time. Even holdovers take surface mission length.        
                        self.arc_real_time[n,i,j,t] = edge_section[4][1]  # Real edge time (for oxygen, food, water)
                    if (t % 2) == 0 and edge_section[4][0] != 1:  # Outbound LET transfer arc exists on even time step
                        self.arc_exist[i,j,t] = edge_section[3]  # Does arc exist?
                    else:  # No return LETs so no inbound arcs
                        self.arc_exist[i,j,t] = 0
                    self.arc_disc_time[n,i,j,t] = int(edge_section[4][2])  # "fake" time
        for t in self.T:  # Go back and remove LET arcs that cross trimmed time steps
            if t < 6:
                self.arc_exist[3,1,t] = 0
            elif t + 10 < max(self.T):
                if max([sum(self.arc_real_time[n,0,0,a] for a in range(t, t + self.arc_disc_time[n,0,3,t] + self.arc_disc_time[n,3,1,t + self.arc_disc_time[n,0,3,t]]))
                    for n in range(self.number_vehicles)]) - 1 * SYNODIC_PERIOD / DAYS_PER_SUNANGLE > max(
                    [self.arc_real_time[n,0,3,t] + self.arc_real_time[n,3,1,t + self.arc_disc_time[n,0,3,t]] for n in range(self.number_vehicles)]
                ):        
                    self.arc_exist[0,3,t] = 0
                if sum(self.arc_exist[0,3,t - self.arc_disc_time[n,0,3,t]] for n in
                    range(self.number_vehicles)) == 0:  # If no more inbound LETs exist at this time, remove the outbound LET
                    self.arc_exist[3,1,t] = 0
                else:
                    self.arc_exist[3,1,t] = 1
            elif t + 3 < max(self.T):
                self.arc_exist[0,3,t] = 0
                if sum(self.arc_exist[0,3,t - self.arc_disc_time[n,0,3,t]] for n in
                    range(self.number_vehicles)) == 0:  # If no more inbound LETs exist at this time, remove the outbound LET
                    self.arc_exist[3,1,t] = 0
                else:
                    self.arc_exist[3,1,t] = 1
            else:
                self.arc_exist[0,3,t] = 0
                self.arc_exist[3,1,t] = 0

        return

    def init_ISRU(self) -> None:
        '''
            Initialise ISRU parameters
            TODO: Add external dataclass so this isn't hard coded
        '''
        # ISRU productivity, kg/day/kg plant
        ISRU_rate_per_plant_kg = 4  # kg/yr/kg plant from Schreiner (2016) Fig. 10
        power_per_ISRU_rate = 0.01  # kW/kg/yr from Schreiner (2016) Fig. 10
        specific_power = 6.5E-3  # kW/kg from KRUSTY Gibson (2015) Tab. 1
        power_mass_per_ISRU_rate = power_per_ISRU_rate / specific_power  # kg power/kg/yr
        self.ISRU_rate = 1 / (1 / ISRU_rate_per_plant_kg + power_mass_per_ISRU_rate) / 365 * DAYS_PER_SUNANGLE  # kg/day/kg plant
        self.ISRU_maintenance = 0.1 / 365 * DAYS_PER_SUNANGLE  # 10% of ISRU plant mass per year
        return
    
    def init_maintenance(self) -> None:
        '''
            Initialise infrastructure maintenance parameters
            TODO: Add external dataclass so this isn't hard coded
        '''
        # ISRU productivity, kg/day/kg plant
        self.infra_maintenance = 0.1 / 365 * DAYS_PER_SUNANGLE  # 10% of ISRU plant mass per year
        return

    def init_consum(self) -> None:
        '''
            Initialise consumable payload masses
            TODO: Add external dataclass so this isn't hard coded
        '''
        # Consumable payload masses
        food_cost = 0.28 + 0.015 + 0.05 + 0.06 + 0.7  # /day/person, dehydrated
        water_cost = 5.31 + 1.06  # water+tank
        oxygen_cost = 0.84 + 0.34  # oxygen+tank
        self.consumption_cost = (food_cost + water_cost + oxygen_cost) * DAYS_PER_SUNANGLE  # kg/sun angle/person
        return

    def init_demand_matrices(self) -> None:
        '''Initialise demand matrices
        '''
        self.demand_matrix = np.zeros((self.number_nodes, 
                                       self.number_payload_types, 
                                       max(self.T)+1, 
                                       self.number_vehicles), 
                                       dtype=np.float32)
        
        self.hold_matrix = np.zeros((self.number_nodes, 
                                     self.number_payload_types, 
                                     max(self.T)+1, 
                                     self.number_vehicles), 
                                     dtype=np.float32)

        # Unlimited supply from Earth for some things
        for t in self.T:
            for n in self.N:
                for c in [2,3,4,6,7]:
                    self.demand_matrix[0,c,t,n] = float('inf')  # Infinite supply from node 0 (Earth)

        self.mutable_demand_matrix = np.zeros((self.number_nodes, self.number_payload_types, self.number_vehicles, self.number_payloads), dtype=np.float32)
        self.mutable_hold_matrix = np.zeros((self.number_nodes, self.number_payload_types, self.number_vehicles, self.number_payloads), dtype=np.float32)
                        
        # Vehicle supply
        for n in range(self.spacecraft.number_non_stack_vehicles):
            for t in range(self.spacecraft.available_from[n]*2, max(self.T)+1, self.spacecraft.launch_frequency[n]*2):
                self.demand_matrix[0,0,t,n] += 1

        for pay, row in enumerate(self.campaign_reqs):
            self.mutable_demand_matrix[int(row[2]),int(row[0]),0,pay] = (np.float32(row[1]))  # Supplies
            if int(row[0]) == 1:  # Crew always return
                self.mutable_hold_matrix[int(row[3]),int(row[0]),0,pay] += float(row[1])  # Hold at destination
                # self.mutable_demand_matrix[int(row[2]),int(row[0]),0,pay] = -float(row[1])  # Demand at origin
            elif int(row[0]) == 2:  # Permanent infrastructure remains until end of mission
                self.mutable_hold_matrix[int(row[3]),int(row[0]),0, pay] += int(row[1]) # Hold at destination at required time
                self.demand_matrix[int(row[3]),int(row[0]),self.T[-1],0] -= int(row[1]) # Demand at destination at end of mission
            else:
                self.mutable_demand_matrix[int(row[3]),int(row[0]),0,pay] = -np.float32(row[1])  # Demand
        
        return 

    def init_vars(self, names=False) -> None|list:
        self.xI_index = tuplelist([(n, i, j, c, io, t, om) for om in self.omega for n in self.N for i in self.I for j in self.I for c in self.CI for io in self.IO for t in self.T if self.valid_arc[n,i,j,t]])
        self.xI_cost = {}
        self.xF_index = tuplelist([(n, i, j, c, io, t, om) for om in self.omega for n in self.N for i in self.I for j in self.I for c in self.CF for io in self.IO for t in self.T if self.valid_arc[n,i,j,t]])
        self.xF_cost = {}
        self.B_index = tuplelist([(pay, t) for pay in range(self.number_payloads) for t in self.T])
        for ind in self.xI_index:
            self.xI_cost[ind] = self.arc_cost_matrix_I[ind[:-1]]
        for ind in self.xF_index:
            self.xF_cost[ind] = self.arc_cost_matrix_F[ind[:-1]]

        if names:
            xI_name = "xI"
            xF_name = "xF"
            B_name = "Binary"
            return [xI_name, self.xI_index], [xF_name, self.xF_index], [B_name, self.B_index]
        else:
            return

    def init_schedule(self) -> None:
        '''
            Extract scheduling information from campaign requirements
        '''
        self.soft_precursors = [[] for pay in self.P]
        self.strict_precursors = [[] for pay in self.P]
        self.co_payloads = [[] for pay in self.P]
        for pay in self.P:
            self.soft_precursors[pay] = [int(el) for el in self.campaign_reqs[pay][6].split(',') if self.campaign_reqs[pay][6] not in ['', ' ']]
            self.strict_precursors[pay] = [int(el) for el in self.campaign_reqs[pay][7].split(',') if self.campaign_reqs[pay][7] not in ['', ' ']]
            self.co_payloads[pay] = [int(el) for el in self.campaign_reqs[pay][8].split(',') if self.campaign_reqs[pay][8] not in ['', ' ']]
        return       
    
    def build_primal_model(self,
                        subproblem:Optional[list[list[int]]]=None,
                        relaxed:bool=False,
                        subsolution:bool|Model=False,
                        prev_MILP_solution:bool|Model=False,
                        gen_var:Optional[list[str]]=None) -> Model:
        
        #print("Building scheduling model...")
        model = self.build_discrete_scheduling(subproblem=subproblem, relaxed=relaxed, subsolution=subsolution, prev_MILP_solution=prev_MILP_solution, gen_var=gen_var)

        if self.output is not False:
            model.optimize()
            writer = pd.ExcelWriter(self.output)
            for n in self.N:
                time_list = []
                for t in self.T:
                    node_list = []
                    node_keys = []
                    for i in self.I:
                        for j in self.I:
                            if self.valid_arc[n,i,j,t]:
                                payload_list = []
                                for c in self.C:
                                    if c < len(self.CI):
                                        payload_list.append(pd.DataFrame({c: model.xI[n, i, j, c, 0, t].x}, index=[t]))
                                    else:
                                        payload_list.append(
                                            pd.DataFrame({c: model.xF[n, i, j, c - len(self.CI), 0, t].x}, index=[t]))
                                node_keys.append(str([i, j]))
                                node_list.append(pd.concat(payload_list, axis=1))
                    time_list.append(pd.concat(node_list, keys=node_keys, axis=1))
                time_df = pd.concat(time_list)
                sheetname = 'Vehicle ' + str(n)
                time_df.to_excel(writer, header=True, sheet_name=sheetname)
            writer.close()
            return model, self.arc_disc_time, self.valid_arc
        else:
            return model    

    def build_discrete_scheduling(self,
                                subproblem:Optional[list[list[int]]]=None,
                                relaxed:bool=False,
                                subsolution:bool|Model=False,
                                prev_MILP_solution:bool|Model=False,
                                gen_var=None) -> Model:
        
        """
        Builds the discrete scheduling model based on the given inputs.

        Parameters:
            - subproblem (bool or list, optional): Flag indicating whether it is a subproblem. Defaults to False. Should list the subproblem
            time indices if not false.
            - relaxed (bool, optional): Flag indicating whether the integer variables should be relaxed. Defaults to False.
            - subsolution (bool or Model, optional): A solution to the subproblem. Defaults to False.
            - prev_MILP_solution (bool or Model, optional): A previous MILP solution. Defaults to False.
        Returns:
            - Model
        """  

        assert (subsolution is not False and subproblem is False) is False, "Subproblem columns must be defined if a solved subproblem model is provided"
        assert (relaxed is True and prev_MILP_solution is not False) is False, "Cannot relax the model if a previous MILP solution is provided" 

        if subproblem is not None and subsolution is False:  # If subproblem, only consider the time steps in the subproblem for the binary variables
            B_index = tuplelist(tuple(index) for index in subproblem[0])
        else:  # Otherwise, consider all time steps for the binary variables
            B_index = self.B_index

        model = Model('Model')
    
        xI, xF, B, alpha, beta, gamma, delta = self.add_vars(model, B_index, subproblem, relaxed, subsolution)

        objExpr0 =  delta
        
        objExpr1 = sum(self.distr_prob[om]*alpha[om] for om in self.omega)

        model.setObjectiveN(objExpr0, 0, priority=1)
        model.setObjectiveN(objExpr1, 1, priority=2)
                             
        self.add_constrs(model, xI, xF, B=B, B_index=B_index, alpha=alpha)

        model.addConstr(beta*sum(self.distr_prob[om]*(1-alpha[om]) for om in self.omega) == 1)

        model.addConstr(gamma 
                        == sum(self.distr_prob[om]*(1-alpha[om])*(
                                sum(self.arc_cost_matrix_I[ind[:-1]]*xI[ind] for ind in self.xI_index if ind[-1] == om) +
                                sum(self.arc_cost_matrix_F[ind[:-1]]*xF[ind] for ind in self.xF_index if ind[-1] == om)
                                ) for om in self.omega))
                            
        model.addConstr(delta == beta*gamma)
        
        return model

    def add_vars(self, 
                 model:Model, 
                 B_index:tuplelist,
                 subproblem:Optional[list[list[int]]]=None,
                 relaxed:bool=False,
                 subsolution:Optional[Model]=None
                ) -> Var:

        alpha = model.addVars(self.omega, name='alpha', vtype=GRB.BINARY)
        beta = model.addVar(name='beta', vtype=GRB.CONTINUOUS, lb=0)
        gamma = model.addVar(name='gamma', vtype=GRB.CONTINUOUS, lb=0)
        delta = model.addVar(name='delta', vtype=GRB.CONTINUOUS, lb=0)
        xF = model.addVars(self.xF_index, obj=self.xF_cost, name='xF', vtype=GRB.CONTINUOUS)

        if relaxed:
            xI = model.addVars(self.xI_index, obj=self.xI_cost, name='xI', vtype=GRB.CONTINUOUS)

            B = model.addVars(B_index, obj=0, name='Binary', vtype=GRB.CONTINUOUS, ub=1)

            # If a submodel solution was provided, fix the variables to the subsolution by fixing the bounds
            if subsolution is not None:
                for ind in B_index:
                    if list(ind) in subproblem[0]:
                        B[ind].setAttr('ub', 1.01*subsolution.getVarByName("Binary[%d,%d]" % (ind)).X)
                        B[ind].setAttr('lb', 0.99*subsolution.getVarByName("Binary[%d,%d]" % (ind)).X)
                    else:
                        B[ind].setAttr('ub', 0)
                for ind in self.xI_index:
                    xI[ind].setAttr('ub', 1.01*subsolution.getVarByName("xI[%d,%d,%d,%d,%d,%d]" % (ind)).X)
                    xI[ind].setAttr('lb', 0.99*subsolution.getVarByName("xI[%d,%d,%d,%d,%d,%d]" % (ind)).X)
                for ind in self.xF_index:
                    xF[ind].setAttr('ub', 1.01*subsolution.getVarByName("xF[%d,%d,%d,%d,%d,%d]" % (ind)).X)
                    xF[ind].setAttr('lb', 0.99*subsolution.getVarByName("xF[%d,%d,%d,%d,%d,%d]" % (ind)).X)

        else: # Otherwise, binary and integer variables are binary/integer as normal (don't expect a subsolution in this case)
            B = model.addVars(B_index, obj=0, name="Binary", vtype=GRB.BINARY)
            xI = model.addVars(self.xI_index, obj=self.xI_cost, name='xI', vtype=GRB.INTEGER)


        return xI, xF, B, alpha, beta, gamma, delta

    def add_constrs(self, 
                    model:Model,
                    xI:Var,
                    xF:Var,
                    B:Var,
                    B_index:tuplelist,
                    alpha:Var
                ) -> Model:
        
        model = self.add_binary_scheduling_constrs(model, alpha, B, self.B_index)
        model = self.add_demands_constrs(model, xI, xF, B, B_index, alpha)
        model = self.add_capacity_constrs(model, xI, xF)
        model = self.add_commodity_dynamics_constrs(model, xI, xF)

        return model
        
    def add_binary_scheduling_constrs(self, 
                                      model:Model,
                                      alpha:Var,
                                      binary_vars:Var,
                                      B_index:tuplelist
                                    ) -> Model:
        # Binary scheduling constraints
        # Soft precursors
        model.addConstrs((sum(binary_vars[pay, time-self.combinations[om][pay]] for time in range(t+1) if (pay,time-self.combinations[om][pay]) in B_index) 
                          <= alpha[om]*max(self.T) + sum(binary_vars[pre, time-self.combinations[om][pre]] for time in range(t+1) if (pre,time-self.combinations[om][pre]) in B_index)
                        for pay in self.P for t in self.T for pre in self.soft_precursors[pay] for om in self.omega
                        ),
                        name='2A')
        # Strict precursors
        model.addConstrs((sum(binary_vars[pay, time-self.combinations[om][pay]] for time in range(t+1) if (pay,time-self.combinations[om][pay]) in B_index) 
                          <= alpha[om]*max(self.T) + sum(binary_vars[pre, time-self.combinations[om][pre]] for time in range(t-1) if (pre,time-self.combinations[om][pre]) in B_index)
                        for pay in self.P for t in self.T for pre in self.strict_precursors[pay] for om in self.omega
                        ),
                        name='2B')
        #Co-payloads
        model.addConstrs(((1-alpha[om])*(binary_vars[pay, t-self.combinations[om][pay]] - binary_vars[co,t-self.combinations[om][co]]) == 0
                        for pay in self.P for t in self.T for co in self.co_payloads[pay] for om in self.omega
                        if (pay,t-self.combinations[om][pay]) in B_index and self.campaign_reqs[pay][3] > self.campaign_reqs[pay][2] 
                        and co != '' and (co,t-self.combinations[om][co]) in B_index),
                        name='2C_outbound')
        model.addConstrs(((1-alpha[om])*(binary_vars[pay, t-self.combinations[om][pay]] - binary_vars[co,t-1-self.combinations[om][co]]) == 0
                        for pay in self.P for t in self.T for co in self.co_payloads[pay] for om in self.omega
                        if (pay,t-self.combinations[om][pay]) in B_index and self.campaign_reqs[pay][3] < self.campaign_reqs[pay][2] 
                        and co != '' and (co,t-self.combinations[om][co]) in B_index and self.campaign_reqs[co][3] > self.campaign_reqs[co][2]),
                        name='2C_return_co_outbound')
        model.addConstrs(((1-alpha[om])*(binary_vars[pay, t-self.combinations[om][pay]] - binary_vars[co,t-self.combinations[om][co]] ) == 0
                        for pay in self.P for t in self.T for co in self.co_payloads[pay] for om in self.omega
                        if (pay,t-self.combinations[om][pay]) in B_index and self.campaign_reqs[pay][3] < self.campaign_reqs[pay][2] 
                        and co != '' and (co,t-self.combinations[om][co]) in B_index and self.campaign_reqs[co][3] < self.campaign_reqs[co][2]),
                        name='2C_return_co_return')
        # Everything launches once
        model.addConstrs((sum(binary_vars[pay, t] for t in self.T if (pay, t) in B_index) == 1 
                        for pay in self.P),
                        name='2D')
        # # Launch windows
        model.addConstrs(((1-alpha[om])*binary_vars[pay, t-self.combinations[om][pay]] == 0 
                        for pay in self.P for t in self.T  for om in self.omega
                        if (pay, t-self.combinations[om][pay]) in B_index and (t-self.combinations[om][pay] < self.campaign_reqs[pay][4]*2 or t-self.combinations[om][pay] > self.campaign_reqs[pay][5]*2 
                                                    or (self.campaign_reqs[pay][3] > self.campaign_reqs[pay][2] and t % 2 == 1)
                                                    or (self.campaign_reqs[pay][3] < self.campaign_reqs[pay][2] and t % 2 == 0))),
                        name='2E')
        
        # Don't allow scheduled launch after official end of campaign, only delays can happen here
        model.addConstrs((binary_vars[pay, t] == 0 for pay in self.P for t in self.T if (pay, t) in B_index and t > max(self.T)-2*len(self.noise)),
                        name='2F')
        
        return model

    def add_demands_constrs(self,
                            model:Model,
                            xI:Var,
                            xF:Var,
                            B:Var,
                            B_index:tuplelist,
                            alpha:Var
                        ) -> Model:
        # Vehicles
        if self.stacks == []:
            model.addConstrs((sum(xI[n, i, j, 0, 0, t, om] for j in self.I if (n, i, j, 0, 0, t, om) in self.xI_index) - sum(xI[n, j, i, 0, 1, t - self.arc_disc_time[n,j,i,t], om] for j in self.I 
                            if (n, j, i, 0, 1, t - self.arc_disc_time[n,j,i,t], om) in self.xI_index) <= int(self.demand_matrix[i,0,t,n]) 
                            for i in self.I for t in self.T for n in self.N for om in self.omega),
                            name='1A')
        else:
            model.addConstrs((sum(xI[n, i, j, 0, 0, t, om] for j in self.I if (n, i, j, 0, 0, t, om) in self.xI_index) 
                                + sum(sum(xI[int(self.non_stack_vehicles+stack_num), i, j, 0, 0, t, om] for j in self.I if (int(self.non_stack_vehicles+stack_num), i, j, 0, 0, t, om) in self.xI_index) 
                                    for stack_num in range(len(self.stacks)) if n in self.stacks[stack_num])
                                - sum(xI[n, j, i, 0, 1, t-self.arc_disc_time[n,j,i,t], om] for j in self.I if (n, j, i, 0, 1, t-self.arc_disc_time[n,j,i,t], om) in self.xI_index)
                                - sum(sum(xI[int(self.non_stack_vehicles+stack_num), j, i, 0, 1, t-self.arc_disc_time[n,j,i,t], om] for j in self.I if (int(self.non_stack_vehicles+stack_num), j, i, 0, 1, t-self.arc_disc_time[n,j,i,t], om) in self.xI_index)
                                    for stack_num in range(len(self.stacks)) if n in self.stacks[stack_num])
                            <= int(self.demand_matrix[i,0,t,n])
                            for i in self.I for t in self.T for n in self.N  for om in self.omega
                            if n < self.non_stack_vehicles),
                            name='1A')
        # Other integer payloads
        model.addConstrs((sum(sum(xI[n, i, j, c, 0, t, om] for j in self.I if (n, i, j, c, 0, t, om) in self.xI_index) for n in self.N) 
                                - sum(sum(xI[n, j, i, c, 1, t - self.arc_disc_time[n,j,i,t], om] for j in self.I if (n, j, i, c, 1, t - self.arc_disc_time[n,j,i,t], om) in self.xI_index) for n in self.N)
                            <= alpha[om]*self.bigM + sum(self.demand_matrix[i,c,t,n] + sum((B[p, t-self.combinations[om][p]]-B[p, t-1-self.combinations[om][p]])*self.mutable_demand_matrix[i,c,n,p] 
                                                                                           for p in self.P if ((p, t-self.combinations[om][p]) in B_index and p, t-1-self.combinations[om][p]) in B_index) for n in self.N) 
                            for i in self.I for t in self.T for c in self.CI for om in self.omega if (c > 0 and i != 0)),
                            name='1B_ints')

        # Continuous payloads
        model.addConstrs((sum(sum(xF[n, i, j, c, 0, t, om] for j in self.I if (n, i, j, c, 0, t, om) in self.xF_index) for n in self.N) 
                                - sum(sum(xF[n, j, i, c, 1, t - self.arc_disc_time[n,j,i,t], om] for j in self.I if (n, j, i, c, 1, t - self.arc_disc_time[n,j,i,t], om) in self.xF_index) for n in self.N)
                            <= alpha[om]*self.bigM + sum(self.demand_matrix[i,c+self.number_int_payloads,t,n] for n in self.N) 
                                + sum(sum(B[p, t-self.combinations[om][p]]*self.mutable_demand_matrix[i,c+self.number_int_payloads,n,p] for p in self.P if (p, t-self.combinations[om][p]) in B_index) for n in self.N) 
                            for i in self.I for t in self.T for c in self.CF for om in self.omega),
                            name='1B_cont')
        
        # Hold constraints
        model.addConstrs((sum(xI[n, i, i, c, t % 2, t - t % 2, om]  for n in self.N if (n, i, i, c, t % 2, t - t % 2, om) in self.xI_index)     
                            >= -alpha[om]*self.bigM + sum(sum(B[p, t-self.combinations[om][p]]*self.mutable_hold_matrix[i,c,n,p] for p in self.P if (p, t-self.combinations[om][p]) in B_index) for n in self.N) 
                            for i in self.I for t in self.T for c in self.CI for om in self.omega if c > 0),
                            name='1C_ints')
        
        model.addConstrs((sum(xF[n, i, i, 0, t % 2, t - t % 2, om] for n in self.N if (n, i, i, 0, t % 2, t - t % 2, om) in self.xF_index) 
                            >= -alpha[om]*self.bigM + sum(sum(sum(B[p, t_prime] for t_prime in range(int(t-self.combinations[om][p]+1)) if (p,t_prime) in B_index)*self.mutable_hold_matrix[i,2,n,p] for p in self.P) for n in self.N) 
                            for i in self.I for t in self.T for om in self.omega),
                            name='1C_infra')
        
        model.addConstrs((sum(xF[n, i, i, c, t % 2, t - t % 2, om] for n in self.N if (n, i, i, c, t % 2, t - t % 2, om) in self.xF_index) 
                            >= -alpha[om]*self.bigM + sum(sum(sum(B[p, t_prime] for t_prime in range(int(t-self.combinations[om][p]+1)) if (p, t_prime) in B_index)*self.mutable_hold_matrix[i,c+self.number_int_payloads,n,p] for p in self.P ) for n in self.N) 
                            for i in self.I for t in self.T for c in self.CF[1:] for om in self.omega),
                            name='1C_cont')
        return model

    def add_capacity_constrs(self,
                            model:Model,
                            xI:Var,
                            xF:Var
                        ) -> Model:
        # Capacities
        # Oxidiser
        model.addConstrs((xF[n, i, j, 4, 0, t, om] <= self.spacecraft.oxy_ratio[n]*xI[n, i, j, 0, 0, t, om]*self.spacecraft.prop_cap[n]
                         for n in self.N for i in self.I for j in self.I for t in self.T  for om in self.omega
                         if (n, i, j, 4, 0, t, om) in self.xF_index),
                         name='3A')
                                          
        # Fuel
        model.addConstrs((xF[n, i, j, 5, 0, t, om] <= (1-self.spacecraft.oxy_ratio[n])*xI[n, i, j, 0, 0, t, om]*self.spacecraft.prop_cap[n]
                         for n in self.N for i in self.I for j in self.I for t in self.T for om in self.omega
                         if (n, i, j, 5, 0, t, om) in self.xF_index),
                         name='3B')
                                            
        # Payload
        model.addConstrs((100*xI[n, i, j, 1, 0, t, om] + xF[n, i, j, 0, 0, t, om] + xF[n, i, j, 1, 0, t, om] + xF[n, i, j, 2, 0, t, om] + xF[n, i, j, 3, 0, t, om]
                        <= xI[n, i, j, 0, 0, t, om] * self.spacecraft.payload_cap[n]
                        for n in self.N for i in self.I for j in self.I for t in self.T for om in self.omega
                        if (n, i, j, 1, 0, t, om) in self.xI_index and (n, i, j, 0, 0, t, om) in self.xI_index 
                        and (n, i, j, 0, 0, t, om) in self.xF_index and (n, i, j, 1, 0, t, om) in self.xF_index and (n, i, j, 2, 0, t, om) in self.xF_index and (n, i, j, 3, 0, t, om) in self.xF_index
                        and (i != j or i not in (2,4))),
                        name='3C')
        model.addConstrs((100*xI[n, i, i, 1, 0, t, om] + xF[n, i, i, 1, 0, t, om] + xF[n, i, i, 2, 0, t, om] + xF[n, i, i, 3, 0, t, om]    
                        <= xI[n, i, i, 0, 0, t, om] * self.spacecraft.payload_cap[n]
                        for n in self.N for t in self.T for om in self.omega for i in (2,4)
                        if (n, i, i, 1, 0, t, om) in self.xI_index and (n, i, i, 0, 0, t, om) in self.xI_index
                        and (n, i, i, 1, 0, t, om) in self.xF_index and (n, i, i, 2, 0, t, om) in self.xF_index and (n, i, i, 3, 0, t, om) in self.xF_index),
                        name='3C_moon_surface')
        return model

    def add_commodity_dynamics_constrs(self,
                                       model:Model,
                                       xI:Var,
                                       xF:Var
                                    ) -> Model:
        # Dynamics
        model.addConstrs((xI[n, i, j, c, 1, t, om] == xI[n, i, j, c, 0, t, om] 
                        for n in self.N for i in self.I for j in self.I for c in self.CI for t in self.T for om in self.omega
                        if (n, i, j, c, 1, t, om) in self.xI_index and (n, i, j, c, 0, t, om) in self.xI_index),
                        name='4A_int')
        
        # Maintenance 
        model.addConstrs((sum(xF[n, i, i, 1, 1, t, om] for n in self.N if (n, i, i, 1, 1, t, om) in self.xF_index) 
                        == sum(xF[n, i, i, 1, 0, t, om] - (self.infra_maintenance*self.arc_real_time[n,i, i, t])*xF[n, i, i, 0, 0, t, om] for n in self.N 
                                if (n, i, i, 1, 0, t, om) in self.xF_index and (n, i, i, 0, 0, t, om) in self.xF_index)
                        for t in self.T for om in self.omega for i in (2,4)),
                        name="4_maintenance_moon")
        model.addConstrs((xF[n, i, j, 1, 1, t, om] == xF[n, i, j, 1, 0, t, om]
                        for n in self.N for i in self.I for j in self.I for t in self.T for om in self.omega
                        if (n, i, j, 1, 1, t, om) in self.xF_index and (n, i, j, 1, 0, t, om) in self.xF_index and (i != j or i not in (2,4))),
                        name='4_maintenance_other')
        # Consumable 
        model.addConstrs((xF[n, i, j, 2, 1, t, om] == xF[n, i, j, 2, 0, t, om] - self.consumption_cost*self.arc_real_time[n,i,j,t]*xI[n, i, j, 1, 0, t, om]
                        for n in self.N for i in self.I for j in self.I for t in self.T for om in self.omega
                        if (n, i, j, 2, 1, t, om) in self.xF_index and (n, i, j, 2, 0, t, om) in self.xF_index and (n, i, j, 1, 0, t, om) in self.xI_index),
                        name='4_consumables')

        # Oxidiser
        model.addConstrs((xF[n, i, j, 4, 1, t, om] 
                        <= ((1-self.spacecraft.oxy_boil_off_rate[n])**self.arc_real_time[n,i,j,t])*(xF[n, i, j, 4, 0, t, om]
                            - (self.spacecraft.oxy_ratio[n])*self.arc_FPI[n,i,j,t]*(100*xI[n, i, j, 1, 0, t, om]                          
                            + xF[n, i, j, 0, 0, t, om] + xF[n, i, j, 1, 0, t, om] + xF[n, i, j, 2, 0, t, om]
                            + xF[n, i, j, 3, 0, t, om] + xF[n, i, j, 4, 0, t, om] + xF[n, i, j, 5, 0, t, om]  + self.spacecraft.dry_mass[n]*xI[n, i, j, 0, 0, t, om]))
                        for n in self.N for i in self.I for j in self.I for t in self.T for om in self.omega
                        if (n, i, j, 5, 1, t, om) in self.xF_index and (n, i, j, 5, 0, t, om) in self.xF_index and (n, i, j, 1, 0, t, om) in self.xI_index
                        and (n, i, j, 0, 0, t, om) in self.xF_index and (n, i, j, 1, 0, t, om) in self.xF_index and (n, i, j, 2, 0, t, om) in self.xF_index
                        and (n, i, j, 3, 0, t, om) in self.xF_index and (n, i, j, 4, 0, t, om) in self.xF_index),
                        name='4_oxy')
        # Fuel 
        model.addConstrs((xF[n, i, j, 5, 1, t, om] 
                        <= ((1-self.spacecraft.fuel_boil_off_rate[n])**self.arc_real_time[n,i,j,t])*(xF[n, i, j, 5, 0, t, om]
                            - (1-self.spacecraft.oxy_ratio[n])*self.arc_FPI[n,i,j,t]*(100*xI[n, i, j, 1, 0, t, om]                          
                            + xF[n, i, j, 0, 0, t, om] + xF[n, i, j, 1, 0, t, om] + xF[n, i, j, 2, 0, t, om]
                            + xF[n, i, j, 3, 0, t, om] + xF[n, i, j, 4, 0, t, om] + xF[n, i, j, 5, 0, t, om]  + self.spacecraft.dry_mass[n]*xI[n, i, j, 0, 0, t, om]))
                        for n in self.N for i in self.I for j in self.I for t in self.T for om in self.omega
                        if (n, i, j, 5, 1, t, om) in self.xF_index and (n, i, j, 5, 0, t, om) in self.xF_index and (n, i, j, 1, 0, t, om) in self.xI_index
                        and (n, i, j, 0, 0, t, om) in self.xF_index and (n, i, j, 1, 0, t, om) in self.xF_index and (n, i, j, 2, 0, t, om) in self.xF_index
                        and (n, i, j, 3, 0, t, om) in self.xF_index and (n, i, j, 4, 0, t, om) in self.xF_index),
                        name='4_fuel')
        # Other cont. payloads
        model.addConstrs((xF[n, i, j, c, 1, t, om] == xF[n, i, j, c, 0, t, om]
                        for n in self.N for i in self.I for j in self.I for t in self.T for c in (0, 3) for om in self.omega
                        if (n, i, j, c, 1, t, om) in self.xF_index and (n, i, j, c, 0, t, om) in self.xF_index),
                        name='4_other_cont')   
        return model

def model_results_to_excel(model:Model, output:str) -> None:
    '''
        Print the solved network flow model to an Excel file
    '''
    writer = pd.ExcelWriter(output)
    for n in model.N:
        time_list = []
        for t in model.T:
            node_list = []
            node_keys = []
            for i in model.I:
                for j in model.J:
                    if model.edgeExist[i, j, t]:
                        payload_list = []
                        for p in model.P:
                            if p < len(model.PI):
                                payload_list.append(pd.DataFrame({p: model.xI[n, i, j, p, 0, t].value}, index=[t]))
                            else:
                                payload_list.append(pd.DataFrame({p: model.xF[n, i, j, p-len(model.PI), 0, t].value}, index=[t]))
                        node_keys.append(str([i, j]))
                        node_list.append(pd.concat(payload_list, axis=1))
            time_list.append(pd.concat(node_list, keys=node_keys, axis=1))
        time_df = pd.concat(time_list)
        sheetname = 'Vehicle ' + str(n)
        time_df.to_excel(writer, header=True, sheet_name=sheetname)
    writer.close()
    return
